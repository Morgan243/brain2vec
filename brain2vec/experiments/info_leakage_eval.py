import uuid
import os
import copy
import time
from datetime import datetime
from os.path import join as pjoin

import matplotlib.pyplot
import sklearn.linear_model
import torch
from typing import List, Optional
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from typing import ClassVar, Union, Dict, Optional, Tuple
from dataclasses import field
from dataclasses import make_dataclass

import attr

from mmz import utils

from mmz import experiments as bxp
from mmz import models as zm

from brain2vec import models as bmp
from brain2vec import datasets
from brain2vec import experiments
from dataclasses import dataclass
import json
from simple_parsing import subgroups
#from ecog_speech import result_parsing
from torchdata.datapipes import iter #import IterableWrapper, Mapper, Concater, Sampler
from torchdata.datapipes.map import SequenceWrapper, Concater
from torch.utils.data import RandomSampler
from torch.utils.data import default_collate
from brain2vec.models import base_fine_tuners as ft_models

from brain2vec.datasets import BaseDataset
from brain2vec.datasets.harvard_sentences import HarvardSentencesDatasetOptions, HarvardSentences
from brain2vec.models import base_fine_tuners as bft
from brain2vec.experiments import fine_tune as ft

from mmz.models import copy_model_state

logger = utils.get_logger(__name__)

with_logger = utils.with_logger


class SingleChannelAttacker(torch.nn.Module):
    def __init__(self, input_size, outputs,
                 linear_hidden_n, n_layers,
                 pre_trained_model_output_key,
                 dropout_rate=0., batch_norm=True,
                 auto_eval_mode=True, freeze_pre_train_weights=True):
        super().__init__()
        #self.pre_trained_model = pre_trained_model
        #self.output_model = output_model
        self.input_size = input_size
        self.pre_trained_model_output_key = pre_trained_model_output_key
        #self.pre_trained_model_forward_kws = pre_trained_model_forward_kws
        self.auto_eval_mode = auto_eval_mode
        self.freeze_pre_train_weights = freeze_pre_train_weights
        self.linear_hidden_n = linear_hidden_n
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        self.n_layers = n_layers
        self.outputs = outputs

        def make_linear(outputs, regularize=True, activation=torch.nn.LeakyReLU):
            l = list()
            if regularize and self.dropout_rate > 0.:
                l.append(torch.nn.Dropout(self.dropout_rate))

            l.append(torch.nn.LazyLinear(outputs))
            #torch.nn.init.xavier_uniform_(l[-1].weight)

            if regularize and self.batch_norm:
                l.append(torch.nn.LazyBatchNorm1d(momentum=0.2, track_running_stats=True, affine=True))

            if activation is not None:
                l.append(activation())

            return torch.nn.Sequential(*l)

        self.classifier = torch.nn.Sequential(
            *[make_linear(self.linear_hidden_n) for i in range(self.n_layers - 1)],
            *make_linear(self.outputs, regularize=False, activation=None)
        )

        if self.outputs == 1:
            self.classifier.append(torch.nn.Sigmoid())

    def forward(self, x):
        x_arr = x[self.pre_trained_model_output_key]
        return self.classifier(x_arr.reshape(x_arr.shape[0], self.input_size))


class MultiChannelAttacker(torch.nn.Module):
    def __init__(self, input_shape, hidden_encoder='linear',
                 dropout=0., batch_norm=False, linear_hidden_n=16, n_layers=2,
                 outputs=1):
        super().__init__()
        self.input_shape = input_shape
        #self.c2v_m = c2v_m
        self.outputs = outputs
        self.dropout_rate = dropout
        self.batch_norm = batch_norm

        #self.mc_from_1d = base_ft.MultiChannelFromSingleChannel(self.input_shape, self.c2v_m)

        #self.S = self.input_shape[0]
        #self.T = self.input_shape[1]
        self.S, self.T, self.C = self.input_shape
        #self.T, self.C = self.c2v_m.T, self.c2v_m.C
        self.h_dim = self.S * self.C


        #B, T, S, C = output_arr.shape

        #output_arr_t = output_arr.reshape(B, T, -1)

        #hidden_encoder = 'linear' if hidden_encoder is None else hidden_encoder
        self.hidden_encoder_input = hidden_encoder

        self.classifier_head = torch.nn.Identity()

        def make_linear(outputs, regularize=True, activation=torch.nn.LeakyReLU):
            l = list()
            if regularize and self.dropout_rate > 0.:
                l.append(torch.nn.Dropout(self.dropout_rate))

            l.append(torch.nn.LazyLinear(outputs))
            #torch.nn.init.xavier_uniform_(l[-1].weight)

            if regularize and self.batch_norm:
                l.append(torch.nn.LazyBatchNorm1d(momentum=0.2, track_running_stats=True, affine=True))

            if activation is not None:
                l.append(activation())

            return torch.nn.Sequential(*l)

        if isinstance(hidden_encoder, torch.nn.Module):
            self.hidden_encoder = hidden_encoder
        elif hidden_encoder == 'linear':
            self.lin_dim = self.outputs
            self.hidden_encoder = torch.nn.Sequential(
                *[make_linear(linear_hidden_n) for i in range(n_layers - 1)],
                *make_linear(self.outputs, regularize=False, activation=None)
            )
            if self.outputs == 1:
                self.classifier_head = torch.nn.Sequential(torch.nn.Sigmoid())

            self.feat_arr_reshape = (-1, self.h_dim * self.T)
            #self.classifier_head = torch.nn.Sequential(torch.nn.Sigmoid() if self.outputs == 1
            #                                           else torch.nn.Identity())
        elif hidden_encoder == 'transformer':
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.h_dim, dropout=self.dropout_rate,
                                                             nhead=2, batch_first=True,
                                                             activation="gelu")

            self.hidden_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.feat_arr_reshape = (-1, self.T, self.h_dim)
            self.lin_dim = self.h_dim * self.T

            # h_size = 32
            self.classifier_head = torch.nn.Sequential(*[
                # base.Reshape((self.C * self.T,)),
                # torch.nn.Linear(self.lin_dim, h_size),
                # torch.nn.BatchNorm1d(h_size),
                # torch.nn.LeakyReLU(),
                # torch.nn.Linear(h_size, h_size),
                # torch.nn.LeakyReLU(),
                torch.nn.Linear(self.lin_dim, self.outputs),
                # torch.nn.Sigmoid()
            ])
        else:
            raise ValueError(f"Don't understand hidden_encoder = '{hidden_encoder}'")

    def forward(self, input_d: dict):
        #feat_d = self.mc_from_1d(input_d)
        #feat_arr = feat_d['output']
        if 'output' not in input_d:
            print(input_d.keys())

        feat_arr = input_d['output']

        B = feat_arr.shape[0]

        #trf_arr = feat_arr.reshape(B, self.T, self.h_dim)
        trf_arr = feat_arr.reshape(*self.feat_arr_reshape)
        trf_out_arr = self.hidden_encoder(trf_arr)
        lin_in_arr = trf_out_arr.reshape(B, self.lin_dim)

        return self.classifier_head(lin_in_arr)


@dataclass
class DatasetWithModel:
    p_tuples: List[Tuple]
    reindex_map: Dict
    dataset_cls: BaseDataset

    member_model: torch.nn.Module
    mc_member_model: Optional[torch.nn.Module] = None

    dataset: Optional[HarvardSentences] = None

    def create_mc_model(self):
        self.mc_member_model = bft.MultiChannelFromSingleChannel(input_shape=(
            len(self.dataset.selected_columns), 256, self.member_model.C
        ), model_1d=self.member_model)
        return self

    def split_on_key_level(self, keys,
                           stratify: Optional=None,
                           test_size: Optional = None) -> Tuple['DatasetWithModel', 'DatasetWithModel']:
        assert self.dataset is not None
        split_0, split_1 = self.dataset.split_select_random_key_levels(keys=keys, stratify=stratify, test_size=test_size)

        split_0_dset_with_model = DatasetWithModel(p_tuples=self.p_tuples,
                                                   reindex_map=self.reindex_map,
                                                   dataset_cls=self.dataset_cls,
                                                   member_model=self.member_model,
                                                   mc_member_model=self.mc_member_model,
                                                   dataset=split_0)

        split_1_dset_with_model = DatasetWithModel(p_tuples=self.p_tuples,
                                                   reindex_map=self.reindex_map,
                                                   dataset_cls=self.dataset_cls,
                                                   member_model=self.member_model,
                                                   mc_member_model=self.mc_member_model,
                                                   dataset=split_1)

        return split_0_dset_with_model, split_1_dset_with_model

    def as_feature_extracting_datapipe(self, batch_size, device, multi_channel: bool = True):
        self.mc_member_model = self.mc_member_model.to(device)
        self.feature_model = self.member_model.to(device) if not multi_channel else self.mc_member_model.to(device)

        model_out_dataset = SequenceWrapper(self.dataset).batch(batch_size).map(default_collate)

        if multi_channel:
            def run_mc_model(_d):
                with torch.no_grad():
                    return {_k: _t.to('cpu') if torch.is_tensor(_t) else _t for _k, _t in
                            self.feature_model(
                                {k: t.to(device) for k, t in _d.items()}
                            ).items()
                            }

            model_out_dataset = model_out_dataset.map(run_mc_model).in_memory_cache()
        else:
            def run_sc_model(_d):
                with torch.no_grad():
                    ret_d = {_k: _t.to('cpu') for _k, _t in
                            self.feature_model({k: t.to(device) for k, t in _d.items()},
                                               **dict(features_only=True, mask=False)
                                               ).items()
                             }
                    ret_d['target_arr'] = _d['target_arr']
                    return ret_d

            model_out_dataset = model_out_dataset.map(run_sc_model).in_memory_cache()

        model_out_dataset.selected_columns = self.dataset.selected_columns
        model_out_dataset.get_target_shape = self.dataset.get_target_shape
        model_out_dataset.get_target_labels = self.dataset.get_target_labels
        return model_out_dataset

    def patch_attributes(self, obj):
        obj.selected_columns = self.dataset.selected_columns
        obj.get_target_shape = self.dataset.get_target_shape
        obj.get_target_labels = self.dataset.get_target_labels
        return obj


class DatasetWithModelBaseTask(bxp.TaskOptions):
    @classmethod
    def load_and_check_pretrain_opts(cls, results, expected_dataset_name, dataset_cls, expected_n_pretrain_sets=2) -> List[Tuple]:
        sm_member_dataset_opts = results['dataset_options']
        assert sm_member_dataset_opts['dataset_name'] == expected_dataset_name
        pretrain_sets = dataset_cls.make_tuples_from_sets_str(sm_member_dataset_opts['train_sets'])
        assert len(pretrain_sets) == 2
        return pretrain_sets

    @classmethod
    def load_pretrained_model_results(cls,
                                      pretrained_result_input_path: str = None,
                                      pretrained_result_model_base_path: str = None,
                                      device=None):

        assert_err = "pretrained_result_input_path must be provided"
        assert pretrained_result_input_path is not None, assert_err

        from brain2vec.experiments import load_model_from_results

        if pretrained_result_model_base_path is None:
            pretrained_result_model_base_path = os.path.join(
                os.path.split(pretrained_result_input_path)[0], 'models')

        print(f"Loading pretrained model from results in {pretrained_result_input_path}"
              f" (base path = {pretrained_result_model_base_path})")
        with open(pretrained_result_input_path, 'r') as f:
            result_json = json.load(f)

        print(f"\tKEYS: {list(sorted(result_json.keys()))}")
        if 'model_name' not in result_json:
            logger.info("MODEL NAME MISSING - setting to cog2vec")
            result_json['model_name'] = 'cog2vec'

        if 'cv' in result_json:
            min_pretrain_bce = min(result_json['cv']['bce_loss'])
            logger.info(f"MIN PRETRAINED BCE LOSS: {min_pretrain_bce}")
        else:
            logger.info(f"Pretraining result didn't have a 'cv' to check the loss of ... D:")

        pretrained_model = load_model_from_results(result_json, base_model_path=pretrained_result_model_base_path)
        if device is not None:
            pretrained_model = pretrained_model.to(device)

        return pretrained_model, result_json

    def load_one_dataset_with_model(self, dset_w_model: DatasetWithModel, sensor_columns='valid'):
        base_kws = dict(pre_processing_pipeline=self.dataset.pre_processing_pipeline,
                        data_subset=self.dataset.data_subset,
                        label_reindex_col=self.dataset.label_reindex_col,
                        extra_output_keys=self.dataset.extra_output_keys.split(',')
                        if self.dataset.extra_output_keys is not None else None,
                        flatten_sensors_to_samples=self.dataset.flatten_sensors_to_samples)
        kws = dict(
            patient_tuples=dset_w_model.p_tuples,
            label_reindex_map=dset_w_model.reindex_map,
            sensor_columns=sensor_columns,
            **base_kws
        )
        dset_w_model.dataset = dset_w_model.dataset_cls(**kws)
        dset_w_model.create_mc_model()
        return dset_w_model


@with_logger
@dataclass
class OneModelMembershipInferenceFineTuningTask(DatasetWithModelBaseTask):
    task_name: str = "membership_inference_classification_fine_tuning"

    pretrained_target_model_result_input: bxp.ResultInputOptions = None
    dataset: datasets.DatasetOptions = HarvardSentencesDatasetOptions(train_sets='AUTO-PRETRAINING',
                                                                      flatten_sensors_to_samples=False,
                                                                      label_reindex_col='patient',
                                                                      split_cv_from_test=False,
                                                                      pre_processing_pipeline='random_sample')
    method: str = '2d_linear'
    weight_decay: float = 0.

    squeeze_target: ClassVar[bool] = True

    dataset_with_model_d: dict = field(default=None, init=False)

    def make_criteria_and_target_key(self):
        criterion = torch.nn.BCELoss()
        target_key = 'target_arr'
        return criterion, target_key

    def get_member_model_output_shape(self) -> Tuple:
        member_model = self.target_model
        return member_model.T, member_model.C

    def make_dataset_and_loaders(self, **kws):
        dataset_cls = datasets.BaseDataset.get_dataset_by_name(self.dataset.dataset_name)
        all_sets = dataset_cls.make_tuples_from_sets_str('*')

        self.target_model, self.target_model_results = self.load_pretrained_model_results(
            pretrained_result_input_path=self.pretrained_target_model_result_input.result_file,
            pretrained_result_model_base_path=self.pretrained_target_model_result_input.model_base_path
        )

        # Assumes set to AUTO-PRETRAINING, so the train tuples are all the sets that the loaded model was pretrained on
        pretrain_train_sets_opt = self.target_model_results['dataset_options']['train_sets']
        all_true_pos_set = dataset_cls.make_tuples_from_sets_str(pretrain_train_sets_opt)

        # Thus, the sets not pretrained on are the true negatives
        all_true_neg_set = list(set(all_sets) - set(all_true_pos_set))
        logger.info(f"True neg set of length {len(all_true_neg_set)} contains: {all_true_neg_set}")

        #debias_train_pos_set = all_true_pos_set

        # Determine the test set's single positive participant set - the attacker will not train on this member
        # Was a test set specified in the options directly?
        if self.dataset.test_sets is not None and self.dataset.test_sets != '':
            # If an integer
            if self.dataset.test_sets.isdigit():
                true_pos_ix_of_test = [int(self.dataset.test_sets)]
                self.dataset.test_sets = None
            else:
                raise ValueError(f"Dont understand test set - it wasn't an integet: {self.dataset.test_sets}")
        # No test set specified, so select a random one from true positive set
        elif self.dataset.test_sets is None:
            true_pos_ix_of_test = np.random.choice(len(all_true_pos_set), 1).tolist()
        else:
            raise ValueError(f'Dont understand test set: {self.dataset.test_sets}')

        assert all(_ix < len(all_true_pos_set) for _ix in true_pos_ix_of_test), "Test positive outside ix range of pos"
        assert all(_ix >= 0 for _ix in true_pos_ix_of_test), "Test pos is negative...???"

        # Pull out the selected true positive tuple to be in the test set
        debias_test_pos_set = [all_true_pos_set[_ix] for _ix in true_pos_ix_of_test]
        # Training positives are all the positives that are not in the test set
        debias_train_pos_set = [t for t in all_true_pos_set if t not in debias_test_pos_set]

        # Take half of true negatives (i.e. not seen in pretraining procedure
        true_neg_ix_of_test = np.random.choice(len(all_true_neg_set), len(all_true_neg_set) // 2).tolist()
        debias_test_neg_set = [all_true_neg_set[_ix] for _ix in true_neg_ix_of_test]
        debias_train_neg_set = [t for t in all_true_neg_set if t not in debias_test_neg_set]

        train_label_reindex_map = dict()
        train_label_reindex_map.update({t[1]: 0 for t in debias_train_neg_set})
        train_label_reindex_map.update({t[1]: 1 for t in debias_train_pos_set})
        logger.info(f"Train label reindex map created: {train_label_reindex_map}")

        test_label_reindex_map = dict()
        test_label_reindex_map.update({t[1]: 0 for t in debias_test_neg_set})
        test_label_reindex_map.update({t[1]: 1 for t in debias_test_pos_set})
        logger.info(f"Test label reindex map created: {test_label_reindex_map}")

        sensor_columns = kws.pop('sensor_columns', 'valid')
        train_dataset_with_model = DatasetWithModel(
            p_tuples=debias_train_neg_set + debias_train_pos_set,
            reindex_map=train_label_reindex_map,
            dataset_cls=HarvardSentences,
            member_model=self.target_model
        )
        train_dataset_with_model = self.load_one_dataset_with_model(train_dataset_with_model,
                                                                    sensor_columns=sensor_columns)
        seleceted_sensor_columns = train_dataset_with_model.dataset.selected_columns

        train_dataset_with_model, cv_dataset_with_model = train_dataset_with_model.split_on_key_level(
            keys=('patient', 'sent_code'),
            #stratify='label',
            test_size=0.25)

        train_datapipe = train_dataset_with_model.as_feature_extracting_datapipe(
            self.dataset.batch_size,
            self.device, '2d' in self.method)
        train_datapipe = train_dataset_with_model.patch_attributes(train_datapipe)


        cv_datapipe = cv_dataset_with_model.as_feature_extracting_datapipe(
            self.dataset.batch_size_eval if self.dataset.batch_size_eval is not None else self.dataset.batch_size,
            self.device, '2d' in self.method)
        cv_datapipe = cv_dataset_with_model.patch_attributes(cv_datapipe)

        # ---
        test_dataset_with_model = DatasetWithModel(
            p_tuples=debias_test_neg_set + debias_test_pos_set,
            reindex_map=test_label_reindex_map,
            dataset_cls=HarvardSentences,
            member_model=self.target_model
        )
        test_dataset_with_model = self.load_one_dataset_with_model(test_dataset_with_model,
                                                                   sensor_columns=seleceted_sensor_columns)

        test_datapipe = test_dataset_with_model.as_feature_extracting_datapipe(
            self.dataset.batch_size_eval if self.dataset.batch_size_eval is not None else self.dataset.batch_size,
            self.device, '2d' in self.method)
        test_datapipe = test_dataset_with_model.patch_attributes(test_datapipe)
        # --- End Test data pipe setup ---


        self.dataset_with_model_d = dict(train=train_dataset_with_model,
                                         cv=cv_dataset_with_model,
                                         test=test_dataset_with_model)
        ret_datapipe_d = dict(train=train_datapipe,
                              cv=cv_datapipe,
                              test=test_datapipe)

        return dict(ret_datapipe_d), dict(ret_datapipe_d), dict(ret_datapipe_d)


@with_logger
@dataclass
class ShadowClassifierMembershipInferenceFineTuningTask(DatasetWithModelBaseTask):
    task_name: str = "membership_inference_classification_fine_tuning"
    pretrained_shadow_model_result_input_0: bxp.ResultInputOptions = None
    pretrained_shadow_model_result_input_1: bxp.ResultInputOptions = None
    pretrained_target_model_result_input: bxp.ResultInputOptions = None

    dataset: datasets.DatasetOptions = HarvardSentencesDatasetOptions(train_sets='AUTO-PRETRAINING',
                                                                      flatten_sensors_to_samples=False,
                                                                      label_reindex_col='patient',
                                                                      split_cv_from_test=False,
                                                                      pre_processing_pipeline='random_sample')
    method: str = '2d_linear'
    weight_decay: float = 0.

    squeeze_target: ClassVar[bool] = True

    dataset_with_model_d: dict = field(default=None, init=False)

    def make_criteria_and_target_key(self):
        criterion = torch.nn.BCELoss()
        target_key = 'target_arr'
        return criterion, target_key

    def load_models(self):
        # Load a "shadow" model (M_s) that was pretrained on 2 patients (M_s0, M_s1)
        # Load a "target" model (M_t) that was pretrained on 2 patients (M_t0, M_t1)
        #  |-> This leaves 3 patients unseen (u0, u1, u2)
        # Train adv. on M_s0 vs. u0
        # CV adv. on M_s1 vs u1
        # Test Adv on M_t0,1 vs. u2 (downsample M_t0,1)
        #  |-> The imbalance, while not great, at least simulates real-world scenario well
        self.shadow_model_results_map = dict()
        self.shadow_model_map = dict()

        # ---
        # Load models
        self.logger.info('--------------------')
        self.logger.info("Shadow model 0: " + str(self.pretrained_shadow_model_result_input_0))
        self.shadow_model_map[0], self.shadow_model_results_map[0] = self.load_pretrained_model_results(
            self.pretrained_shadow_model_result_input_0.result_file,
            self.pretrained_shadow_model_result_input_0.model_base_path,
            device=self.device
        )

        self.logger.info('--------------------')
        self.logger.info("Shadow model 1: " + str(self.pretrained_shadow_model_result_input_1))
        self.shadow_model_map[1], self.shadow_model_results_map[1] = self.load_pretrained_model_results(
                self.pretrained_shadow_model_result_input_1.result_file,
                self.pretrained_shadow_model_result_input_1.model_base_path,
                device=self.device
            )

        self.logger.info('--------------------')
        self.logger.info("Target model: " + str(self.pretrained_target_model_result_input))
        self.target_model, self.target_model_results = self.load_pretrained_model_results(
            self.pretrained_target_model_result_input.result_file,
            self.pretrained_target_model_result_input.model_base_path,
            device=self.device
        )
        assert len(self.shadow_model_results_map) == 2

    def get_member_model_output_shape(self) -> Tuple:
        member_model = self.shadow_model_map[0]
        return member_model.T, member_model.C

    def make_dataset_and_loaders(self, **kws):
        self.load_models()

        # We'll use the CLI options to set the expected dataset - later assert that all models used same dataset
        dataset_cls = datasets.BaseDataset.get_dataset_by_name(self.dataset.dataset_name)
        all_sets = dataset_cls.make_tuples_from_sets_str('*')

        sm_pretrain_sets_d = {k: self.load_and_check_pretrain_opts(res, self.dataset.dataset_name, dataset_cls)
                              for k, res in self.shadow_model_results_map.items()}

        # -------------
        # Setup training data pipe for attacker
        shadow_training_sets_with_model: Dict[int, DatasetWithModel] = dict()

        for ii, (sm_k, sm_pretrained_model) in enumerate(self.shadow_model_map.items()):
            # sm_k and ii should be the same
            shadow_training_sets_with_model[ii] = DatasetWithModel(
                # Take the first pretrain set from each pretrained model
                p_tuples=[pretrain_tuples[ii] for _sm_k, pretrain_tuples in sm_pretrain_sets_d.items()],
                # Pretraining participants from this model are the positive class (member)
                reindex_map={pretrain_tuples[ii][1]: 1 if sm_k == _sm_k else 0
                             for _sm_k, pretrain_tuples in sm_pretrain_sets_d.items()},
                dataset_cls=dataset_cls,
                member_model=sm_pretrained_model
            )
        assert len(shadow_training_sets_with_model) == 2
        shadow_training_sets_with_model[0] = self.load_one_dataset_with_model(shadow_training_sets_with_model[0],
                                                                              sensor_columns=kws.get('sensor_columns',
                                                                                                     'valid'))

        selected_columns = shadow_training_sets_with_model[0].dataset.selected_columns
        shadow_training_sets_with_model[1] = self.load_one_dataset_with_model(shadow_training_sets_with_model[1],
                                                                              sensor_columns=selected_columns)

        train_pipes = [s.as_feature_extracting_datapipe(self.dataset.batch_size, self.device, '2d' in self.method)
                       for k, s in shadow_training_sets_with_model.items()]

        training_datapipe = Concater(*train_pipes).shuffle()
        training_datapipe = shadow_training_sets_with_model[0].patch_attributes(training_datapipe)
        # --- End Training data pipe setup ---

        # -------------
        # -- Setup CV
        shadow_cv_sets_with_model: Dict[int, DatasetWithModel] = dict()
        for ii, (sm_k, sm_pretrained_model) in enumerate(reversed(self.shadow_model_map.items())):
            # sm_k and ii should be the same
            shadow_cv_sets_with_model[ii] = DatasetWithModel(
                # Take the first pretrain set from each pretrained model
                p_tuples=[pretrain_tuples[ii] for _sm_k, pretrain_tuples in sm_pretrain_sets_d.items()],
                # Pretraining participants from this model are the positive class (member)
                reindex_map={pretrain_tuples[ii][1]: 1 if sm_k == _sm_k else 0
                             for _sm_k, pretrain_tuples in sm_pretrain_sets_d.items()},
                dataset_cls=dataset_cls,
                member_model=sm_pretrained_model
            )
        assert len(shadow_cv_sets_with_model) == 2
        shadow_cv_sets_with_model[0] = self.load_one_dataset_with_model(shadow_cv_sets_with_model[0], selected_columns)
        shadow_cv_sets_with_model[1] = self.load_one_dataset_with_model(shadow_cv_sets_with_model[1], selected_columns)

        cv_datapipe = Concater(*[s.as_feature_extracting_datapipe(self.dataset.batch_size, self.device,
                                                                  '2d' in self.method)
                                 for k, s in shadow_cv_sets_with_model.items()]).shuffle()
        cv_datapipe = shadow_cv_sets_with_model[0].patch_attributes(cv_datapipe)
        # --- End CV data pipe setup ---

        # -------------
        # -- Set up target as the test set
        target_pretrain_sets = self.load_and_check_pretrain_opts(self.target_model_results, self.dataset.dataset_name,
                                                                 dataset_cls)
        # Sets used by either shadow models or target model
        used_sets = target_pretrain_sets + list(sum(sm_pretrain_sets_d.values(), list()))
        # Sets that were not used in any of the pretrained models, shadow or target
        unused_sets = [s for s in all_sets if s not in used_sets]
        # Test's positive are from the last models pretrain sets
        test_reindex_map = {pt[1]: 1 for pt in target_pretrain_sets}
        # Test's negative are from the remaining patient
        test_reindex_map.update({pt[1]: 0 for pt in unused_sets})

        shadow_test_set_with_model = DatasetWithModel(
            p_tuples=target_pretrain_sets + unused_sets,
            reindex_map=test_reindex_map,
            dataset_cls=dataset_cls,
            member_model=self.target_model
        )
        shadow_test_set_with_model = self.load_one_dataset_with_model(shadow_test_set_with_model, selected_columns)
        test_datapipe = shadow_test_set_with_model.as_feature_extracting_datapipe(
            self.dataset.batch_size_eval if self.dataset.batch_size_eval is not None else self.dataset.batch_size,
            self.device, '2d' in self.method)
        test_datapipe = shadow_test_set_with_model.patch_attributes(test_datapipe)
        # --- End Test data pipe setup ---

        self.dataset_with_model_d = dict(train=shadow_training_sets_with_model,
                                         cv=shadow_cv_sets_with_model,
                                         test=shadow_test_set_with_model)
        ret_datapipe_d = dict(train=training_datapipe, cv=cv_datapipe, test=test_datapipe)

        return dict(ret_datapipe_d), dict(ret_datapipe_d), dict(ret_datapipe_d)


@with_logger
@dataclass
class InfoLeakageExperiment(bxp.Experiment):
    task: ShadowClassifierMembershipInferenceFineTuningTask = subgroups(
        {'shadow_classifier_mi': ShadowClassifierMembershipInferenceFineTuningTask,
         'one_model_mi': OneModelMembershipInferenceFineTuningTask
         },
        default='shadow_classifier_mi')

    attacker_model: ft.FineTuningModel = ft.FineTuningModel()

    # ---
    result_model: torch.nn.Module = field(default=None, init=False)

    shadow_model_results_map: dict = field(default=None, init=False)
    shadow_model_map: dict = field(default=None, init=False)
    target_model: dict = field(default=None, init=False)
    target_labels: dict = field(default=None, init=False)
    target_model_results: dict = field(default=None, init=False)
    dataset_map: dict = field(default=None, init=False)
    dl_map: dict = field(default=None, init=False)
    eval_dl_map: dict = field(default=None, init=False)

    outputs_map: dict = field(default=None, init=False)
    output_cm_map: dict = field(default=None, init=False)
    performance_map: dict = field(default=None, init=False)
    clf_str_map: dict = field(default=None, init=False)

    def initialize(self):
        if getattr(self, 'initialized', False):
            return self

        #self.load_models()

        # --
        # Load Datasets
        self.dataset_map, self.dl_map, self.eval_dl_map = self.task.make_dataset_and_loaders(
            sensor_columns=list(range(64))
        )

        n_sensors = len(self.dataset_map['train'].selected_columns)
        self.target_labels = self.dataset_map['train'].get_target_labels()

        #member_model = self.task.shadow_model_map[0]
        T, C = self.task.get_member_model_output_shape()
        if '2d' in self.attacker_model.fine_tuning_method:
            attack_arr_shape = T, n_sensors, C
            self.model_kws = dict(input_shape=attack_arr_shape)
            self.result_model = MultiChannelAttacker(**self.model_kws)
        else:
            attack_arr_shape =T * C
            self.model_kws = dict(
                input_size=attack_arr_shape, outputs=1,
                linear_hidden_n=self.attacker_model.linear_hidden_n,
                n_layers=self.attacker_model.n_layers,
                dropout_rate=self.attacker_model.dropout,
                batch_norm=self.attacker_model.batch_norm,
                # pre_trained_model_forward_kws = dict(features_only=True, mask=False),
                pre_trained_model_output_key='x'
            )
            self.result_model = SingleChannelAttacker(**self.model_kws)

        criterion, target_key = self.task.make_criteria_and_target_key()

        logger.info(f"Criterion for {self.task.task_name}: {criterion} on {target_key}")

        self.trainer = ShadowClassifierTrainer(model_map=dict(model=self.result_model), opt_map=dict(),
                                               train_data_gen=self.dl_map['train'],
                                               cv_data_gen=self.eval_dl_map.get('cv'),
                                               input_key='signal_arr',
                                               learning_rate=self.task.learning_rate,
                                               early_stopping_patience=self.task.early_stopping_patience,
                                               lr_adjust_on_cv_loss=self.task.lr_adjust_patience is not None,
                                               lr_adjust_on_plateau_kws=dict(patience=self.task.lr_adjust_patience,
                                                                 factor=self.task.lr_adjust_factor),
                                               weight_decay=self.task.weight_decay,
                                               target_key=target_key,
                                               criterion=criterion,
                                               device=self.task.device,
                                               squeeze_target=self.task.squeeze_target,
                                               cache_dataloaders=False
                                               )
        self.initialized = True
        return self

    def train(self):
        self.attacker_model_train_results = self.trainer.train(self.task.n_epochs)

    def eval(self):
        outputs_map = self.trainer.generate_outputs(**self.eval_dl_map)
        class_val_to_label_d, class_labels = self.dataset_map['train'].get_target_labels()
        from sklearn.metrics import confusion_matrix
        output_cm_map = dict()
        for part_name, out_d in outputs_map.items():
            preds_arr: np.ndarray = out_d['preds'].squeeze()
            if preds_arr.ndim == 1:
                n_preds = 1 - preds_arr
                preds_arr = np.concatenate([n_preds.reshape(-1, 1), preds_arr.reshape(-1, 1)], axis=1)
            preds_df = pd.DataFrame(preds_arr, columns=class_labels)
            y_s = pd.Series(out_d['actuals'].reshape(-1), index=preds_df.index, name='actuals')
            y_label_s = y_s.map(class_val_to_label_d)
            cm = confusion_matrix(y_label_s, preds_df.idxmax(1), labels=preds_df.columns.tolist())
            output_cm_map[part_name] = pd.DataFrame(cm, columns=preds_df.columns, index=preds_df.columns).to_json()

        performance_map = dict()
        clf_str_map = None
        if self.task.dataset.dataset_name == 'hvs':
            target_shape = self.dataset_map['train'].get_target_shape()

            kws = dict(threshold=(0.5 if target_shape == 1 else None))
            clf_str_map = utils.make_classification_reports(outputs_map, **kws)
            if target_shape == 1:
                performance_map = {part_name: utils.performance(outputs_d['actuals'],
                                                                outputs_d['preds'] > 0.5)
                                   for part_name, outputs_d in outputs_map.items()}
            else:
                performance_map = {part_name: utils.multiclass_performance(outputs_d['actuals'],
                                                                           outputs_d['preds'].argmax(1))
                                   for part_name, outputs_d in outputs_map.items()}
            print(performance_map)

        self.outputs_map, self.output_cm_map, self.performance_map, self.clf_str_map = [outputs_map, output_cm_map,
                                                                                        performance_map, clf_str_map]

    def run(self):
        self.initialize()
        self.train()
        self.eval()
        self.save()

    def save(self):
        #####
        # Prep a results structure for saving - everything must be json serializable (no array objects)
        res_dict = self.create_result_dictionary(
            model_name=self.attacker_model.model_name,
            epoch_outputs=self.attacker_model_train_results,
            train_selected_columns=self.dataset_map['train'].selected_columns,  # dataset_map['train'].selected_columns,
            #test_selected_flat_indices=dataset_map['test'].selected_flat_indices,
            #selected_flat_indices={k: d.selected_flat_indices for k, d in dataset_map.items()},
            selected_flat_indices={k: d.selected_levels_df.to_json() if hasattr(d, 'selected_levels_df') else None
                                   for k, d in self.dataset_map.items()},
            best_model_epoch=getattr(self.trainer, 'best_model_epoch', None),
            num_trainable_params=utils.number_of_model_params(self.result_model),
            num_params=utils.number_of_model_params(self.result_model, trainable_only=False),
            model_kws=self.model_kws,
            classification_summaries=self.clf_str_map,
            **self.performance_map,
            #**eval_res_map,
            #pretrained_result_input=vars(self.pretrained_result_input),
            experiment_options=vars(self),
            task_options=vars(self.task),
            dataset_options=vars(self.task.dataset),
            #auto_selected_train_sets=getattr(self, 'auto_selected_train_sets'),
            #dataset_options=vars(self.dataset),
            result_output_options=vars(self.result_output),
            model_options=vars(self.attacker_model),
            #pretrained_results=self.pretraining_results,
            confusion_matrices=self.output_cm_map
        )
        uid = res_dict['uid']
        name = res_dict['name']

        self.save_results(self.result_model, result_file_name=name, result_output=self.result_output,
                          model_file_name=uid, res_dict=res_dict)

        return self.trainer, self.performance_map



@attr.s
class ShadowClassifierTrainer(bmp.Trainer):
    input_key = attr.ib('signal_arr')
    squeeze_target = attr.ib(False)

    squeeze_first = True


    def loss(self, model_output_d, input_d, as_tensor=True):
        target = (input_d[self.target_key].squeeze() if self.squeeze_target
                                   else input_d[self.target_key])
        if isinstance(self.criterion, (torch.nn.BCEWithLogitsLoss, torch.nn.BCELoss)):
            target = target.float()

        model_output_d = model_output_d.reshape(*target.shape)

        crit_loss = self.criterion(model_output_d.float() if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss) else model_output_d,
                                   target)
        return crit_loss

    def _eval(self, epoch_i, dataloader, model_key='model'):
        """
        trainer's internal method for evaluating losses,
        snapshotting best models and printing results to screen
        """
        model = self.model_map[model_key].eval()
        self.best_cv = getattr(self, 'best_cv', np.inf)

        preds_l, actuals_l, loss_l = list(), list(), list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):
                    input_d = {k: v.to(self.device) for k, v in _x.items()}
                    #input_arr = input_d[self.input_key]
                    #actual_arr = input_d[self.target_key]
                    m_output = model(input_d)

                    #actuals = input_d[self.target_key]

                    #loss = self.criterion(preds, actuals)
                    loss = self.loss(m_output, input_d)

                    loss_l.append(loss.detach().cpu().item())

                    pbar.update(1)

                mean_loss = np.mean(loss_l)
                desc = "Mean Eval Loss: %.5f" % mean_loss
                reg_l = 0.
                if self.model_regularizer is not None:
                    reg_l = self.model_regularizer(model)
                    desc += (" (+ %.6f reg loss = %.6f)" % (reg_l, mean_loss + reg_l))

                overall_loss = (mean_loss + reg_l)

                if overall_loss < self.best_cv:

                    self.best_model_state = copy_model_state(model)
                    self.best_model_epoch = epoch_i
                    self.best_cv = overall_loss
                    desc += "[[NEW BEST]]"

                pbar.set_description(desc)

        self.model_map['model'].train()
        return dict(primary_loss=overall_loss, cv_losses=loss_l)

    def train_inner_step(self, epoch_i, data_batch):
        """
        Core training method - gradient descent - provided the epoch number and a batch of data and
        must return a dictionary of losses.
        """
        res_d = dict()

        model = self.model_map['model'].to(self.device)
        optim = self.opt_map['model']
        model = model.train()

        model.zero_grad()
        optim.zero_grad()

        input_d = {k: v.to(self.device) for k, v in data_batch.items()}
        #input_d = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data_batch.items()}
        #input_arr = input_d[self.input_key]
        #actual_arr = input_d[self.target_key]
        #m_output = model(input_arr)
        #input_d = self.train_pretrained_model(input_d)
        m_output = model(input_d)

        #crit_loss = self.criterion(m_output, actual_arr)
        #crit_loss = self.loss(m_output, actual_arr)
        crit_loss = self.loss(m_output, input_d)
        res_d['crit_loss'] = crit_loss.detach().cpu().item()

        if self.model_regularizer is not None:
            reg_l = self.model_regularizer(model)
            res_d['bwreg'] = reg_l.detach().cpu().item()
        else:
            reg_l = 0

        loss = crit_loss + reg_l
        res_d['total_loss'] = loss.detach().cpu().item()
        loss.backward()
        optim.step()
        model = model.eval()
        return res_d

    def generate_outputs_from_model_inner_step(self, model, data_batch, criterion=None,
                                               input_key='signal_arr', target_key='text_arr', device=None,
                                               ):
        #X = data_batch[input_key].to(device)

        #if self.squeeze_first:
        #    X = X.squeeze()

        with torch.no_grad():
            model.eval()
            model.to(device)
            input_d = {k: v.to(self.device) for k, v in data_batch.items()}
            preds = model(input_d)

            #loss_v = self.loss(preds, input_d)
        #score_d = self.score_training(m_d, as_tensor=True)
        eval_d = dict(preds=preds, actuals=input_d[self.target_key], #loss=torch.tensor(loss_v)
                      )#dict(**score_d, **loss_d)

        return eval_d



if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser(description="Information Leakage Experiments")
    # parser.add_arguments("--pretrained_inspection_plots", action='store_true', default=False)
    parser.add_arguments(InfoLeakageExperiment, dest='info_leakage')
    # parser.add_arguments(TransferLearningOptions, dest='transfer_learning')
    # parser.add_arguments(TransferLearningResultParsingOptions, dest='tl_result_parsing')
    args = parser.parse_args()
    tl: InfoLeakageExperiment = args.info_leakage
    tl.run()
