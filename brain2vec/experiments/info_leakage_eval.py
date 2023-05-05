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
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from dataclasses import make_dataclass

import attr
from joblib import Parallel, parallel_backend, delayed

from mmz import utils

from mmz import experiments as bxp
from mmz import models as zm

from brain2vec import models as bmp
from brain2vec import datasets
from brain2vec import experiments
from dataclasses import dataclass
import json
from simple_parsing import subgroups
from torchdata.datapipes import iter as td_iter
from torchdata.datapipes.map import SequenceWrapper, Concater
from torch.nn import functional as F
from torch.utils.data import RandomSampler
from torch.utils.data import default_collate
from brain2vec.models import base_fine_tuners as ft_models

from brain2vec.datasets import BaseDataset
from brain2vec.datasets.harvard_sentences import (
    HarvardSentencesDatasetOptions,
    HarvardSentences,
)
from brain2vec.models import base_fine_tuners as bft
from brain2vec.experiments import fine_tune as ft

from mmz.models import copy_model_state
from pathlib import Path


logger = utils.get_logger(__name__)

with_logger = utils.with_logger


class SingleChannelAttacker(torch.nn.Module):
    def __init__(
        self,
        input_size,
        outputs,
        linear_hidden_n,
        n_layers,
        pre_trained_model_output_key,
        dropout_rate=0.0,
        batch_norm=True,
        auto_eval_mode=True,
        freeze_pre_train_weights=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.pre_trained_model_output_key = pre_trained_model_output_key
        self.auto_eval_mode = auto_eval_mode
        self.freeze_pre_train_weights = freeze_pre_train_weights
        self.linear_hidden_n = linear_hidden_n
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        self.n_layers = n_layers
        self.outputs = outputs

        self.classifier = torch.nn.Sequential(
            *[
                zm.LinearBlock.make_linear_block(
                    self.linear_hidden_n,
                    dropout_rate=self.dropout_rate,
                    batch_norm=batch_norm,
                )
                for i in range(self.n_layers - 1)
            ],
            *zm.LinearBlock.make_linear_block(
                self.outputs,
                dropout_rate=self.dropout_rate,
                batch_norm=batch_norm,
            ),
        )

        if self.outputs == 1:
            self.classifier.append(torch.nn.Sigmoid())
        else:
            self.classifier.append(torch.nn.Softmax(dim=1))

    def forward(self, x):
        x_arr = x[self.pre_trained_model_output_key]
        return self.classifier(x_arr.reshape(x_arr.shape[0], self.input_size))


class MultiChannelAttacker(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_encoder="linear",
        dropout=0.0,
        batch_norm=False,
        linear_hidden_n=16,
        n_layers=2,
        outputs=1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.outputs = outputs
        self.dropout_rate = dropout
        self.batch_norm = batch_norm

        self.n_layers = n_layers
        self.n_channels, self.n_times, self.n_embed = self.input_shape
        self.h_dim = self.n_channels * self.n_times * self.n_embed
        self.linear_hidden_n = linear_hidden_n

        self.hidden_encoder_input = hidden_encoder

        self.classifier_head = torch.nn.Identity()

        if isinstance(hidden_encoder, torch.nn.Module):
            print(foo)
            self.hidden_encoder = hidden_encoder
        elif hidden_encoder == "linear":
            self.lin_dim = self.outputs
            self.hidden_encoder = torch.nn.Sequential(
                *[
                    zm.LinearBlock.make_linear_block(
                        self.linear_hidden_n,
                        dropout_rate=self.dropout_rate,
                        batch_norm=batch_norm,
                    )
                    for i in range(self.n_layers - 1)
                ],
                *zm.LinearBlock.make_linear_block(
                    self.outputs,
                    dropout_rate=self.dropout_rate,
                    batch_norm=batch_norm,
                ),
            )

            # self.hidden_encoder = torch.nn.Sequential(
            #    *[make_linear(linear_hidden_n) for i in range(n_layers - 1)],
            #    *make_linear(self.outputs, regularize=False, activation=None)
            # )
            if self.outputs == 1:
                self.classifier_head = torch.nn.Sequential(torch.nn.Sigmoid())

            self.feat_arr_reshape = (-1, self.h_dim)
            # self.classifier_head = torch.nn.Sequential(torch.nn.Sigmoid() if self.outputs == 1
            #                                           else torch.nn.Identity())
        elif hidden_encoder == "transformer":
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=self.h_dim,
                dropout=self.dropout_rate,
                nhead=2,
                batch_first=True,
                activation="gelu",
            )

            self.hidden_encoder = torch.nn.TransformerEncoder(
                encoder_layer, num_layers=1
            )
            self.feat_arr_reshape = (-1, self.T, self.h_dim)
            self.lin_dim = self.h_dim * self.T

            # h_size = 32
            self.classifier_head = torch.nn.Sequential(
                *[
                    torch.nn.Linear(self.lin_dim, self.outputs),
                ]
            )
        else:
            raise ValueError("Don't understand hidden_encoder " f"= '{hidden_encoder}'")

    def forward(self, input_d: dict):
        if "output" not in input_d:
            print(input_d.keys())

        feat_arr = input_d["output"]

        B = feat_arr.shape[0]

        trf_arr = feat_arr.reshape(*self.feat_arr_reshape)
        trf_out_arr = self.hidden_encoder(trf_arr)
        lin_in_arr = trf_out_arr.reshape(B, self.lin_dim)

        return self.classifier_head(lin_in_arr)


@dataclass
class DatasetWithModel:
    p_tuples: List[Tuple]
    reindex_map: Optional[Dict]
    dataset_cls: HarvardSentences

    member_model: torch.nn.Module
    mc_member_model: Optional[torch.nn.Module] = None
    mc_n_channels_to_sample: Optional[int] = None

    dataset: Optional[HarvardSentences] = None
    model_output_key: str = "x"  # X or 'features'
    features_only: bool = True
    mask: bool = False
    output_loss: bool = False

    target_shape: int = 2

    def create_mc_model(self):
        if self.mc_n_channels_to_sample is None:
            assert self.dataset.selected_columns is not None, (
                "mc_n_channels_to_sample must be set if dataset doesn't "
                "have fixed number of sensors"
            )
            n_channels = len(self.dataset.selected_columns)
        else:
            assert isinstance(self.mc_n_channels_to_sample, int), (
                "mc_n_channels_to_sample must be int, "
                f"got {type(self.mc_n_channels_to_sample)}"
            )
            msg = f"mc_n_channels_to_sample = {self.mc_n_channels_to_sample}"
            assert self.mc_n_channels_to_sample > 0, msg
            n_channels = self.mc_n_channels_to_sample

        self.mc_member_model = bft.MultiChannelFromSingleChannel(
            input_shape=(n_channels, 256, self.member_model.C),
            model_1d=self.member_model,
            model_output_key=self.model_output_key,
            forward_kws=dict(features_only=True, mask=False),
        )
        return self

    def split_on_key_level(
            self,
            keys,
            # These are passed eventually to sklearn's train_test_split
            stratify: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            test_size: Optional[Union[int, float]] = None
            ) -> Tuple["DatasetWithModel", "DatasetWithModel"]:
        assert self.dataset is not None
        split_0, split_1 = self.dataset.split_select_random_key_levels(
                keys=keys, stratify=stratify, test_size=test_size
                )

        split_0_dset_with_model = DatasetWithModel(
            p_tuples=self.p_tuples,
            reindex_map=self.reindex_map,
            dataset_cls=self.dataset_cls,
            member_model=self.member_model,
            mc_member_model=self.mc_member_model,
            mc_n_channels_to_sample=self.mc_n_channels_to_sample,
            model_output_key=self.model_output_key,
            target_shape=self.target_shape,
            dataset=split_0,
        )

        split_1_dset_with_model = DatasetWithModel(
            p_tuples=self.p_tuples,
            reindex_map=self.reindex_map,
            dataset_cls=self.dataset_cls,
            member_model=self.member_model,
            mc_member_model=self.mc_member_model,
            mc_n_channels_to_sample=self.mc_n_channels_to_sample,
            model_output_key=self.model_output_key,
            target_shape=self.target_shape,
            dataset=split_1,
        )

        return split_0_dset_with_model, split_1_dset_with_model

    def split_on_time(
        self, split_time: float
    ) -> Tuple["DatasetWithModel", "DatasetWithModel"]:
        assert self.dataset is not None
        split_0, split_1 = self.dataset.split_select_at_time(split_time=split_time)

        split_0_dset_with_model = DatasetWithModel(
            p_tuples=self.p_tuples,
            reindex_map=self.reindex_map,
            dataset_cls=self.dataset_cls,
            member_model=self.member_model,
            mc_member_model=self.mc_member_model,
            mc_n_channels_to_sample=self.mc_n_channels_to_sample,
            model_output_key=self.model_output_key,
            target_shape=self.target_shape,
            dataset=split_0,
        )

        split_1_dset_with_model = DatasetWithModel(
            p_tuples=self.p_tuples,
            reindex_map=self.reindex_map,
            dataset_cls=self.dataset_cls,
            member_model=self.member_model,
            mc_member_model=self.mc_member_model,
            mc_n_channels_to_sample=self.mc_n_channels_to_sample,
            model_output_key=self.model_output_key,
            target_shape=self.target_shape,
            dataset=split_1,
        )

        return split_0_dset_with_model, split_1_dset_with_model

    def as_feature_extracting_datapipe(
        self,
        batch_size,
        device,
        multi_channel: bool = True,
        batches_per_epoch: Optional[int] = None,
        one_hot_encode_target: bool = True,
    ):
        if self.mc_member_model is not None:
            self.mc_member_model = self.mc_member_model.to(device)

        if multi_channel and self.mc_member_model is None:
            self.create_mc_model()

        self.feature_model = (
            self.mc_member_model.to(device) if multi_channel else self.member_model.to(device)
        )

        model_out_dataset = SequenceWrapper(self.dataset)
        if isinstance(self.mc_n_channels_to_sample, int):
            # TODO: Sample the channels after going through the model, that way
            # can cache before sampling and still save significant time and
            # memory
            def sample_mc_channels(_d):
                x = _d["signal_arr"]
                sample_ixes = torch.randint(
                    low=0, high=x.shape[0], size=(self.mc_n_channels_to_sample,)
                )
                updates = {
                    k: arr[sample_ixes]
                    for k, arr in _d.items()
                    if torch.is_tensor(arr) and arr.ndim > 1
                }
                output = dict(**_d)
                output.update(updates)
                return output

            model_out_dataset = model_out_dataset.map(sample_mc_channels)

        model_out_dataset = model_out_dataset.batch(batch_size).map(default_collate)

        if multi_channel:

            def run_mc_model(_d):
                with torch.no_grad():
                    return {
                        _k: _t.to("cpu") if torch.is_tensor(_t) else _t
                        for _k, _t in self.feature_model(
                            {k: t.to(device) if torch.is_tensor(t) else t
                             for k, t in _d.items()},
                            #features_only=True, mask=False,
                            features_only=self.features_only, mask=self.mask,
                            output_loss=self.output_loss
                        ).items()
                    }

            model_out_dataset = model_out_dataset.map(run_mc_model)
        else:

            def run_sc_model(_d):
                with torch.no_grad():
                    ret_d = {
                        _k: _t.to("cpu") if isinstance(_t, torch.Tensor) else _t
                        for _k, _t in self.feature_model(
                            {k: t.to(device) if isinstance(t, torch.Tensor) else t
                             for k, t in _d.items()},
                            # Features only returns only the CNN outputs
                            #features_only=True, mask=False,
                            features_only=self.features_only, mask=self.mask,
                            output_loss=self.output_loss
                        ).items()
                    }
                    ret_d["target_arr"] = _d["target_arr"]
                    return ret_d

            model_out_dataset = model_out_dataset.map(run_sc_model)

        def _one_hot_target(_d):
            _d["target_cls_ix"] = _d["target_arr"]
            _d["target_arr"] = F.one_hot(
                _d["target_arr"], num_classes=self.target_shape
            )
            return _d

        if batches_per_epoch is not None:
            model_out_dataset = td_iter.Sampler(
                model_out_dataset.to_iter_datapipe(),
                sampler=torch.utils.data.RandomSampler,
                sampler_kwargs=dict(num_samples=batches_per_epoch),
            ).to_map_datapipe()

        model_out_dataset = model_out_dataset.map(_one_hot_target).in_memory_cache()

        model_out_dataset = self.patch_attributes(model_out_dataset)
        return model_out_dataset

    def patch_attributes(self, obj):
        def _get_target_shape(*args):
            return self.target_shape

        obj.selected_columns = self.dataset.selected_columns
        obj.get_target_shape = _get_target_shape
        obj.get_target_labels = self.dataset.get_target_labels
        return obj


class DatasetWithModelBaseTask(bxp.TaskOptions):
    dataset: datasets.DatasetOptions

    dataset_map: Optional[Dict[str, BaseDataset]] = field(default=None, init=False)
    dataset_with_model_d: Optional[Dict[str, DatasetWithModel]] = field(default=None, init=False)

    def get_target_rates(self, normalize=True, as_series: bool = False):
        return {part: dset_with_model.dataset.get_target_rates(normalize=normalize, as_series=as_series)
                for part, dset_with_model in self.dataset_with_model_d.items()}

    @classmethod
    def load_and_check_pretrain_opts(
        cls, results, expected_dataset_name, dataset_cls, expected_n_pretrain_sets=2
    ) -> List[Tuple]:
        sm_member_dataset_opts = results["dataset_options"]
        assert sm_member_dataset_opts["dataset_name"] == expected_dataset_name
        pretrain_sets = dataset_cls.make_tuples_from_sets_str(
            sm_member_dataset_opts["train_sets"]
        )
        assert len(pretrain_sets) == expected_n_pretrain_sets
        return pretrain_sets

    @classmethod
    def load_pretrained_model_results(
        cls,
        pretrained_result_input_path: str = None,
        pretrained_result_model_base_path: str = None,
        device=None,
    ):
        assert_err = "pretrained_result_input_path must be provided"
        assert pretrained_result_input_path is not None, assert_err

        from brain2vec.experiments import load_model_from_results

        if pretrained_result_model_base_path is None:
            pretrained_result_model_base_path = os.path.join(
                os.path.split(pretrained_result_input_path)[0], "models"
            )

        print(
            f"Loading pretrained model from results in "
            f"{pretrained_result_input_path}"
            f" (base path = {pretrained_result_model_base_path})"
        )
        with open(pretrained_result_input_path, "r") as f:
            result_json = json.load(f)

        print(f"\tKEYS: {list(sorted(result_json.keys()))}")
        if "model_name" not in result_json:
            logger.info("MODEL NAME MISSING - setting to cog2vec")
            result_json["model_name"] = "cog2vec"

        if "cv" in result_json:
            min_pretrain_bce = min(result_json["cv"]["bce_loss"])
            logger.info(f"MIN PRETRAINED BCE LOSS: {min_pretrain_bce}")
        else:
            logger.info(
                "Pretraining result didn't have a 'cv' to check" " the loss of ... D:"
            )

        pretrained_model = load_model_from_results(
            result_json, base_model_path=pretrained_result_model_base_path
        )
        if device is not None:
            pretrained_model = pretrained_model.to(device)

        return pretrained_model, result_json

    def load_one_dataset_with_model(
        self,
        dset_w_model: DatasetWithModel,
        sensor_columns: str,
        is_multichannel: bool = False,
    ):
#        if sensor_columns is None:
#            sensor_columns = (
#                "good_for_participant"
#                if self.dataset.flatten_sensors_to_samples
#                else "valid"
#            )

        if isinstance(self.dataset.pipeline_params, str):
            self.dataset.pipeline_params = eval(self.dataset.pipeline_params)

        kws, _, _ = self.dataset.make_dataset_kws(train_p_tuples=dset_w_model.p_tuples,
                                                  train_data_kws=dict(label_reindex_map=dset_w_model.reindex_map),
                                                  train_sensor_columns=sensor_columns
                                                  )

        #base_kws = dict(
        #    pre_processing_pipeline=self.dataset.pre_processing_pipeline,
        #    pipeline_params=self.dataset.pipeline_params,
        #    data_subset=self.dataset.data_subset,
        #    label_reindex_col=self.dataset.label_reindex_col,
        #    extra_output_keys=self.dataset.extra_output_keys.split(",")
        #                        if self.dataset.extra_output_keys is not None else None,
        #    flatten_sensors_to_samples=self.dataset.flatten_sensors_to_samples,
        #)
        #kws = dict(
        #    patient_tuples=dset_w_model.p_tuples,
        #    label_reindex_map=dset_w_model.reindex_map,
        #    sensor_columns=sensor_columns,
        #    **base_kws,
        #)
        dset_w_model.dataset = dset_w_model.dataset_cls(**kws)

        if is_multichannel:
            dset_w_model.create_mc_model()

        return dset_w_model

    def get_target_labels(self) -> Dict[str, List]:
        return self.dataset_map["train"].get_target_labels()


@with_logger
@dataclass
class ReidentificationTask(DatasetWithModelBaseTask):
    task_name: str = "reidentification_classification_fine_tuning"
    dataset: datasets.DatasetOptions = HarvardSentencesDatasetOptions(
        train_sets="AUTO-PRETRAINING",
        flatten_sensors_to_samples=True,
        label_reindex_col="patient",
        split_cv_from_test=False,
        pipeline_params="{}",
        pre_processing_pipeline="random_sample",
    )

    pretrained_model_input: bxp.ResultInputOptions = None

    train_sample_slice: ClassVar[slice] = slice(0, 0.3)
    test_sample_slice: ClassVar[slice] = slice(0.7, 1.0)

    weight_decay: float = 0.0
    squeeze_target: ClassVar[bool] = True

    def make_criteria_and_target_key(self):
        cls_rates = self.dataset_with_model_d["train"].dataset.get_target_weights()
        self.logger.info(f"WEIGHTS: {cls_rates}")
        weight = torch.Tensor(cls_rates).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        target_key = "target_arr"
        return criterion, target_key

    def get_member_model_output_shape(self):
        return self.pretrained_model.T, self.pretrained_model.C

    def make_dataset_and_loaders(self, **kws):
        dataset_cls = datasets.BaseDataset.get_dataset_by_name(
            self.dataset.dataset_name
        )

        # Load the pretrained model and it's options JSON data
        (
            self.pretrained_model,
            self.pretrained_model_opts,
        ) = self.load_pretrained_model_results(
            self.pretrained_model_input.result_file,
            self.pretrained_model_input.model_base_path,
        )

        # Check that the loaded model was pretrained on the dataset
        # that was specified
        pretrained_dset_options = self.pretrained_model_opts["dataset_options"]
        assert pretrained_dset_options["dataset_name"] == self.dataset.dataset_name
        # Get a list of set patient tuples used for pretraining
        pretrain_sets: List[Tuple] = dataset_cls.make_tuples_from_sets_str(
            pretrained_dset_options["train_sets"]
        )

        is_mc = not self.dataset.flatten_sensors_to_samples

        # Pipeline parameters are supposed to be a dict for passing to pipeline.set_params()
        # - At this point, the pipeline parameters may be None or a string passed from the CLI
        if isinstance(self.dataset.pipeline_params, str):
            self.dataset.pipeline_params = eval(self.dataset.pipeline_params)
        # Make sure it's an empty dictionary so later calls to `update()` the dict will work
        elif self.dataset.pipeline_params is None:
            self.dataset.pipeline_params = dict()

        # Object to wrap the dataset and the model for caching the pretrained models extracted features
        train_dataset_with_model = DatasetWithModel(
            p_tuples=pretrain_sets,
            reindex_map=None,
            dataset_cls=dataset_cls,
            member_model=self.pretrained_model,
            target_shape=len(pretrain_sets),
        )

        # Update the current pipeline_params with train slice - which should be an early section of the data
        self.dataset.pipeline_params.update(
            {"rnd_stim__slice_selection": self.train_sample_slice}
        )
        # Load up the data - this modifies the obect in place
        sensor_columns = 'valid' if is_mc else 'good_for_participant'
        self.load_one_dataset_with_model(
            train_dataset_with_model, sensor_columns=sensor_columns,
            is_multichannel=is_mc
        )

        # The train set determines the selected columns
        if sensor_columns == 'valid':
            # selected_columns will be a set of valid train time columns
            # When not valid, like good for participant used, then this attribute is None
            selected_columns = train_dataset_with_model.dataset.selected_columns
        else:
            selected_columns = sensor_columns

        (
            train_dataset_with_model,
            cv_dataset_with_model,
        ) = train_dataset_with_model.split_on_key_level(
            keys=("patient", "sent_code"), test_size=0.30
        )
        # train_dataset_with_model, cv_dataset_with_model = train_dataset_with_model.split_on_time(0.5)

        test_dataset_with_model = DatasetWithModel(
            p_tuples=pretrain_sets,
            reindex_map=None,
            dataset_cls=dataset_cls,
            member_model=self.pretrained_model,
            target_shape=len(pretrain_sets),
        )
        # self.dataset.pipeline_params = {"rnd_stim__slice_selection": self.test_sample_slice}
        self.dataset.pipeline_params.update(
            {"rnd_stim__slice_selection": self.test_sample_slice}
        )
        self.load_one_dataset_with_model(
            test_dataset_with_model,
            sensor_columns=selected_columns,
            is_multichannel=is_mc,
        )

        # -----
        train_pipe = train_dataset_with_model.as_feature_extracting_datapipe(
            self.dataset.batch_size,
            batches_per_epoch=self.dataset.batches_per_epoch,
            device=self.device,
            multi_channel=is_mc,
        )

        test_batch_size = (
            self.dataset.batch_size
            if self.dataset.batch_size_eval is None
            else self.dataset.batch_size_eval
        )
        test_batches_per_epoch = (
            self.dataset.batches_per_epoch
            if self.dataset.batches_per_eval_epoch is None
            else self.dataset.batches_per_eval_epoch
        )

        cv_pipe = cv_dataset_with_model.as_feature_extracting_datapipe(
            test_batch_size,
            batches_per_epoch=test_batches_per_epoch,
            device=self.device,
            multi_channel=is_mc,
        )

        test_pipe = test_dataset_with_model.as_feature_extracting_datapipe(
            test_batch_size,
            batches_per_epoch=test_batches_per_epoch,
            device=self.device,
            multi_channel=is_mc,
        )

        self.dataset_with_model_d = dict(
            train=train_dataset_with_model,
            cv=cv_dataset_with_model,
            test=test_dataset_with_model,
        )
        self.dataset_map = dict(train=train_pipe, cv=cv_pipe, test=test_pipe)

        return dict(self.dataset_map), dict(self.dataset_map), dict(self.dataset_map)


@with_logger
@dataclass
class OneModelMembershipInferenceFineTuningTask(DatasetWithModelBaseTask):
    """
    Given access to a Target model trained on 3 participants, with attacker's access
    to two participants used in pretraining and 2 participants not used in training.
    This leaves another two participants not used in either pretraining nor in attacker training.
    These two participants and the remaining pretraining participant not seen by the attacker are used
    as the final test of the attacker's model
    """

    task_name: str = "one_model_mi_classification_fine_tuning"

    pretrained_target_model_result_input: Optional[bxp.ResultInputOptions] = None
    dataset: HarvardSentencesDatasetOptions = HarvardSentencesDatasetOptions(
        train_sets="AUTO-PRETRAINING",
        flatten_sensors_to_samples=False,
        label_reindex_col="patient",
        split_cv_from_test=False,
        pre_processing_pipeline="random_sample",
    )
    pretrained_result_dir: Optional[str] = None
    method: str = "2d_linear"
    model_output_key: str = "x"
    n_channels_to_sample: Optional[int] = None
    weight_decay: float = 0.0

    squeeze_target: ClassVar[bool] = True

    # -- Non-init fields
    dataset_with_model_d: dict = field(default=None, init=False)
    target_model: torch.nn.Module = field(default=None, init=False)
    target_model_results: dict = field(default=None, init=False)

    def make_criteria_and_target_key(self):
        cls_rates = self.dataset_with_model_d["train"].dataset.get_target_weights()
        self.logger.info(f"WEIGHTS: {cls_rates}")
        weight = torch.Tensor(cls_rates).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        target_key = "target_arr"
        return criterion, target_key

    def get_member_model_output_shape(self) -> Tuple:
        member_model = self.target_model
        return member_model.T, member_model.C

    def get_num_channels(self):
        if self.n_channels_to_sample is not None:
            return self.n_channels_to_sample
        else:
            return len(self.dataset_with_model_d["train"].selected_columns)

    def get_target_labels(self):
        target_label_map = {0: "non_member", 1: "member"}
        target_labels = list(target_label_map.values())
        return target_label_map, target_labels

    @staticmethod
    def find_model_pretrained_on(pretrained_result_dir):
        pass

    def infer_target_model_from_train_sets(self):
        assert self.dataset.train_sets is not None
        assert self.pretrained_result_dir is not None
        assert os.path.isdir(self.pretrained_result_dir)
        from brain2vec.experiments import load_results_to_frame

        pretrained_model_dir = os.path.join(self.pretrained_result_dir, "models")
        model_glob_p = os.path.join(self.pretrained_result_dir, "*.json")

        # -----------
        # Load *all* pretrained results into a frame
        results_df = load_results_to_frame(model_glob_p)
        result_options_df = experiments.upack_result_options_to_columns(results_df)

        result_options_df.datetime.min(), result_options_df.datetime.max()
        assert (
            result_options_df.train_sets.is_unique
        ), f"Some train sets in {model_glob_p} are duplicated"
        assert (
            result_options_df.dataset_name.nunique() == 1
        ), "Expected only one dataset type"

        # Prep the dataset class
        dataset_name = result_options_df.dataset_name.unique()[0]
        assert (
            dataset_name == self.dataset.dataset_name
        ), f"Expected model dataset to be {self.dataset.dataset_name}"
        # all_pretrain_str_set = result_options_df.train_sets.pipe(set)
        dataset_cls = BaseDataset.get_dataset_by_name(dataset_name)

        # --
        # The TRAIN set passed is the desired pretrain set for the target model
        input_pretrain_tuples_l = dataset_cls.make_tuples_from_sets_str(
            self.dataset.train_sets
        )
        # Get the unique training tuples ever seen in all the pretraining
        # models training data
        pretrain_result_tuples_s = result_options_df.train_sets.map(
            dataset_cls.make_tuples_from_sets_str
        )
        assert (
            pretrain_result_tuples_s.apply(len).nunique() == 1
        ), f"Expected models in {self.pretrained_result_dir} to have train sets with the same number of participants"

        # Find the pretrained model that was trained on the input
        target_models_m = pretrain_result_tuples_s.apply(
            lambda l: all(_l in input_pretrain_tuples_l for _l in l)
        )
        target_model_result_files = result_options_df[target_models_m].name.values
        assert (
            len(target_model_result_files) == 1
        ), "Expected only one model to match pretrain inputs"
        pretrained_target_model_file = target_model_result_files[0]

        target_result_file_p = os.path.join(
            self.pretrained_result_dir, pretrained_target_model_file
        )
        input_options = bxp.ResultInputOptions(
            target_result_file_p, pretrained_model_dir
        )
        self.pretrained_target_model_result_input = input_options

        return self

    def make_dataset_and_loaders(self, **kws):
        # Preliminaries
        sensor_columns = kws.get("sensor_columns", "good_for_participant")
        is_multichannel = True if "2d" in self.method else False
        # -

        dataset_cls = datasets.BaseDataset.get_dataset_by_name(
            self.dataset.dataset_name
        )
        all_sets = dataset_cls.make_tuples_from_sets_str("*")

        if self.pretrained_result_dir is not None:
            assert (
                self.pretrained_target_model_result_input is None
            ), "Pretrained result input is defined, but a pretrain_result_dir also passed"
            self.infer_target_model_from_train_sets()

        # Load target model and it's results
        (
            self.target_model,
            self.target_model_results,
        ) = self.load_pretrained_model_results(
            pretrained_result_input_path=self.pretrained_target_model_result_input.result_file,
            pretrained_result_model_base_path=self.pretrained_target_model_result_input.model_base_path,
        )

        # Identify the participants used to pretrain the target model
        pretrain_train_sets_opt = self.target_model_results["dataset_options"][
            "train_sets"
        ]
        all_true_pos_set = dataset_cls.make_tuples_from_sets_str(
            pretrain_train_sets_opt
        )

        # Identify all the participants not used in pretraining the model
        all_true_neg_set = list(set(all_sets) - set(all_true_pos_set))

        logger.info(
            f"True neg set of length {len(all_true_neg_set)} contains: {all_true_neg_set}"
        )

        # Determine the test set's single positive participant set - the attacker will not train on this member
        # Was a test set specified in the options directly?
        if self.dataset.test_sets is not None and self.dataset.test_sets != "":
            # If an integer is coming as coming as a string type - conver it to actual integer
            if (
                isinstance(self.dataset.test_sets, str)
                and self.dataset.test_sets.isdigit()
            ):
                self.dataset.test_sets = int(self.dataset.test_sets)

            # If an integer
            if isinstance(self.dataset.test_sets, int):
                true_pos_ix_of_test = [self.dataset.test_sets]
                self.dataset.test_sets = None
            else:
                raise ValueError(
                    f"Dont understand test set - it wasn't an integet: {self.dataset.test_sets}"
                )
        # No test set specified, so select a random one from true positive set
        elif self.dataset.test_sets is None:
            true_pos_ix_of_test = np.random.choice(len(all_true_pos_set), 1).tolist()
        else:
            raise ValueError(f"Dont understand test set: {self.dataset.test_sets}")

        assert all(
            _ix < len(all_true_pos_set) for _ix in true_pos_ix_of_test
        ), "Test positive outside ix range of pos"
        assert all(
            _ix >= 0 for _ix in true_pos_ix_of_test
        ), "Test pos is negative...???"

        # Pull out the selected true positive tuple to be in the test set
        debias_test_pos_set = [all_true_pos_set[_ix] for _ix in true_pos_ix_of_test]
        # Training positives are all the positives that are not in the test set
        debias_train_pos_set = [
            t for t in all_true_pos_set if t not in debias_test_pos_set
        ]

        # Take half of true negatives (i.e. not seen in pretraining procedure)
        # to make attacker's testing negatives
        true_neg_ix_of_test = np.random.choice(
            len(all_true_neg_set), len(all_true_neg_set) // 2
        ).tolist()
        debias_test_neg_set = [all_true_neg_set[_ix] for _ix in true_neg_ix_of_test]

        # Take the remaining true negatives for use in attacker's training
        debias_train_neg_set = [
            t for t in all_true_neg_set if t not in debias_test_neg_set
        ]

        train_label_reindex_map = dict()
        train_label_reindex_map.update({t[1]: 0 for t in debias_train_neg_set})
        train_label_reindex_map.update({t[1]: 1 for t in debias_train_pos_set})
        logger.info(f"Train label reindex map created: {train_label_reindex_map}")

        test_label_reindex_map = dict()
        test_label_reindex_map.update({t[1]: 0 for t in debias_test_neg_set})
        test_label_reindex_map.update({t[1]: 1 for t in debias_test_pos_set})
        logger.info(f"Test label reindex map created: {test_label_reindex_map}")

        train_dataset_with_model = DatasetWithModel(
            p_tuples=debias_train_neg_set + debias_train_pos_set,
            reindex_map=train_label_reindex_map,
            dataset_cls=HarvardSentences,
            member_model=self.target_model,
            mc_n_channels_to_sample=self.n_channels_to_sample,
            model_output_key=self.model_output_key,
        )
        train_dataset_with_model = self.load_one_dataset_with_model(
            train_dataset_with_model,
            sensor_columns=sensor_columns,
            is_multichannel=is_multichannel,
        )
        if sensor_columns is None or sensor_columns == "good_for_participant":
            selected_sensor_columns = sensor_columns
        else:
            selected_sensor_columns = train_dataset_with_model.dataset.selected_columns

        (
            train_dataset_with_model,
            cv_dataset_with_model,
        ) = train_dataset_with_model.split_on_key_level(
            keys=("patient", "sent_code"), test_size=0.25
        )

        train_datapipe = train_dataset_with_model.as_feature_extracting_datapipe(
            self.dataset.batch_size,
            batches_per_epoch=self.dataset.batches_per_epoch,
            device=self.device,
            multi_channel=is_multichannel,
        )
        train_datapipe = train_dataset_with_model.patch_attributes(train_datapipe)

        cv_datapipe = cv_dataset_with_model.as_feature_extracting_datapipe(
            self.dataset.batch_size_eval
            if self.dataset.batch_size_eval is not None
            else self.dataset.batch_size,
            batches_per_epoch=self.dataset.batches_per_epoch,
            device=self.device,
            multi_channel=is_multichannel,
        )
        cv_datapipe = cv_dataset_with_model.patch_attributes(cv_datapipe)

        # ---
        test_dataset_with_model = DatasetWithModel(
            p_tuples=debias_test_neg_set + debias_test_pos_set,
            reindex_map=test_label_reindex_map,
            dataset_cls=HarvardSentences,
            member_model=self.target_model,
            mc_n_channels_to_sample=self.n_channels_to_sample,
            model_output_key=self.model_output_key,
        )
        test_dataset_with_model = self.load_one_dataset_with_model(
            test_dataset_with_model,
            sensor_columns=selected_sensor_columns,
            is_multichannel=is_multichannel,
        )

        test_datapipe = test_dataset_with_model.as_feature_extracting_datapipe(
            self.dataset.batch_size_eval
            if self.dataset.batch_size_eval is not None
            else self.dataset.batch_size,
            batches_per_epoch=self.dataset.batches_per_epoch,
            device=self.device,
            multi_channel=is_multichannel,
        )
        test_datapipe = test_dataset_with_model.patch_attributes(test_datapipe)
        # --- End Test data pipe setup ---

        self.dataset_with_model_d = dict(
            train=train_dataset_with_model,
            cv=cv_dataset_with_model,
            test=test_dataset_with_model,
        )
        self.dataset_map = dict(
            train=train_datapipe, cv=cv_datapipe, test=test_datapipe
        )

        return dict(self.dataset_map), dict(self.dataset_map), dict(self.dataset_map)


@with_logger
@dataclass
class ShadowClassifierMembershipInferenceFineTuningTask(DatasetWithModelBaseTask):
    task_name: str = "shadow_mi_classification_fine_tuning"
    pretrained_result_dir: Optional[str] = None

    dataset: datasets.DatasetOptions = HarvardSentencesDatasetOptions(
        train_sets="AUTO-PRETRAINING",
        flatten_sensors_to_samples=True,
        label_reindex_col="patient",
        split_cv_from_test=False,
        pre_processing_pipeline="random_sample",
    )
    method: str = "1d_linear"
    model_output_key: str = "x"
    n_channels_to_sample: Optional[int] = None
    weight_decay: float = 0.0

    squeeze_target: ClassVar[bool] = True

    dataset_with_model_d: Optional[dict] = field(default=None, init=False)

    def make_criteria_and_target_key(self):
        cls_rates = self.get_target_weights()
        self.logger.info(f"WEIGHTS: {cls_rates}")
        weight = torch.Tensor(cls_rates).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        
        # criterion = torch.nn.BCELoss()
        # weight = torch.Tensor([1., 3.]).to(self.device)
        # criterion = torch.nn.CrossEntropyLoss(weight=weight)
        #criterion = torch.nn.CrossEntropyLoss()  # weight=weight)
        target_key = "target_arr"
        return criterion, target_key

    def load_models(self):
        if self.dataset.test_sets is None:
            raise ValueError(f"test_sets is None, but it must be set")

        self.infer_models_from_test_sets()

    def infer_models_from_test_sets(self):
        assert self.dataset.test_sets is not None
        assert self.pretrained_result_dir is not None
        assert os.path.isdir(self.pretrained_result_dir)
        from brain2vec.experiments import load_results_to_frame

        pretrained_model_dir = os.path.join(self.pretrained_result_dir, "models")
        model_glob_p = os.path.join(self.pretrained_result_dir, "*.json")

        # -----------
        # Load *all* pretrained results into a frame
        results_df = load_results_to_frame(model_glob_p)
        result_options_df = experiments.upack_result_options_to_columns(results_df)

        assert (
            result_options_df.train_sets.is_unique
        ), f"Some train sets in {model_glob_p} are duplicated"
        assert (
            result_options_df.dataset_name.nunique() == 1
        ), "Expected only one dataset type"

        # Prep the dataset class
        dataset_name = result_options_df.dataset_name.unique()[0]
        assert (
            dataset_name == self.dataset.dataset_name
        ), f"Expected model dataset to be {self.dataset.dataset_name}"
        # all_pretrain_str_set = result_options_df.train_sets.pipe(set)
        dataset_cls = BaseDataset.get_dataset_by_name(dataset_name)

        # The test set for target model
        unique_test_tuples_l = dataset_cls.make_tuples_from_sets_str(
            self.dataset.test_sets
        )
        # Get the unique training tuples ever seen in all the pretraining models training data
        train_set_tuples_s = result_options_df.train_sets.map(
            dataset_cls.make_tuples_from_sets_str
        )

        assert (
            train_set_tuples_s.apply(len).nunique() == 1
        ), f"Expected models in {self.pretrained_result_dir} to have train sets with the same number of participants"

        # Unique participants used in the training set
        unique_train_tuples_l = set(train_set_tuples_s.sum())

        # Determine which participants were not used in the test set - these are usable by the attacker
        shadow_model_tuples = set(unique_train_tuples_l) - set(unique_test_tuples_l)

        # Find the models that were pretrained on participants exclusively in the test set of participant tuples
        target_models_m = train_set_tuples_s.apply(
            lambda l: all(_l in unique_test_tuples_l for _l in l)
        )
        target_model_result_files = result_options_df[target_models_m].name.values

        # Find the models that were pretrained on participants exclusively in the test set of participant tuples
        shadow_models_m = train_set_tuples_s.apply(
            lambda l: all(_l in shadow_model_tuples for _l in l)
        )
        shadow_model_result_files = result_options_df[shadow_models_m].name.values

        self.shadow_model_results_map = dict()
        self.shadow_model_map = dict()
        self.target_model_map = dict()
        self.target_model_results_map = dict()

        for sm_id, sm_file in enumerate(shadow_model_result_files):
            result_file_p = os.path.join(self.pretrained_result_dir, sm_file)
            self.logger.info("--------------------")
            self.logger.info(f"Shadow model {sm_id}: " + str(result_file_p))

            (
                self.shadow_model_map[sm_file],
                self.shadow_model_results_map[sm_file],
            ) = self.load_pretrained_model_results(
                result_file_p, pretrained_model_dir, device=self.device
            )

        for tgt_id, tgt_file in enumerate(target_model_result_files):
            result_file_p = os.path.join(self.pretrained_result_dir, tgt_file)
            self.logger.info("--------------------")
            self.logger.info(f"Target model {tgt_id}: " + str(result_file_p))

            (
                self.target_model_map[tgt_file],
                self.target_model_results_map[tgt_file],
            ) = self.load_pretrained_model_results(
                result_file_p, pretrained_model_dir, device=self.device
            )

    def get_num_channels(self):
        if self.n_channels_to_sample is not None:
            return self.n_channels_to_sample
        else:
            return len(self.dataset_with_model_d["train"].selected_columns)

    def get_target_rates(self, normalize=True, as_series: bool = False):
        part_target_rate_d = dict()
        for part, dset_with_model_sm_d in self.dataset_with_model_d.items():

            totals_d = {sm_id: sm_dset_with_model.dataset.get_target_rates(normalize=False, as_series=True)
                        #for part, dset_with_model_sm_d in self.dataset_with_model_d.items()
                        for sm_id, sm_dset_with_model in dset_with_model_sm_d.items()
                            }
            # Resulting frame will have columns for each part and 
            totals_s = pd.DataFrame(totals_d).T.sum()
            self.logger.info(f"TOTALS: {totals_s}")
            # TODO: check that dims are correct here and implement normalizing and as_series options
            if normalize:
                totals_s = totals_s / totals_s.sum()
                self.logger.info(f"Normed rate: {totals_s}")
                    

            part_target_rate_d[part] = totals_s if as_series else totals_s.values
        
        return part_target_rate_d
    
    def get_target_weights(self) -> np.ndarray:
        rates = self.get_target_rates()['train']
        return 1. / rates

    def get_member_model_output_shape(self) -> Tuple:
        member_model = next(iter(self.shadow_model_map.values()))
        return member_model.T, member_model.C

    def make_dataset_and_loaders(self, **kws) -> Tuple[Dict, Dict, Dict]:
        self.load_models()
        # Target models and shadow models need to be
        assert all(
            tgt_id not in self.shadow_model_map
            for tgt_id in self.target_model_map.keys()
        ), "IDs in target and shadow maps are not unique"
        sensor_columns = kws.get("sensor_columns", None)

        # We'll use the CLI options to set the expected dataset - later assert that all models used same dataset
        dataset_cls = datasets.BaseDataset.get_dataset_by_name(
            self.dataset.dataset_name
        )

        sm_pretrain_sets_d = {
            k: self.load_and_check_pretrain_opts(
                res, self.dataset.dataset_name, dataset_cls
            )
            for k, res in self.shadow_model_results_map.items()
        }
        target_pretrain_sets_d = {
            k: self.load_and_check_pretrain_opts(
                res, self.dataset.dataset_name, dataset_cls
            )
            for k, res in self.target_model_results_map.items()
        }

        # All set tuples used across shadow models
        unique_shadow_sets = list(set(sum(sm_pretrain_sets_d.values(), list())))

        # Select two shadow models from the train set to use in the CV
        cv_method = 'pt'
        if cv_method == 'simple':
            # - TODO: maybe select on participant and their three pretrained models? Stronger early stopping
            cv_shadow_models = np.random.choice(list(self.shadow_model_map.keys()), size=2)
        elif cv_method == 'pt':
            cv_pt_ix = np.random.choice(list(range(len(unique_shadow_sets))))
            cv_pt_tuple = unique_shadow_sets[cv_pt_ix]
            cv_shadow_models = np.array([sm_id
                                for sm_id, sm_res_d in self.shadow_model_results_map.items() 
                                if f'UCSD-{cv_pt_tuple[1]}' in sm_res_d['dataset_options']['train_sets']])

            self.logger.info(f'CV tuple: {cv_pt_tuple}, SM IDs: {cv_shadow_models}')

        is_multichannel = True if "2d" in self.method else False
        sensor_columns = (
            "good_for_participant" if sensor_columns is None else sensor_columns
        )
        selected_columns = sensor_columns

        all_dataset_with_model_map = {sm_k: DatasetWithModel(p_tuples=unique_shadow_sets, 
                                                             reindex_map={
                                                                 s[1]: 1 if s in sm_pretrain_sets_d[sm_k] else 0
                                                                 for s in unique_shadow_sets
                                                                 },
                                                             dataset_cls=dataset_cls,
                                                             member_model=m,
                                                             mc_n_channels_to_sample=self.n_channels_to_sample,
                                                             model_output_key=self.model_output_key,
                                                             output_loss=self.model_output_key == 'bce_loss'
                                                             )
                                      for sm_k, m in self.shadow_model_map.items()}
            
        unique_target_sets: List[Tuple] = list(
            set(sum(target_pretrain_sets_d.values(), list()))
        )

        all_dataset_with_model_map.update(
                {t_k: DatasetWithModel(p_tuples=unique_target_sets, 
                                       reindex_map={
                                           s[1]: 1 if s in target_pretrain_sets_d[t_k] else 0
                                           for s in unique_target_sets
                                           },
                                       dataset_cls=dataset_cls,
                                       member_model=m,
                                       mc_n_channels_to_sample=self.n_channels_to_sample,
                                       model_output_key=self.model_output_key,
                                       output_loss=self.model_output_key == 'bce_loss'
                                       )
                 for t_k, m in self.target_model_map.items()}

                )

        loaded_dsets = Parallel(backend='loky', n_jobs=len(all_dataset_with_model_map.keys()), verbose=10)(
                delayed(self.load_one_dataset_with_model)(dset_with_model, 
                                                          sensor_columns=selected_columns,
                                                          is_multichannel=is_multichannel)
                for sm_or_t_k, dset_with_model in all_dataset_with_model_map.items()
                )

        all_dataset_with_model_map = dict(zip(all_dataset_with_model_map.keys(), loaded_dsets))

        # - Train - 
        shadow_training_sets_with_model = {sm_k: m for sm_k, m in all_dataset_with_model_map.items()
                                           if sm_k not in cv_shadow_models}
        train_pipes = [
            s.as_feature_extracting_datapipe(
                self.dataset.batch_size,
                batches_per_epoch=self.dataset.batches_per_epoch,
                device=self.device,
                multi_channel=is_multichannel,
            )
            for k, s in shadow_training_sets_with_model.items()
        ]

        training_datapipe = Concater(*train_pipes).shuffle()
        training_datapipe = next(iter(shadow_training_sets_with_model.values())).patch_attributes(
            training_datapipe
        )

        # - CV - 
        shadow_cv_sets_with_model = {sm_k: m for sm_k, m in all_dataset_with_model_map.items()
                                     if sm_k in cv_shadow_models}
        cv_datapipe = Concater(
                *[
                    s.as_feature_extracting_datapipe(
                        self.dataset.batch_size,
                        batches_per_epoch=self.dataset.batches_per_epoch,
                        device=self.device,
                        multi_channel=is_multichannel,
                        )
                    for k, s in shadow_cv_sets_with_model.items()
                    ]
                ).shuffle()
        cv_datapipe = next(iter(shadow_cv_sets_with_model.values())).patch_attributes(
                cv_datapipe
                )
 

        test_sets_with_model = {t_k: m for t_k, m in all_dataset_with_model_map.items()
                                     if t_k in self.target_model_map.keys()}

        test_batch_size = (
            self.dataset.batch_size
            if self.dataset.batch_size_eval is None
            else self.dataset.batch_size_eval
        )
        test_batches_per_epoch = (
            self.dataset.batches_per_epoch
            if self.dataset.batches_per_eval_epoch is None
            else self.dataset.batches_per_eval_epoch
        )

        test_datapipe = Concater(
            *[
                s.as_feature_extracting_datapipe(
                    batch_size=test_batch_size,
                    batches_per_epoch=test_batches_per_epoch,
                    device=self.device,
                    multi_channel=is_multichannel,
                )
                for k, s in test_sets_with_model.items()
            ]
        ).shuffle()

        test_datapipe = next(iter(test_sets_with_model.values())).patch_attributes(
            test_datapipe
        )

        self.dataset_with_model_d = dict(
            train=shadow_training_sets_with_model,
            cv=shadow_cv_sets_with_model,
            test=test_sets_with_model,
        )
        ret_datapipe_d = dict(
            train=training_datapipe, cv=cv_datapipe, test=test_datapipe
        )
        self.dataset_map = ret_datapipe_d

        return dict(ret_datapipe_d), dict(ret_datapipe_d), dict(ret_datapipe_d)


#    def _make_dataset_and_loaders(self, **kws) -> Tuple[Dict, Dict, Dict]:
#        self.load_models()
#        # Target models and shadow models need to be
#        assert all(
#            tgt_id not in self.shadow_model_map
#            for tgt_id in self.target_model_map.keys()
#        ), "IDs in target and shadow maps are not unique"
#        sensor_columns = kws.get("sensor_columns", None)
#
#        # We'll use the CLI options to set the expected dataset - later assert that all models used same dataset
#        dataset_cls = datasets.BaseDataset.get_dataset_by_name(
#            self.dataset.dataset_name
#        )
#
#        sm_pretrain_sets_d = {
#            k: self.load_and_check_pretrain_opts(
#                res, self.dataset.dataset_name, dataset_cls
#            )
#            for k, res in self.shadow_model_results_map.items()
#        }
#        target_pretrain_sets_d = {
#            k: self.load_and_check_pretrain_opts(
#                res, self.dataset.dataset_name, dataset_cls
#            )
#            for k, res in self.target_model_results_map.items()
#        }
#
#        # All set tuples used across shadow models
#        unique_shadow_sets = list(set(sum(sm_pretrain_sets_d.values(), list())))
#
#        # Select two shadow models from the train set to use in the CV
#        cv_method = 'pt'
#        if cv_method == 'simple':
#            # - TODO: maybe select on participant and their three pretrained models? Stronger early stopping
#            cv_shadow_models = np.random.choice(list(self.shadow_model_map.keys()), size=2)
#        elif cv_method == 'pt':
#            cv_pt_ix = np.random.choice(list(range(len(unique_shadow_sets))))
#            cv_pt_tuple = unique_shadow_sets[cv_pt_ix]
#            cv_shadow_models = np.array([sm_id
#                                for sm_id, sm_res_d in self.shadow_model_results_map.items() 
#                                if f'UCSD-{cv_pt_tuple[1]}' in sm_res_d['dataset_options']['train_sets']])
#
#            self.logger.info(f'CV tuple: {cv_pt_tuple}, SM IDs: {cv_shadow_models}')
#
#        # -------------
#        # Setup training data pipe for attacker
#        shadow_training_sets_with_model: Dict[str, DatasetWithModel] = dict()
#
#        for sm_k, sm_pretrained_model in self.shadow_model_map.items():
#            if sm_k in cv_shadow_models:
#                continue
#
#            shadow_training_sets_with_model[sm_k] = DatasetWithModel(
#                p_tuples=unique_shadow_sets,
#                # Pretraining participants from this model are the positive class (member)
#                reindex_map={
#                    s[1]: 1 if s in sm_pretrain_sets_d[sm_k] else 0
#                    for s in unique_shadow_sets
#                },
#                dataset_cls=dataset_cls,
#                member_model=sm_pretrained_model,
#                mc_n_channels_to_sample=self.n_channels_to_sample,
#                model_output_key=self.model_output_key,
#            )
#
#        # Load the first o
#        is_multichannel = True if "2d" in self.method else False
#        sensor_columns = (
#            "good_for_participant" if sensor_columns is None else sensor_columns
#        )
#        selected_columns = sensor_columns
#        #with parallel_backend('threading', n_jobs=len(shadow_training_sets_with_model)):
#        st_with_models = Parallel(#prefer='threads', 
#                                  backend='loky',
#                                  n_jobs=len(shadow_training_sets_with_model),
#                                  verbose=10)(delayed(self.load_one_dataset_with_model)(st_with_model, 
#                                                                                        sensor_columns=selected_columns,
#                                                                                        is_multichannel=is_multichannel)
#                                              for k, st_with_model in shadow_training_sets_with_model.items())
#        shadow_training_sets_with_model = dict(zip(shadow_training_sets_with_model.keys(), 
#                                                   st_with_models))
#
#            #for k, st_with_model in shadow_training_sets_with_model.items():
#            #    self.load_one_dataset_with_model(
#            #            st_with_model,
#            #            sensor_columns=selected_columns,
#            #            is_multichannel=is_multichannel,
#            #            )
#
#        train_pipes = [
#            s.as_feature_extracting_datapipe(
#                self.dataset.batch_size,
#                batches_per_epoch=self.dataset.batches_per_epoch,
#                device=self.device,
#                multi_channel=is_multichannel,
#            )
#            for k, s in shadow_training_sets_with_model.items()
#        ]
#
#        training_datapipe = Concater(*train_pipes).shuffle()
#        training_datapipe = next(iter(shadow_training_sets_with_model.values())).patch_attributes(
#            training_datapipe
#        )
#        # --- End Training data pipe setup ---
#
#        # -------------
#        # -- Setup CV
#        shadow_cv_sets_with_model: Dict[int, DatasetWithModel] = dict()
#        for sm_k, sm_pretrained_model in self.shadow_model_map.items():
#            if sm_k not in cv_shadow_models:
#                continue
#            # sm_k and ii should be the same
#            shadow_cv_sets_with_model[sm_k] = DatasetWithModel(
#                p_tuples=unique_shadow_sets,
#                # Pretraining participants from this model are the positive class (member)
#                reindex_map={
#                    s[1]: 1 if s in sm_pretrain_sets_d[sm_k] else 0
#                    for s in unique_shadow_sets
#                },
#                dataset_cls=dataset_cls,
#                member_model=sm_pretrained_model,
#                mc_n_channels_to_sample=self.n_channels_to_sample,
#                model_output_key=self.model_output_key,
#            )
#
##            shadow_cv_sets_with_model[sm_k] = self.load_one_dataset_with_model(
##                shadow_cv_sets_with_model[sm_k],
##                selected_columns,
##                is_multichannel=is_multichannel,
##            )
#            
#        cv_st_with_models = Parallel(n_jobs=len(shadow_cv_sets_with_model), 
#                                     backend='loky',
#                                     verbose=10)(
#                delayed(self.load_one_dataset_with_model)(st_with_model, 
#                                                          sensor_columns=selected_columns,
#                                                          is_multichannel=is_multichannel)
#                for k, st_with_model in shadow_cv_sets_with_model.items()
#                )
#        shadow_cv_sets_with_model = dict(zip(shadow_cv_sets_with_model.keys(), cv_st_with_models))
#
#        cv_datapipe = Concater(
#            *[
#                s.as_feature_extracting_datapipe(
#                    self.dataset.batch_size,
#                    batches_per_epoch=self.dataset.batches_per_epoch,
#                    device=self.device,
#                    multi_channel=is_multichannel,
#                )
#                for k, s in shadow_cv_sets_with_model.items()
#            ]
#        ).shuffle()
#        cv_datapipe = next(iter(shadow_cv_sets_with_model.values())).patch_attributes(
#            cv_datapipe
#        )
#        # --- End CV data pipe setup ---
#
#        # -------------
#        # -- Set up target as the test set
#        unique_target_sets: List[Tuple] = list(
#            set(sum(target_pretrain_sets_d.values(), list()))
#        )
#        # Sets used by target models
#        test_sets_with_model: Dict[int, DatasetWithModel] = dict()
#        for t_k, t_pretrained_model in self.target_model_map.items():
#            target_pretrain_sets = target_pretrain_sets_d[t_k]
#            test_sets_with_model[t_k] = DatasetWithModel(
#                p_tuples=unique_target_sets,
#                reindex_map={
#                    s[1]: 1 if s in target_pretrain_sets else 0
#                    for s in unique_target_sets
#                },
#                dataset_cls=dataset_cls,
#                member_model=t_pretrained_model,
#                mc_n_channels_to_sample=self.n_channels_to_sample,
#                model_output_key=self.model_output_key,
#            )
#
#            test_sets_with_model[t_k] = self.load_one_dataset_with_model(
#                test_sets_with_model[t_k],
#                selected_columns,
#                is_multichannel=is_multichannel,
#            )
#
#        test_batch_size = (
#            self.dataset.batch_size
#            if self.dataset.batch_size_eval is None
#            else self.dataset.batch_size_eval
#        )
#        test_batches_per_epoch = (
#            self.dataset.batches_per_epoch
#            if self.dataset.batches_per_eval_epoch is None
#            else self.dataset.batches_per_eval_epoch
#        )
#        test_datapipe = Concater(
#            *[
#                s.as_feature_extracting_datapipe(
#                    batch_size=test_batch_size,
#                    batches_per_epoch=test_batches_per_epoch,
#                    device=self.device,
#                    multi_channel=is_multichannel,
#                )
#                for k, s in test_sets_with_model.items()
#            ]
#        ).shuffle()
#
#        test_datapipe = next(iter(test_sets_with_model.values())).patch_attributes(
#            test_datapipe
#        )
#        # --- End Test data pipe setup ---
#
#        self.dataset_with_model_d = dict(
#            train=shadow_training_sets_with_model,
#            cv=shadow_cv_sets_with_model,
#            test=test_sets_with_model,
#        )
#        ret_datapipe_d = dict(
#            train=training_datapipe, cv=cv_datapipe, test=test_datapipe
#        )
#        self.dataset_map = ret_datapipe_d
#
#        return dict(ret_datapipe_d), dict(ret_datapipe_d), dict(ret_datapipe_d)

    def get_target_labels(self):
        target_label_map = {0: "non_member", 1: "member"}
        target_labels = list(target_label_map.values())
        return target_label_map, target_labels


@with_logger
@dataclass
class InfoLeakageExperiment(bxp.Experiment):
    task: ShadowClassifierMembershipInferenceFineTuningTask = subgroups(
        {
            "shadow_classifier_mi": ShadowClassifierMembershipInferenceFineTuningTask,
            "one_model_mi": OneModelMembershipInferenceFineTuningTask,
            "reid": ReidentificationTask,
        },
        default="shadow_classifier_mi",
    )

    attacker_model: ft.FineTuningModel = ft.FineTuningModel(
        fine_tuning_method="1d_linear"
    )
    task_dataset_dir: Optional[str] = field(default=None)

    # ---
    result_model: Optional[torch.nn.Module] = field(default=None, init=False)

    shadow_model_results_map: dict = field(default=None, init=False)
    shadow_model_map: dict = field(default=None, init=False)
    target_model: dict = field(default=None, init=False)
    target_labels: dict = field(default=None, init=False)
    target_label_map: dict = field(default=None, init=False)
    target_model_results: dict = field(default=None, init=False)
    dataset_map: dict = field(default=None, init=False)
    dl_map: dict = field(default=None, init=False)
    eval_dl_map: dict = field(default=None, init=False)

    outputs_map: dict = field(default=None, init=False)
    output_cm_map: dict = field(default=None, init=False)
    performance_map: dict = field(default=None, init=False)
    clf_str_map: dict = field(default=None, init=False)

    def initialize(self):
        if getattr(self, "initialized", False):
            return self

        # Task needs to load the data in either 2d or 1d for the attacker model
        self.task.method = self.attacker_model.fine_tuning_method

        # Load datasets from task
        (
            self.dataset_map,
            self.dl_map,
            self.eval_dl_map,
        ) = self.task.make_dataset_and_loaders()

        # Get target label information from the task
        self.target_label_map, self.target_labels = self.task.get_target_labels()
        num_classes = len(self.target_labels)
        logger.info(f"Labels for {num_classes} classes: {self.target_labels}")
        
        self.task.get_target_rates()

        T, C = self.task.get_member_model_output_shape()

        if "2d" in self.attacker_model.fine_tuning_method:
            if hasattr(self.task, "get_num_channels"):
                n_sensors = self.task.get_num_channels()
            else:
                n_sensors = len(self.dataset_map["train"].selected_columns)

            attack_arr_shape = n_sensors, T, C
            self.model_kws = dict(input_shape=attack_arr_shape, outputs=num_classes)
            self.result_model = MultiChannelAttacker(**self.model_kws)
        else:
            #attack_arr_shape = T * C
            attack_arr_shape = 2 if self.task.model_output_key == 'bce_loss' else T * C 
            self.model_kws = dict(
                input_size=attack_arr_shape,
                outputs=num_classes,
                linear_hidden_n=self.attacker_model.linear_hidden_n,
                n_layers=self.attacker_model.n_layers,
                dropout_rate=self.attacker_model.dropout,
                batch_norm=self.attacker_model.batch_norm,
                pre_trained_model_output_key=self.task.model_output_key
                #pre_trained_model_output_key="x",
            )
            self.result_model = SingleChannelAttacker(**self.model_kws)

        criterion, target_key = self.task.make_criteria_and_target_key()

        logger.info(f"Criterion for {self.task.task_name}: {criterion} on {target_key}")

        self.trainer = ShadowClassifierTrainer(
            model_map=dict(model=self.result_model),
            opt_map=dict(),
            train_data_gen=self.dl_map["train"],
            cv_data_gen=self.eval_dl_map.get("cv"),
            input_key="signal_arr",
            learning_rate=self.task.learning_rate,
            early_stopping_patience=self.task.early_stopping_patience,
            lr_adjust_on_cv_loss=self.task.lr_adjust_patience is not None,
            lr_adjust_on_plateau_kws=dict(
                patience=self.task.lr_adjust_patience, factor=self.task.lr_adjust_factor
            ),
            weight_decay=self.task.weight_decay,
            target_key=target_key,
            criterion=criterion,
            device=self.task.device,
            squeeze_target=self.task.squeeze_target,
            cache_dataloaders=False,
        )
        self.initialized = True
        return self

    def train(self):
        self.attacker_model_train_results = self.trainer.train(self.task.n_epochs)

    def _eval_part(self, part_name, out_d,
                   class_labels,
                   class_val_to_label_d,
                   ):
        out_d = self.trainer.generate_outputs(**{part_name: self.eval_dl_map[part_name]})[part_name]
        #for part_name, out_d in outputs_map.items():
        logger.info(("-" * 20) + part_name + ("-" * 20))

        logger.debug("Parsing predictions")
        preds_arr: np.ndarray = out_d["preds"].squeeze()
        preds_df = pd.DataFrame(preds_arr, columns=class_labels)

        logger.debug("Parsing actuals")
        if out_d["actuals"].squeeze().ndim == 2:
            y_arr = out_d["actuals"].squeeze().argmax(-1).reshape(-1)
        else:
            y_arr = out_d["actuals"].reshape(-1)

        logger.debug("Storing into series")
        y_s = pd.Series(y_arr, index=preds_df.index, name="actuals")
        # print(y_s)
        y_label_s = y_s.map(class_val_to_label_d)

        logger.debug("Calculating confusion matric")
        cm = confusion_matrix(
            y_label_s, preds_df.idxmax(1), labels=preds_df.columns.tolist()
        )

        output_cm = pd.DataFrame(
            cm, columns=preds_df.columns, index=preds_df.columns
        ).to_json()

        logger.debug("Calculating classification report")
        output_clf_report = classification_report(
            y_label_s, preds_df.idxmax(axis=1)
        )
        print(output_clf_report)

        output_performance = utils.multiclass_performance(
            y_label_s, preds_df.idxmax(axis=1)
        )
        logger.info(output_performance)

        return part_name, output_cm, output_clf_report, output_performance

    def eval_parallel(self):
        class_labels = self.target_labels
        class_val_to_label_d = self.target_label_map
        
        result_tuples_l = Parallel(backend='loky', n_jobs=3, verbose=10)(
                delayed(self._eval_part)(part_name=part_name,
                                         out_d=out_d,
                                         class_labels=class_labels, 
                                         class_val_to_label_d=class_val_to_label_d)
                for part_name, out_d in outputs_map.items()
                )
        output_cmap = {t[0]: t[1] for t in result_tuples_l}
        output_clf_report_map = {t[0]: t[2] for t in result_tuples_l}
        output_performance_map = {t[0]: t[3] for t in result_tuples_l}

        self.outputs_map, self.output_cm_map, self.performance_map, self.clf_str_map = [
            outputs_map,
            output_cm_map,
            output_performance_map,
            output_clf_report_map,
        ]

    def eval(self):
        outputs_map = self.trainer.generate_outputs(**self.eval_dl_map)

        # TODO: This will not work for other tasks
        # class_val_to_label_d, class_labels = self.dataset_map['train'].get_target_labels()
        # if isinstance(self.task, ReidentificationTask):
        #     class_val_to_label_d, class_labels = self.dataset_map['train'].get_target_labels()
        # else:
        #    class_val_to_label_d = {0: 'nonmember', 1: 'member'}
        # class_labels = list(class_val_to_label_d.values())
        class_labels = self.target_labels
        class_val_to_label_d = self.target_label_map
        
        output_cm_map = dict()
        output_clf_report_map = dict()
        output_performance_map = dict()

        for part_name, out_d in outputs_map.items():
            logger.info(("-" * 20) + part_name + ("-" * 20))
            preds_arr: np.ndarray = out_d["preds"].squeeze()
            # print(np.unique(preds_arr))
            # if preds_arr.ndim == 1:
            #    n_preds = 1 - preds_arr
            #    preds_arr = np.concatenate([n_preds.reshape(-1, 1), preds_arr.reshape(-1, 1)], axis=1)
            preds_df = pd.DataFrame(preds_arr, columns=class_labels)
            logger.debug("Parsing actuals")
            if out_d["actuals"].squeeze().ndim == 2:
                y_arr = out_d["actuals"].squeeze().argmax(-1).reshape(-1)
            else:
                y_arr = out_d["actuals"].reshape(-1)
            logger.debug("Storing into series")
            y_s = pd.Series(y_arr, index=preds_df.index, name="actuals")
            # print(y_s)
            y_label_s = y_s.map(class_val_to_label_d)
            logger.debug("Calculating confusion matric")
            cm = confusion_matrix(
                y_label_s, preds_df.idxmax(1), labels=preds_df.columns.tolist()
            )
            # cm = confusion_matrix(y_label_s, preds_df.values.argmax(1))#, labels=preds_df.columns.tolist())
            output_cm_map[part_name] = pd.DataFrame(
                cm, columns=preds_df.columns, index=preds_df.columns
            ).to_json()
            logger.debug("Calculating classification report")
            output_clf_report_map[part_name] = classification_report(
                y_label_s, preds_df.idxmax(axis=1)
            )
            print(output_clf_report_map[part_name])

            output_performance_map[part_name] = utils.multiclass_performance(
                y_label_s, preds_df.idxmax(axis=1)
            )
        logger.info(output_performance_map)

        self.outputs_map, self.output_cm_map, self.performance_map, self.clf_str_map = [
            outputs_map,
            output_cm_map,
            output_performance_map,
            output_clf_report_map,
        ]

    def run(self):
        self.initialize()
        if self.task_dataset_dir is not None:
            self.persist_task_dataset()
            return

        self.train()
        self.eval()
        self.save()

    def persist_task_dataset(self):
        Path(self.task_dataset_dir).mkdir(parents=True, exist_ok=True)
        for part_name, part_dl in self.eval_dl_map.items():
            part_p = os.path.join(self.task_dataset_dir, part_name)
            Path(part_p).mkdir(parents=True, exist_ok=True)
            for batch_i, batch_d in enumerate(
                tqdm(part_dl, desc="Persisting task dataset")
            ):
                batch_p = os.path.join(part_p, f"{batch_i}.pkl")
                with open(file=batch_p, mode="wb") as f:
                    pickle.dump(batch_d, f)

    def save(self):
        #####
        # Prep a results structure for saving - everything must be json serializable (no array objects)
        dataset_rates = {part: rate_s.to_dict() 
                         for part, rate_s in self.task.get_target_rates(normalize=False, as_series=True).items()}
        res_dict = self.create_result_dictionary(
            model_name=self.attacker_model.model_name,
            epoch_outputs=self.attacker_model_train_results,
            train_selected_columns=self.dataset_map[
                "train"
            ].selected_columns,  # dataset_map['train'].selected_columns,
            # test_selected_flat_indices=dataset_map['test'].selected_flat_indices,
            # selected_flat_indices={k: d.selected_flat_indices for k, d in dataset_map.items()},
            selected_flat_indices={
                k: d.selected_levels_df.to_json()
                if hasattr(d, "selected_levels_df")
                else None
                for k, d in self.dataset_map.items()
            },
            dataset_rates=dataset_rates,
            best_model_epoch=getattr(self.trainer, "best_model_epoch", None),
            num_trainable_params=utils.number_of_model_params(self.result_model),
            num_params=utils.number_of_model_params(
                self.result_model, trainable_only=False
            ),
            model_kws=self.model_kws,
            classification_summaries=self.clf_str_map,
            **self.performance_map,
            # **eval_res_map,
            # pretrained_result_input=vars(self.pretrained_result_input),
            experiment_options=vars(self),
            task_options=vars(self.task),
            dataset_options=vars(self.task.dataset),
            # auto_selected_train_sets=getattr(self, 'auto_selected_train_sets'),
            # dataset_options=vars(self.dataset),
            result_output_options=vars(self.result_output),
            model_options=vars(self.attacker_model),
            # pretrained_results=self.pretraining_results,
            confusion_matrices=self.output_cm_map,
        )
        uid = res_dict["uid"]
        name = res_dict["name"]

        self.save_results(
            self.result_model,
            result_file_name=name,
            result_output=self.result_output,
            model_file_name=uid,
            res_dict=res_dict,
        )

        return self.trainer, self.performance_map


@attr.s
class ShadowClassifierTrainer(bmp.Trainer):
    input_key = attr.ib("signal_arr")
    squeeze_target = attr.ib(False)

    squeeze_first = True

    def loss(self, model_output_d, input_d, as_tensor=True):
        target = (
            input_d[self.target_key].squeeze()
            if self.squeeze_target
            else input_d[self.target_key]
        )
        if isinstance(self.criterion, (torch.nn.BCEWithLogitsLoss, torch.nn.BCELoss)):
            target = target.float()

        model_output_d = model_output_d.reshape(*target.shape)

        crit_loss = self.criterion(
            model_output_d.float()
            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss)
            else model_output_d,
            target.float(),
        )
        return crit_loss

    def _eval(self, epoch_i, dataloader, model_key="model"):
        """
        trainer's internal method for evaluating losses,
        snapshotting best models and printing results to screen
        """
        model = self.model_map[model_key].eval()
        self.best_cv = getattr(self, "best_cv", np.inf)

        preds_l, actuals_l, loss_l = list(), list(), list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):
                    input_d = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                               for k, v in _x.items()}
                    # input_arr = input_d[self.input_key]
                    # actual_arr = input_d[self.target_key]
                    m_output = model(input_d)

                    # actuals = input_d[self.target_key]

                    # loss = self.criterion(preds, actuals)
                    loss = self.loss(m_output, input_d)

                    loss_l.append(loss.detach().cpu().item())

                    pbar.update(1)

                mean_loss = np.mean(loss_l)
                desc = "Mean Eval Loss: %.5f" % mean_loss
                reg_l = 0.0
                if self.model_regularizer is not None:
                    reg_l = self.model_regularizer(model)
                    desc += " (+ %.6f reg loss = %.6f)" % (reg_l, mean_loss + reg_l)

                overall_loss = mean_loss + reg_l

                if overall_loss < self.best_cv:
                    self.best_model_state = copy_model_state(model)
                    self.best_model_epoch = epoch_i
                    self.best_cv = overall_loss
                    desc += "[[NEW BEST]]"

                pbar.set_description(desc)

        self.model_map["model"].train()
        return dict(primary_loss=overall_loss, cv_losses=loss_l)

    def train_inner_step(self, epoch_i, data_batch):
        """
        Core training method - gradient descent - provided the epoch number
        and a batch of data and must return a dictionary of losses.
        """
        res_d = dict()

        model = self.model_map["model"].to(self.device)
        optim = self.opt_map["model"]
        model = model.train()

        model.zero_grad()
        optim.zero_grad()

        input_d = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                   for k, v in data_batch.items()}
        m_output = model(input_d)

        crit_loss = self.loss(m_output, input_d)
        res_d["crit_loss"] = crit_loss.detach().cpu().item()

        if self.model_regularizer is not None:
            reg_l = self.model_regularizer(model)
            res_d["bwreg"] = reg_l.detach().cpu().item()
        else:
            reg_l = 0

        loss = crit_loss + reg_l
        res_d["total_loss"] = loss.detach().cpu().item()
        loss.backward()
        optim.step()
        model = model.eval()
        return res_d

    def generate_outputs_from_model_inner_step(
        self,
        model,
        data_batch,
        criterion=None,
        input_key="signal_arr",
        target_key="text_arr",
        device=None,
    ):
        # X = data_batch[input_key].to(device)

        # if self.squeeze_first:
        #    X = X.squeeze()

        with torch.no_grad():
            model.eval()
            model.to(device)
            input_d = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in data_batch.items()}
            preds = model(input_d)

            # loss_v = self.loss(preds, input_d)
        # score_d = self.score_training(m_d, as_tensor=True)
        eval_d = dict(
            preds=preds,
            actuals=input_d[self.target_key],  # loss=torch.tensor(loss_v)
        )  # dict(**score_d, **loss_d)

        return eval_d


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser(description="Information Leakage Experiments")
    parser.add_arguments(InfoLeakageExperiment, dest="info_leakage")
    args = parser.parse_args()
    tl: InfoLeakageExperiment = args.info_leakage
    tl.run()
