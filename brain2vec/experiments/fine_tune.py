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
from typing import ClassVar, Union

import attr

from mmz import utils

from mmz import experiments as bxp
from mmz import models as zm

from brain2vec import models as bmp
from brain2vec import datasets
from brain2vec import experiments
from dataclasses import dataclass
from brain2vec.models import base_fine_tuners as ft_models
import json
from simple_parsing import subgroups
#from ecog_speech import result_parsing

from brain2vec.datasets.harvard_sentences import HarvardSentencesDatasetOptions, HarvardSentences

from mmz.models import copy_model_state

logger = utils.get_logger(__name__)

# Override to make the result parsing options optional in this script
@dataclass
class TransferLearningResultParsingOptions(experiments.ResultParsingOptions):
    result_file: Optional[str] = None
    print_results: Optional[bool] = False


@dataclass
class SpeechDetectionFineTuningTask(bxp.TaskOptions):
    task_name: str = "speech_classification_fine_tuning"
    dataset: datasets.DatasetOptions = datasets.DatasetOptions('hvs', train_sets='AUTO-REMAINING',
                                                               flatten_sensors_to_samples=False,
                                                               pre_processing_pipeline='audio_gate')
    method: str = '2d_linear'
    squeeze_target: ClassVar[bool] = False

    def make_criteria_and_target_key(self):
        #pos_weight = torch.FloatTensor([1.0]).to(self.device)
        #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = torch.nn.BCELoss()
        target_key = 'target_arr'
        return criterion, target_key

    def make_datasets_and_loaders(self, **kws):
        return self.dataset.make_datasets_and_loaders(**kws, test_split_kws=dict(test_size=0.5))


@dataclass
class RegionDetectionFineTuningTask(bxp.TaskOptions):
    task_name: str = "region_classification_fine_tuning"
    dataset: datasets.DatasetOptions = HarvardSentencesDatasetOptions(train_sets='AUTO-REMAINING',
                                                                      flatten_sensors_to_samples=False,
                                                                      pre_processing_pipeline='region_classification')
    method: str = '2d_linear'

    squeeze_target: ClassVar[bool] = True

    def make_criteria_and_target_key(self):
        criterion = torch.nn.CrossEntropyLoss()
        target_key = 'target_arr'
        return criterion, target_key

    def make_datasets_and_loaders(self, **kws):
        return self.dataset.make_datasets_and_loaders(**kws, test_split_kws=dict(test_size=0.5))


@dataclass
class ParticipantIdentificationFineTuningTask(RegionDetectionFineTuningTask):
    task_name: str = "participant_classification_fine_tuning"
    dataset: datasets.DatasetOptions = HarvardSentencesDatasetOptions(train_sets='AUTO-PRETRAINING',
                                                                      flatten_sensors_to_samples=False,
                                                                      label_reindex_col='patient',
                                                                      pre_processing_pipeline='random_sample')


@dataclass
class PretrainParticipantIdentificationFineTuningTask(RegionDetectionFineTuningTask):
    task_name: str = "pretrain_participant_classification_fine_tuning"
    dataset: datasets.DatasetOptions = HarvardSentencesDatasetOptions(train_sets='AUTO-PRETRAINING',
                                                                      flatten_sensors_to_samples=False,
                                                                      label_reindex_col='patient',
                                                                      split_cv_from_test=False,
                                                                      pre_processing_pipeline='random_sample')

    def make_criteria_and_target_key(self):
        criterion = torch.nn.BCELoss()
        target_key = 'target_arr'
        return criterion, target_key

    def make_datasets_and_loaders(self, **kws):
        dataset_cls = datasets.BaseDataset.get_dataset_by_name(self.dataset.dataset_name)
        all_sets = dataset_cls.make_tuples_from_sets_str('*')

        # Assumes set to AUTO-PRETRAINING, so the train tuples are all the sets that the loaded model was pretrained on
        all_true_pos_set = kws.pop('train_p_tuples')
        # Thus, the sets not pretrained on are the true negatives
        all_true_neg_set = list(set(all_sets) - set(all_true_pos_set))

        debias_train_pos_set = all_true_pos_set

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
        debias_train_pos_set = [t for t in debias_train_pos_set if t not in debias_test_pos_set]

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

        return self.dataset.make_datasets_and_loaders(**kws,
                                                      train_p_tuples=debias_train_neg_set + debias_train_pos_set,
                                                      train_data_kws=dict(label_reindex_map=train_label_reindex_map),
                                                      test_p_tuples=debias_test_neg_set + debias_test_pos_set,
                                                      test_data_kws=dict(label_reindex_map=test_label_reindex_map))




@dataclass
class WordDetectionFineTuningTask(bxp.TaskOptions):
    task_name: str = "word_classification_fine_tuning"
    dataset: datasets.DatasetOptions = HarvardSentencesDatasetOptions(train_sets='AUTO-REMAINING',
                                                                               flatten_sensors_to_samples=False,
                                                                               pre_processing_pipeline='word_classification',
                                                                               split_cv_from_test=True)
    method: str = '2d_linear'

    squeeze_target: ClassVar[bool] = True

    def make_criteria_and_target_key(self):
        criterion = torch.nn.CrossEntropyLoss()
        target_key = 'target_arr'
        return criterion, target_key

    def make_datasets_and_loaders(self, **kws):
        # Hald the dataset, stratified on word label, belongs in the train data
        # Take the set of unique patient x sentence x word label, then stratify on label (ensure one of each is in train)
        train_split_kws = dict(keys=('patient', 'sent_code', 'label'), stratify='label', test_size=0.5)
        #test_split_kws = dict(keys=('patient', 'sample_ix', 'label'), stratify='label', test_size=0.20)
        if self.dataset.split_cv_from_test:
            # Alt with split from test = True: Split the second occurence
            test_split_kws = dict(keys=('patient', 'label'), test_size=0.50)
        else:
            # Split the second occurence amove cv and test
            test_split_kws = dict(keys=('patient', 'sample_ix', 'label'), stratify='label', test_size=0.20)

        return self.dataset.make_datasets_and_loaders(train_split_kws=train_split_kws,
                                                      test_split_kws=test_split_kws,
                                                      #split_cv_from_test=False,
                                                      **kws)


@dataclass
class FineTuningModel(bmp.DNNModelOptions):
    model_name = 'basic_fine_tuner'
    fine_tuning_method: str = '2d_linear'
    freeze_pretrained_weights: bool = True
    linear_hidden_n: int = 16
    n_layers: int = 1

    classifier_head: Optional[Union[torch.nn.Module, str]] = None

    def create_fine_tuning_model(self, pretrained_model,
                                 n_pretrained_output_channels=None, n_pretrained_output_samples=None,
                                 #fine_tuning_method='1d_linear',
                                 dataset: datasets.BaseDataset = None,
                                 fine_tuning_target_shape=None, n_pretrained_input_channels=None,
                                 n_pretrained_input_samples=256,
                                 #freeze_pretrained_weights=True,
                                 #classifier_head: torch.nn.Module = None
                                 ):
        n_pretrained_output_channels = pretrained_model.C if n_pretrained_output_channels is None else n_pretrained_output_samples
        n_pretrained_output_samples = pretrained_model.T if n_pretrained_output_samples is None else n_pretrained_output_samples

        n_pretrained_input_channels = n_pretrained_input_channels if dataset is None else len(dataset.selected_columns)
        # n_pretrained_input_samples = n_pretrained_input_samples if dataset is None else dataset.get_target_shape()
        fine_tuning_target_shape = dataset.get_target_shape() if fine_tuning_target_shape is None else fine_tuning_target_shape

        logger.info(f"Target shape from dataset: {fine_tuning_target_shape}")

        m = copy.deepcopy(pretrained_model)
        m.quantizer.codebook_indices = None
        if self.freeze_pretrained_weights:
            for param in m.parameters():
                param.requires_grad = False

        if self.fine_tuning_method == '1d_linear':
            # Very simple linear classifier head by default
            if self.classifier_head is None:
                h_size = 32
                classifier_head = torch.nn.Sequential(*[
                    zm.Reshape((n_pretrained_output_channels * n_pretrained_output_samples,)),
                    torch.nn.Linear(n_pretrained_output_channels * n_pretrained_output_samples, h_size),
                    # torch.nn.BatchNorm1d(h_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(h_size, h_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(h_size, fine_tuning_target_shape),
                    torch.nn.Sigmoid()
                ])
            else:
                classifier_head = self.classifier_head

            ft_model = ft_models.FineTuner(pre_trained_model=m, output_model=classifier_head,
                                           pre_trained_model_forward_kws=dict(features_only=True, mask=False),
                                           pre_trained_model_output_key='x',
                                           freeze_pre_train_weights=self.freeze_pretrained_weights)

        elif '2d_' in self.fine_tuning_method:
            if dataset is None:
                raise ValueError(f"A dataset is required for '2d_*' methods in order to see num sensors")
            #from ecog_speech.models import base_transformers
            from brain2vec.models.brain2vec import MultiChannelBrain2Vec

            hidden_enc = 'transformer' if self.fine_tuning_method == '2d_transformers' else 'linear'
            ft_model = MultiChannelBrain2Vec((n_pretrained_input_channels, n_pretrained_input_samples),
                                                             pretrained_model, outputs=fine_tuning_target_shape,
                                                             hidden_encoder=hidden_enc,
                                                             dropout=self.dropout, batch_norm=self.batch_norm,
                                                             linear_hidden_n=self.linear_hidden_n, n_layers=self.n_layers)
        else:
            raise ValueError(f"Unknown ft_method '{self.fine_tuning_method}'")

        return ft_model


@dataclass
class FineTuningExperiment(bxp.Experiment):
    pretrained_result_input: bxp.ResultInputOptions = None
    task: SpeechDetectionFineTuningTask = subgroups(
        {'speech_detection': SpeechDetectionFineTuningTask,
         'region_detection': RegionDetectionFineTuningTask,
         'word_detection': WordDetectionFineTuningTask,
         'participant_detection': ParticipantIdentificationFineTuningTask,
         'data_leakage': PretrainParticipantIdentificationFineTuningTask
         },
        default='region_detection')
        #default=RegionDetectionFineTuningTask())
    model: FineTuningModel = FineTuningModel()

    inspection_plot_path: Optional[str] = None
    n_rs_clf_iter: Optional[int] = None
    rs_clf: Optional[str] = 'svm'
    run_ml: bool = True

    @classmethod
    def load_pretrained_model_results(cls,
                              pretrained_result_input_path: str = None,
                              pretrained_result_model_base_path: str = None):
        #pretrained_result_model_base_path = pretrained_result_model_base_path if options is None else options.pretrained_result_model_base_path
        #pretrained_result_input_path = pretrained_result_input_path if options is None else options.pretrained_result_input_path

        assert_err = "pretrained_result_input_path must be populated as parameter or in options object"
        assert pretrained_result_input_path is not None, assert_err

        result_json = None
        #from ecog_speech.result_parsing import load_model_from_results
        from brain2vec.experiments import load_model_from_results

        if pretrained_result_model_base_path is None:
            pretrained_result_model_base_path = os.path.join(
                os.path.split(pretrained_result_input_path)[0], 'models')

        #result_path = pretrained_result_input_path
        #model_base_path = pretrained_result_model_base_path


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

        return pretrained_model, result_json

    #@property
    def make_pretrained_model(self):
        if not hasattr(self, 'pretrained_model'):
            self.pretrained_model, self.pretraining_results = self.load_pretrained_model_results(
                self.pretrained_result_input.result_file,
                self.pretrained_result_input.model_base_path)
        return self.pretrained_model, self.pretraining_results

    def make_fine_tuning_datasets_and_loaders(self, pretraining_sets=None):
        train_sets = None
        dataset_cls = datasets.BaseDataset.get_dataset_by_name(self.task.dataset.dataset_name)
        pretraining_sets = self.pretraining_results['dataset_options'][
            'train_sets'] if pretraining_sets is None else pretraining_sets
        unseen_sets = dataset_cls.make_remaining_tuples_from_selected(pretraining_sets)

        if self.task.dataset.train_sets == 'AUTO-REMAINING':
            logger.info(f"AUTO-REMAINING: pretrained on {pretraining_sets}, so fine tuning on {train_sets}")
            self.auto_selected_train_sets = unseen_sets
        # To use the same participant data that was used during pretraining
        elif self.task.dataset.train_sets == 'AUTO-PRETRAINING':
            train_sets = dataset_cls.make_tuples_from_sets_str(pretraining_sets)
            logger.info(f"AUTO-PRETRAINING: pretrained on {pretraining_sets}, so fine tuning on {train_sets}")
            self.auto_selected_train_sets = pretraining_sets
        else:
            logger.info(f"Will use specific pretraining: {self.task.dataset.train_sets}")

        # To use the remaining (holdout) participant data that was used during pretraining

        # Auto-Pretraining - get the positive sets
        # train_sets = dataset_cls.make_remaining_tuples_from_selected(pretraining_sets)

        return self.task.make_datasets_and_loaders(train_p_tuples=train_sets)

    def initialize(self):
        if getattr(self, 'initialized', False):
            return self

        # Pretrained model already prepared, parse from its results output
        self.pretrained_model, self.pretraining_results = self.make_pretrained_model()
        self.dataset_map, self.dl_map, self.eval_dl_map = self.make_fine_tuning_datasets_and_loaders()

        # Capture configurable kws separately, so they can be easily saved in the results at the end
        self.fine_tune_model_kws = dict()#dict(fine_tuning_method=self.task.method)
        self.fine_tune_model = self.model.create_fine_tuning_model(self.pretrained_model,
                                                                   dataset=self.dataset_map['train'],
                                                                   **self.fine_tune_model_kws)

        criterion, target_key = self.task.make_criteria_and_target_key()

        logger.info(f"Criterion for {self.task.task_name}: {criterion} on {target_key}")

        trainer_cls = TLTrainer
        self.trainer = trainer_cls(model_map=dict(model=self.fine_tune_model), opt_map=dict(),
                                   train_data_gen=self.dl_map['train'],
                                   cv_data_gen=self.eval_dl_map.get('cv'),
                                   input_key='signal_arr',
                                   learning_rate=self.task.learning_rate,
                                   early_stopping_patience=self.task.early_stopping_patience,
                                   lr_adjust_on_cv_loss=self.task.lr_adjust_patience is not None,
                                   lr_adjust_on_plateau_kws=dict(patience=self.task.lr_adjust_patience,
                                                                 factor=self.task.lr_adjust_factor),
                                   target_key=target_key,
                                   criterion=criterion,
                                   device=self.task.device,
                                   squeeze_target=self.task.squeeze_target
                                   )
        self.initialized = True
        return self

    def train(self):
        if getattr(self, 'trained', False):
            return self

        self.initialize()
        self.fine_tuning_results = self.trainer.train(self.task.n_epochs)
        self.fine_tune_model.load_state_dict(self.trainer.get_best_state())
        self.fine_tune_model.eval()

        self.trained = True
        return self

    def eval(self):
        outputs_map = self.trainer.generate_outputs(**self.eval_dl_map)

        #class_val_to_label_d = next(iter(self.dataset_map['train'].data_maps.values()))['index_source_map']
        #class_labels = [class_val_to_label_d[i] for i in range(len(class_val_to_label_d))]
        #out_d = self.outputs_map['train']
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

            #eval_res_map = {k: ft_trainer.eval_on(_dl).to_dict(orient='list') for k, _dl in eval_dl_map.items()}
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

        return outputs_map, output_cm_map, performance_map, clf_str_map

    def run(self):
        if self.inspection_plot_path is not None:
            self.make_pretrain_inspection_plots()
            return

        self.train()
        self.outputs_map, self.output_cm_map, self.performance_map, self.clf_str_map = self.eval()

        #####
        # Prep a results structure for saving - everything must be json serializable (no array objects)
        res_dict = self.create_result_dictionary(
            model_name=self.model.model_name,
            epoch_outputs=self.fine_tuning_results,
            train_selected_columns=self.dataset_map['train'].selected_columns,  # dataset_map['train'].selected_columns,
            #test_selected_flat_indices=dataset_map['test'].selected_flat_indices,
            #selected_flat_indices={k: d.selected_flat_indices for k, d in dataset_map.items()},
            selected_flat_indices={k: d.selected_levels_df.to_json() if hasattr(d, 'selected_levels_df') else None
                                   for k, d in self.dataset_map.items()},
            best_model_epoch=self.trainer.best_model_epoch,
            num_trainable_params=utils.number_of_model_params(self.fine_tune_model),
            num_params=utils.number_of_model_params(self.fine_tune_model, trainable_only=False),
            model_kws=self.fine_tune_model_kws,
            classification_summaries=self.clf_str_map,
            **self.performance_map,
            #**eval_res_map,
            pretrained_result_input=vars(self.pretrained_result_input),
            task_options=vars(self.task),
            dataset_options=vars(self.task.dataset),
            auto_selected_train_sets=getattr(self, 'auto_selected_train_sets'),
            #dataset_options=vars(self.dataset),
            result_output_options=vars(self.result_output),
            model_options=vars(self.model),
            pretrained_results=self.pretraining_results,
            confusion_matrices=self.output_cm_map
        )
        uid = res_dict['uid']
        name = res_dict['name']

        self.save_results(self.fine_tune_model, result_file_name=name, result_output=self.result_output,
                          model_file_name=uid, res_dict=res_dict)

        return self.trainer, self.performance_map

    def make_pretrain_inspection_plots(self):
        self.initialize()
        p = self.inspection_plot_path
        if p == 'AUTO':
            p = f"pretrain_inspect_{tl.task.task_name}_{tl.pretraining_results['uid']}.pdf"

        fig_d = self.make_plots()
        utils.figures_to_pdf(p, **{k: getattr(v, 'fig', v) for k, v in fig_d.items()})

        for k, f in fig_d.items():
            #fig = f.fig if isinstance(f, sns.FacetGrid) else f
            fig = getattr(f, 'fig', f)
            fig.savefig(p.replace('.pdf', f'_{k}.pdf'), bbox_inches='tight')

    def make_plots(self, class_label_remap=None):

        map_of_label_maps = {
            'speech_classification_fine_tuning': {'stim_pwrt': 'Speaking', 'silence_stim_pwrt_s': 'Not-Speaking'},
            'region_classification_fine_tuning' : {
                'listening_region_stim': 'Listening Region',
                'speaking_region_stim': 'Speaking Region',
                'mouthing_region_stim': 'Mouthing Region',
                 'imagining_region_stim': 'Imagining Region'},
        }

        t_dataset = self.dataset_map['train']
        class_val_to_label_d = next(iter(t_dataset.data_maps.values()))['index_source_map']
        print(class_val_to_label_d)
        if class_label_remap is None:
            class_label_remap = map_of_label_maps.get(self.task.task_name)

        if isinstance(class_label_remap, dict):
            class_val_to_label_d = {k: class_label_remap.get(v, v) for k, v in class_val_to_label_d.items()}
            print(class_val_to_label_d)

        device = 'cuda'

        from tqdm.auto import tqdm
        import torch

        fig_d = dict()
        fig, axs = matplotlib.pyplot.subplots(ncols=2)
        loss_df = pd.DataFrame(self.pretraining_results['epoch_outputs']).T
        title = "".join([(" " + word if ix % 7 else "\n" + word) for ix, word in enumerate(str(self).split())])
        ax = loss_df[['total_loss', 'cv_loss', 'accuracy']].plot(secondary_y='accuracy', logy=True,
                                                                 #title=f"Training of {self.task.task_name}",
                                                                 ax=axs[0])
        #ax.set_title(title, fontsize=8)
        fig.text(1., 1., title, fontsize=8)
        fig.suptitle(f"Training of {self.task.task_name}", y=1.)

        results_l = list()
        m = self.fine_tune_model.mc_from_1d.to(device).eval()
        _dl = self.dl_map['train']
        for batch_d in tqdm(_dl, desc="Running on train batches"):
            with torch.no_grad():
                dev_batch_d = {k: arr.to(device) for k, arr in batch_d.items()}
                feat_d = m.forward(dev_batch_d)  # , features_only=True, mask=False)
                results_l.append(
                    dict(signal_arr=batch_d['signal_arr'],
                         target_arr=batch_d['target_arr'],
                         **{n: arr.detach().cpu().numpy() for n, arr in feat_d.items()}))

        shape_str = str({k: a.shape for k, a in results_l[0].items()})
        print(f"SHAPES: \n {shape_str}")
        # all_x = np.concatenate([r['x'] for r in results_l])
        all_x = np.concatenate([r['output'] for r in results_l])
        all_x_df = pd.DataFrame(all_x.reshape(all_x.shape[0], -1))

        # If there was a target
        all_y = np.concatenate([r['target_arr'] for r in results_l])
        all_y_s = pd.Series(all_y.squeeze(), name='target_val')
        all_y_label_s = all_y_s.map(class_val_to_label_d).rename('target_label')

        corr_speak_df = all_x_df.corrwith(all_y_s)

        ax = corr_speak_df.hist(ax=axs[1])
        #ax.text(0.01, 0.5, "Hist. of flattened b2v features corrwith target ")
        ax.set_title("Hist. of flattened b2v features corrwith target ")
        #kws_str = "".join((" " if i % 5 else "\n") + (f"{str(k)} = {str(v)}") for i, (k, v) in self.pretraining_results['model_kws'].items())
        kws_str = "model options:\n"
        kws_str += "".join((" " + s if i % 5 else "\n" + s)
                          for i, s in enumerate(str(self.pretraining_results['model_kws']).split()))
        kws_str += "dataset options:\n"
        kws_str += "".join((" " + s if i % 5 else "\n" + s)
                           for i, s in enumerate(str(self.pretraining_results['dataset_options']).split()))
        #kws_str = str(self.pretraining_results['model_kws'])
        fig.text(1., 0.5, kws_str, fontsize=7)
        fig.text(1., 0.1, "Batch data shapes as multichannel pretrained:\n" + shape_str, fontsize=7)

        fig.tight_layout()
        #ax.set_title(kws_str, fontsize=10)
        fig_d['pretrain_subplots'] = fig


        _pca_df, fig, axs = viz.pca_and_scatter_plot(all_x_df, c=all_y_s, cmap='jet', alpha=0.3)
        fig_d['pca_2d_scatter'] = fig
        _pca_df, fig, g = viz.pca_and_pair_plot(all_x_df, all_y_label_s,
                                                  n_components=4)
        fig_d['pca_2d_pair'] = g
        s_df = _pca_df.sample(1000)
        fig, ax = viz.scatter3d(*[s_df.iloc[:, i] for i in range(3)], c=all_y_s.loc[s_df.index])
        fig_d['pca_3d_scatter'] = fig

        from sklearn.manifold import TSNE
        n_tsne = 3
        tsne = TSNE(n_components=n_tsne, verbose=3, n_jobs=4, n_iter=3000, learning_rate=100, perplexity=50)
        # tsne_pca_arr = tsne.fit_transform(_pca_df)
        print(all_x_df.head())
        print(all_x_df.dtypes)
        print(all_x_df.dtypes.value_counts())

        tsne_pca_arr = tsne.fit_transform(all_x_df)
        tsne_pca_df = pd.DataFrame(tsne_pca_arr, columns=[f"TSNE-{n}" for n in range(n_tsne)])

        all_y_label_s = all_y_s.map(class_val_to_label_d).rename('target_label')

        import seaborn as sns
        _plt_df = tsne_pca_df.join(all_y_label_s)
        # _plt_df = _plt_df.assign(target_label=_plt_df.target_val.map(class_val_to_label_d))
        plt_kws = dict(hue='target_label', diag_kws=dict(common_norm=False), palette='tab10', kind='hist',
                       diag_kind='kde', vars=_plt_df.columns.drop(['target_label']).tolist())
        g = sns.pairplot(_plt_df, **plt_kws)
        g.legend.set_title('Target Label')
        fig_d['tsne_pair'] = g

        ###
        if not self.run_ml:
            return fig_d

        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        from sklearn.cluster import KMeans
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import f1_score, make_scorer, classification_report
        from sklearn.pipeline import Pipeline

        from tempfile import mkdtemp
        from shutil import rmtree

        # from sklearn.externals.joblib import Memory
        from joblib import Memory
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA
        from sklearn.svm import LinearSVC

        #import autosklearn.classification
        #import autosklearn
        #automl = autosklearn.classification.AutoSklearnClassifier(
        #    time_left_for_this_task=120,
        #    per_run_time_limit=30,
        #    tmp_folder=f'/tmp/{self.task.task_name}_{self.pretraining_results["uid"]}',
        #)
        #automl.fit(all_x_df, all_y_s, dataset_name='breast_cancer')
        #print(automl.leaderboard())
        if self.n_rs_clf_iter is not None:
            #clf = 'hist'
            max_components = min(all_x_df.shape) // 8
            if self.rs_clf == 'svm':
                # the pipeline
                pipe = Pipeline([('reduce_dim', PCA()),
                                 ('clf', SVC(kernel='rbf'))])
                param_dist = dict(  # clf__max_leaf_nodes=range(20, 200),
                                           # clf__max_bins=range(4, 64),
                                           # clf__learning_rate=np.arange(0.0001, 3, 0.001)
                                           clf__C=np.arange(0.0001, 500, 0.001),
                                           # clf__max_features=range(5, 500, 10),
                                           reduce_dim__n_components=range(2, max_components)
                                       )
            elif self.rs_clf == 'hist':
                pipe = Pipeline([('reduce_dim', PCA()),
                                 ('clf', HistGradientBoostingClassifier())])
                param_dist = dict(
                    clf__max_leaf_nodes=range(10, 100),
                    clf__max_bins=range(4, 64),
                    clf__learning_rate=np.arange(0.0001, 2, 0.0001),
                    # clf__max_features=range(5, 500, 10),
                    reduce_dim__n_components=range(2, max_components)
                )
            elif self.rs_clf == 'sgd':
                pipe = Pipeline([
                    ('clf', sklearn.linear_model.SGDClassifier()) ])
                param_dist = dict(
                    clf__penalty=['l2', 'l1', 'elasticnet'],
                    clf__loss=['modified_huber', 'hinge'],
                    clf__alpha=np.arange(0.0001, 0.5, 0.0001)
                )

            rs = RandomizedSearchCV(pipe,
                                       param_distributions=param_dist,
                                       return_train_score=True,
                                       n_iter=self.n_rs_clf_iter, n_jobs=8, cv=3, verbose=2)
            rs.fit(all_x_df, all_y_s)
            print(rs.best_score_)

            rs_res_df = pd.DataFrame(rs.cv_results_).set_index('rank_test_score').sort_index()

            import seaborn as sns
            try:
                g = sns.pairplot(rs_res_df.reset_index(), vars=rs_res_df.filter(like='param_').columns.tolist(),
                             hue='mean_test_score')
                fig_d['rs_svm_pairplot'] = g.fig
                g.fig.suptitle(str(pipe))
                g.fig.tight_layout()
            except Exception as e:
                print(str(e))
                grp_k = rs_res_df.filter(like='param_').columns.tolist()
                ax = rs_res_df.groupby(grp_k).mean_test_score.mean().sort_values().plot.barh(figsize=(7, 15))
                fig = ax.get_figure()
                fig.tight_layout()
                fig_d['rs_barplot'] = fig

            print("Running learning curve plots...")
            fig, axes = viz.plot_learning_curve(rs.best_estimator_, str(rs.best_estimator_), all_x_df, all_y_s,
                                                n_jobs=2, cv=3, train_sizes=np.linspace(0.05, .5, 5))
            fig_d['best_estimator_learning_curve'] = fig

        return fig_d


@attr.s
class TLTrainer(bmp.Trainer):
    input_key = attr.ib('signal_arr')
    squeeze_target = attr.ib(False)
    squeeze_first = True

    def loss(self, model_output_d, input_d, as_tensor=True):
        target = (input_d[self.target_key].squeeze() if self.squeeze_target
                                   else input_d[self.target_key])
        if isinstance(self.criterion, (torch.nn.BCEWithLogitsLoss, torch.nn.BCELoss)):
            target = target.float()

        #model_output_d = model_output_d.reshape(*target.shape)

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
        input_arr = input_d[self.input_key]
        actual_arr = input_d[self.target_key]
        #m_output = model(input_arr)
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

    parser = ArgumentParser(description="ASPEN+MHRG Transfer Learning experiments")
    #parser.add_arguments("--pretrained_inspection_plots", action='store_true', default=False)
    parser.add_arguments(FineTuningExperiment, dest='transfer_learning')
    #parser.add_arguments(TransferLearningOptions, dest='transfer_learning')
    #parser.add_arguments(TransferLearningResultParsingOptions, dest='tl_result_parsing')
    args = parser.parse_args()
    tl: FineTuningExperiment = args.transfer_learning
    tl.run()
