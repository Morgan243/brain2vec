import numpy as np
import os
from mmz import utils
from mmz import experiments as mxp
from mmz import models as bmm

from brain2vec import datasets
from brain2vec.datasets import harvard_sentences, northwestern_words
from brain2vec.models import brain2vec
from brain2vec.experiments import load_results_to_frame, ResultParsingOptions

from simple_parsing import ArgumentParser, choice, subgroups
from dataclasses import dataclass, field
from functools import partial
from typing import Optional
import torch
from brain2vec.experiments.fine_tune import FineTuneResultParsingOptions
from dataclasses import dataclass, make_dataclass
import pandas as pd

logger = utils.get_logger('semi_supervised')
#with_logger = utils.with_logger(prefix_name=__name__)

from typing import Union, List, ClassVar, Dict




@dataclass
class PretrainResultParsingOptions(ResultParsingOptions):
    result_file: Optional[Union[str, List[str]]] = None
    config_labels: Optional[dict] = None
    print_results: Optional[bool] = False

    indicator_eval_d: Dict[str, str] = field(default_factory=lambda : {
        'Large SSL': "quant_num_vars == 160 and n_encoder_layers== 8",
        'Medium SSL': "quant_num_vars == 20 and n_encoder_layers== 8",
        'Small SSL': "quant_num_vars == 10 and n_encoder_layers== 4",
    })

    shapes_d: ClassVar[Dict[tuple,tuple]] = {
        ('UCSD', 4, 1, 1): (525000, 90),
        ('UCSD', 5, 1, 1): (425001, 70),
        ('UCSD', 10, 1, 1): (455000, 80),
        ('UCSD', 18, 1, 1): (450000, 175),
        ('UCSD', 19, 1, 1): (580000, 232),
        ('UCSD', 22, 1, 1): (490000, 94),
        ('UCSD', 28, 1, 1): (450001, 108)
    }

    loss_pub_name_d = dict(
        cv_bce_loss='Contsrastive Loss\n$L_c$',
        cv_perplexity='Diversity Loss\n$L_d$',
        cv_feature_pen='Feature Penalty\n$L_z$',

        codewords_used='$L_d \times$  Codebook Size',
    )

    def __post_init__(self):
        self.results_df = load_results_to_frame(
            self.result_file
        )

        # result_options_df, pretrain_exper_df, mean_train_acc_ctab, mean_cv_acc_ctab, mean_test_acc_ctab = FineTuneResultParsingOptions.preprocess_results_frame(results_df)
        #self.parsed_results = self.preprocess_results_frame(results_df, indicator_eval_d=self.indicator_eval_d)
        #print(self.parsed_results.result_options_df['task_name_pub_fmt_short'].value_counts())

        opt_df_map = {opt_col: self.results_df[opt_col].apply(pd.Series)
                      for opt_col in self.results_df.filter(like='_options').columns}

        #{k: v.shape for k, v in opt_df_map.items()}

        # --
        options_df = opt_df_map['model_options'].copy()
        for opt_name, _df in opt_df_map.items():
            if opt_name == 'model_options':
                continue
            keep_cols = list(set(_df.columns.tolist()) - set(options_df.columns.tolist()))
            options_df = options_df.join(_df[keep_cols].copy())

        self.options_df = options_df

        self.opt_cols = options_df.drop(['model_name', 'save_model_path'], axis=1).columns.tolist()

        self.result_options_df = self.results_df.join(options_df[self.opt_cols])

        self.result_options_df['parent_dir'] = self.result_options_df.path.apply(lambda s: os.path.split(s)[0])
        self.result_options_df['feature_extractor_layers_str'] = self.result_options_df.feature_extractor_layers.apply(eval).apply(lambda l: "|".join(".".join(str(_t) for _t in t) for t in l))
        self.opt_cols.append('feature_extractor_layers_str')

        cv_df = self.result_options_df['cv'].apply(pd.Series)
        self.result_options_df['cv_batch_total_loss_mean'] = cv_df[
            ['bce_loss', 'perplexity', 'feature_pen']
        ].apply(
            lambda r: pd.DataFrame(r.to_dict()).sum(axis=1).mean()
            if r.notnull().all() else np.nan,
            axis=1)
        self.result_options_df['cv_batch_bce_loss_mean'] = cv_df.bce_loss.apply(np.mean)
        self.result_options_df['cv_batch_perplexity_loss_mean'] = cv_df.perplexity.apply(np.mean)
        self.result_options_df['cv_batch_feature_pen_loss_mean'] = cv_df.feature_pen.apply(np.mean)
        self.result_options_df['cv_batch_accuracy_loss_mean'] = cv_df.accuracy.apply(np.mean)

        self.result_options_df['cv_batch_total_loss_std'] = cv_df[
            ['bce_loss', 'perplexity', 'feature_pen']
        ].apply(
            lambda r: pd.DataFrame(r.to_dict()).sum(axis=1).std()
            if r.notnull().all() else np.nan,
            axis=1)
        self.result_options_df['cv_batch_bce_loss_std'] = cv_df.bce_loss.apply(np.std)
        self.result_options_df['cv_batch_perplexity_loss_std'] = cv_df.perplexity.apply(np.std)
        self.result_options_df['cv_batch_feature_pen_loss_std'] = cv_df.feature_pen.apply(np.std)
        self.result_options_df['cv_batch_accuracy_loss_std'] = cv_df.accuracy.apply(np.std)


        if 'model_output_shape' in self.result_options_df.columns:
            self.result_options_df['model_output_shape_str'] = self.result_options_df.model_output_shape\
                .apply(lambda s: tuple(s) if isinstance(s, list) else tuple())\
                .apply(lambda l: "x".join(str(t) for t in l[1:]) if len(l) > 0 else "NA"
                       )
            self.opt_cols.append('model_output_shape_str')

        if self.config_labels is not None:
            self.result_options_df = self.result_options_df.assign(**{lvl_name: self.result_options_df.eval(eval_str)
                                                            for lvl_name, eval_str in self.config_labels.items()})
            if 'training_complete' in self.result_options_df.columns:
                self.result_options_df['training_complete'] = self.result_options_df['training_complete'].fillna(True)
                # result_options_df = result_options_df[result_options_df['training_complete']]

            self.result_options_df['has_config_name'] = self.result_options_df[self.config_labels.keys()].sum(axis=1)
            self.result_options_df['Pretraining Configuration'] = self.result_options_df[self.config_labels.keys()].idxmax(
                axis=1).astype(pd.CategoricalDtype(categories=self.config_labels.keys()))

            self.result_options_df['Pretraining Configuration'] = np.where(self.result_options_df['has_config_name'],
                                                                           self.result_options_df['Pretraining Configuration'],
                                                                           'UNK')
            self.result_options_df['Pretraining Configuration'].value_counts()



@dataclass
class FineTuneResultParsingOptions(ResultParsingOptions):
    result_file: Optional[Union[str, List[str]]] = None
    print_results: Optional[bool] = False

    shapes_d: ClassVar[Dict[tuple,tuple]] = {
        ('UCSD', 4, 1, 1): (525000, 90),
        ('UCSD', 5, 1, 1): (425001, 70),
        ('UCSD', 10, 1, 1): (455000, 80),
        ('UCSD', 18, 1, 1): (450000, 175),
        ('UCSD', 19, 1, 1): (580000, 232),
        ('UCSD', 22, 1, 1): (490000, 94),
        ('UCSD', 28, 1, 1): (450001, 108)
    }

    n_sensors_d: ClassVar[Dict[str, int]] = {f"{k[0]}-{k[1]}": v[1] for k, v in shapes_d.items()}

    all_tuples: ClassVar[List] = harvard_sentences.HarvardSentences.make_tuples_from_sets_str('*')
    all_set_strs: ClassVar[set] = {f"{t[0]}-{t[1]}" for t in all_tuples}
    pt_str_to_pub_id: ClassVar[Dict[str, int]] = {f"{t[0]}-{t[1]}": ii + 1 for ii, t in enumerate(sorted(all_tuples))}

    ParsedResults: ClassVar = make_dataclass("ParsedResults",
                                             ['result_options_df',
                                              'pretrain_exper_df',
                                              'mean_train_acc_ctab',
                                              'mean_cv_acc_ctab',
                                              'mean_test_acc_ctab'])



@dataclass
class SemisupervisedCodebookTaskOptions(mxp.TaskOptions):
    task_name: str = "semisupervised_codebook_training"
    ppl_weight: float = 1.
    feature_pen_weight: float = .0001
    pos_enc_pen_weight: float = 0.


@dataclass
class SemiSupervisedExperiment(mxp.Experiment):
    model: bmm.ModelOptions = subgroups(
        {"brain2vec": partial(brain2vec.Brain2VecOptions),
         #'dummy': brain2vec.Brain2VecOptions
         },
        default='brain2vec'
    )

    dataset: datasets.DatasetOptions = subgroups(
        {
            "hvs": partial(harvard_sentences.HarvardSentencesDatasetOptions,
                           pre_processing_pipeline='random_sample',
                           sensor_columns='good_for_participant'
                           ),
             # Not actually tested
             #"nww": northwestern_words.NorthwesternWordsDatasetOptions
        },
        default='hvs')

    task: mxp.TaskOptions = subgroups(
        {"semi_supervised": partial(SemisupervisedCodebookTaskOptions),
         "dummy": SemisupervisedCodebookTaskOptions},
        default='semi_supervised')
        #default=SemisupervisedCodebookTaskOptions())

    skip_if_results_exists_in: bool = False

    def check_if_results_exist_in(self, check_existing_p):
        import os
        import pandas as pd
        from glob import glob
        check_existing_json_glob = os.path.join(check_existing_p, '*.json')
        results_df = load_results_to_frame(check_existing_json_glob)

        datasets_df = results_df.dataset_options.apply(pd.Series)

        this_train_sets = set(harvard_sentences.HarvardSentences.make_tuples_from_sets_str(self.dataset.train_sets))

        def check_for_match(train_sets_str):
            train_sets = set(harvard_sentences.HarvardSentences.make_tuples_from_sets_str(train_sets_str))
            return this_train_sets == train_sets

        print("This train set: " + str(this_train_sets))
        print(f"Checking sets: {' | '.join(datasets_df.train_sets.unique())}")
        matching_train_set = datasets_df.train_sets.apply(check_for_match)
        if matching_train_set.any():
            print(f"Found matching: {results_df[matching_train_set].to_dict()}")
            return True
        else:
            print("No matching found")
        return False

    def run(self):
        if self.skip_if_results_exists_in and self.check_if_results_exist_in(self.result_output.result_dir):
            print("NOT RUNNING SINCE RESULTS ARE EXISTED")
            return None, None

        # Reduce default test size for sklearn train/test split from 0.25 to 0.2
        dataset_map, dl_map, eval_dl_map = self.dataset.make_datasets_and_loaders(train_split_kws=dict(test_size=0.2))
        model, model_kws = self.model.make_model(dataset_map['train'])

        # Shake out any forward pass errors now by running example data through model - the model has a small random
        # tensor t_in that can be pass in
        with torch.no_grad():
            model(model.t_in)

        # Default lr reduce to False, only setup if at patience is set
        trainer_kws = dict(lr_adjust_on_cv_loss=False)
        if self.task.lr_adjust_patience is not None:
            logger.info(f"Configuring LR scheduler for model: patience={self.task.lr_adjust_patience}")
            lr_schedule_kws = dict(patience=self.task.lr_adjust_patience, factor=self.task.lr_adjust_factor)
            trainer_kws.update(dict(lr_adjust_on_plateau_kws=lr_schedule_kws,
                                    lr_adjust_on_cv_loss=True,
                                    # Needs to match a model name in the model map passed to trainer below
                                    model_name_to_lr_adjust='model'))

        # Initialize a result structure for snaptshotting and final save (later)
        snapshot_res = self.create_result_dictionary()

        # Grab the generated uid and name to use them as file names
        uid = snapshot_res['uid']
        name = snapshot_res['name']
        def on_new_best(_trainer, training_complete=False, **extra_eval_res_map):
            logger.info(f"Snapshotting: {name}")
            snapshot_res.update(dict(
                model_name=self.model.model_name,
                epoch_outputs=_trainer.epoch_res_map,
                train_selected_columns=dataset_map['train'].selected_columns,
                selected_flat_indices={k: d.selected_levels_df.to_json() for k, d in dataset_map.items()},
                best_model_epoch=trainer.best_model_epoch,
                num_trainable_params=utils.number_of_model_params(model),
                num_params=utils.number_of_model_params(model, trainable_only=False),
                model_kws=model_kws,
                model_output_shape=list(model.t_feat_o.shape),
                **extra_eval_res_map,
                model_options=vars(self.model),
                task_options=vars(self.task),
                dataset_options=vars(self.dataset),
                result_output_options=vars(self.result_output),
                training_complete=training_complete,
                experiment_yaml=self.dumps_yaml()
            ))
            self.save_results(model, result_file_name=name, result_output=self.result_output,
                              model_file_name=uid, res_dict=snapshot_res)
            model.to(self.task.device)



        trainer = brain2vec.Brain2VecTrainer(model_map=dict(model=model), opt_map=dict(),
                                             train_data_gen=dl_map['train'], cv_data_gen=eval_dl_map['cv'],
                                             learning_rate=self.task.learning_rate,
                                             early_stopping_patience=self.task.early_stopping_patience,
                                             device=self.task.device,
                                             ppl_weight=self.task.ppl_weight,
                                             feature_pen_weight=self.task.feature_pen_weight,
                                             pos_enc_pen_weight=self.task.pos_enc_pen_weight,
                                             new_best_cv_cb=on_new_best,
                                             **trainer_kws)

        # For some reason the codebook indices isn't always on the right device... so this seems to help force it over
        #trainer.model_map['model'].quantizer.codebook_indices = trainer.model_map['model'].quantizer.codebook_indices.to(trainer.device)

        #trainer.squeeze_first = False

        #####
        # Train
        losses = trainer.train(self.task.n_epochs)

        # reload the best model from memory
        model.load_state_dict(trainer.get_best_state())

        #####
        # Produce predictions and score them
        model.eval()

        # Produce mapping from dataloader names (train/cv/test) to dataframe of batch eval losses
        # This is different from a classification or other model - don't have a way to easily produce stats aggregated
        # across the whole dataset
        eval_res_map = {k: trainer.eval_on(_dl).to_dict(orient='list') for k, _dl in eval_dl_map.items()}

        on_new_best(trainer, True, **eval_res_map)

        # Create the dictionary that will be json serialized as the results
        #snapshot_res.update(dict(
        #    model_name=self.model.model_name,
        #    epoch_outputs=losses,
        #    train_selected_columns=dataset_map['train'].selected_columns,
        #    selected_flat_indices={k: d.selected_levels_df.to_json() for k, d in dataset_map.items()},
        #    best_model_epoch=trainer.best_model_epoch,
        #    num_trainable_params=utils.number_of_model_params(model),
        #    num_params=utils.number_of_model_params(model, trainable_only=False),
        #    model_kws=model_kws,
        #    **eval_res_map,
        #    model_options=vars(self.model),
        #    task_options=vars(self.task),
        #    dataset_options=vars(self.dataset),
        #    result_output_options=vars(self.result_output),
        #    experiment_yaml=self.dumps_yaml()
        #))


        #self.save_results(model, result_file_name=name, result_output=self.result_output,
        #                  model_file_name=uid, res_dict=snapshot_res)

        return trainer, eval_res_map



if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(SemiSupervisedExperiment, dest='semi_supervised')
    args = parser.parse_args()
    experiment: SemiSupervisedExperiment = args.semi_supervised
    logger.info(f"EXPERIMENT: {experiment}")
    experiment.run()
