from mmz import utils
from mmz import experiments as mxp
from mmz import models as bmm

from brain2vec import datasets
from brain2vec.datasets import harvard_sentences, northwestern_words
from brain2vec.models import brain2vec

from simple_parsing import ArgumentParser, choice, subgroups
from dataclasses import dataclass, field
import torch

logger = utils.get_logger('semi_supervised')


@dataclass
class SemisupervisedCodebookTaskOptions(mxp.TaskOptions):
    task_name: str = "semisupervised_codebook_training"
    ppl_weight: float = 1.
    feature_pen_weight: float = .0001


@dataclass
class SemiSupervisedExperiment(mxp.Experiment):
    model: bmm.ModelOptions = subgroups(
        {"brain2vec": brain2vec.Brain2VecOptions,
         'dummy': brain2vec.Brain2VecOptions},
        default=brain2vec.Brain2VecOptions()
    )

    dataset: datasets.DatasetOptions = subgroups(
        {
            "hvs": harvard_sentences.HarvardSentencesDatasetOptions,#(pre_processing_pipeline='random_sample'),
             # Not actually tested
             "nww": northwestern_words.NorthwesternWordsDatasetOptions
        },
        default=harvard_sentences.HarvardSentencesDatasetOptions(pre_processing_pipeline='random_sample'))

    task: mxp.TaskOptions = subgroups(
        {"semi_supervised": SemisupervisedCodebookTaskOptions,
         "dummy": SemisupervisedCodebookTaskOptions},
        default=SemisupervisedCodebookTaskOptions())

    def run(self):
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

        trainer = brain2vec.Brain2VecTrainer(model_map=dict(model=model), opt_map=dict(),
                                                     train_data_gen=dl_map['train'], cv_data_gen=eval_dl_map['cv'],
                                                     learning_rate=self.task.learning_rate,
                                                     early_stopping_patience=self.task.early_stopping_patience,
                                                     device=self.task.device,
                                                     ppl_weight=self.task.ppl_weight,
                                                     feature_pen_weight=self.task.feature_pen_weight,
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

        # Create the dictionary that will be json serialized as the results
        res_dict = self.create_result_dictionary(
            model_name=self.model.model_name,
            epoch_outputs=losses,
            train_selected_columns=dataset_map['train'].selected_columns,
            selected_flat_indices={k: d.selected_levels_df.to_json() for k, d in dataset_map.items()},
            best_model_epoch=trainer.best_model_epoch,
            num_trainable_params=utils.number_of_model_params(model),
            num_params=utils.number_of_model_params(model, trainable_only=False),
            model_kws=model_kws,
            **eval_res_map,
            model_options=vars(self.model),
            task_options=vars(self.task),
            dataset_options=vars(self.dataset),
            result_output_options=vars(self.result_output)
        )

        # Grab the generated uid and name to use them as file names
        uid = res_dict['uid']
        name = res_dict['name']

        self.save_results(model, result_file_name=name, result_output=self.result_output,
                          model_file_name=uid, res_dict=res_dict)

        return trainer, eval_res_map



if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(SemiSupervisedExperiment, dest='semi_supervised')
    args = parser.parse_args()
    experiment: SemiSupervisedExperiment = args.semi_supervised
    logger.info(f"EXPERIMENT: {experiment}")
    experiment.run()
