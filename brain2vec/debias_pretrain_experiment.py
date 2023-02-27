import os

combos_of_3_participants = ['UCSD-10,UCSD-18,UCSD-19',
 'UCSD-10,UCSD-18,UCSD-22',
 'UCSD-10,UCSD-18,UCSD-28',
 'UCSD-10,UCSD-19,UCSD-22',
 'UCSD-10,UCSD-19,UCSD-28',
 'UCSD-10,UCSD-22,UCSD-28',
 'UCSD-18,UCSD-19,UCSD-22',
 'UCSD-18,UCSD-19,UCSD-28',
 'UCSD-18,UCSD-22,UCSD-28',
 'UCSD-19,UCSD-22,UCSD-28',
 'UCSD-4,UCSD-10,UCSD-18',
 'UCSD-4,UCSD-10,UCSD-19',
 'UCSD-4,UCSD-10,UCSD-22',
 'UCSD-4,UCSD-10,UCSD-28',
 'UCSD-4,UCSD-18,UCSD-19',
 'UCSD-4,UCSD-18,UCSD-22',
 'UCSD-4,UCSD-18,UCSD-28',
 'UCSD-4,UCSD-19,UCSD-22',
 'UCSD-4,UCSD-19,UCSD-28',
 'UCSD-4,UCSD-22,UCSD-28',
 'UCSD-4,UCSD-5,UCSD-10',
 'UCSD-4,UCSD-5,UCSD-18',
 'UCSD-4,UCSD-5,UCSD-19',
 'UCSD-4,UCSD-5,UCSD-22',
 'UCSD-4,UCSD-5,UCSD-28',
 'UCSD-5,UCSD-10,UCSD-18',
 'UCSD-5,UCSD-10,UCSD-19',
 'UCSD-5,UCSD-10,UCSD-22',
 'UCSD-5,UCSD-10,UCSD-28',
 'UCSD-5,UCSD-18,UCSD-19',
 'UCSD-5,UCSD-18,UCSD-22',
 'UCSD-5,UCSD-18,UCSD-28',
 'UCSD-5,UCSD-19,UCSD-22',
 'UCSD-5,UCSD-19,UCSD-28',
 'UCSD-5,UCSD-22,UCSD-28']


from mmz.utils import get_logger
from dataclasses import dataclass
from simple_parsing.helpers import JsonSerializable
from typing import Optional
import numpy as np

from brain2vec.experiments.pretrain import SemiSupervisedExperiment, SemisupervisedCodebookTaskOptions
from brain2vec.datasets.harvard_sentences import HarvardSentencesDatasetOptions, HarvardSentences
from mmz.experiments import Experiment
from itertools import combinations
from mmz.experiments import ResultOptions
from time import sleep

logger = get_logger(__name__)

RESULT_PATH = os.environ.get('RESULT_PATH')
MODEL_PATH = os.path.join(RESULT_PATH, 'models')

from pathlib import Path
#creating a new directory called pythondirectory
Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)
Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
#os.mkdir(RESULT_PATH)
#os.mkdir(MODEL_PATH)


all_patient_set_strs = [f"{k}-{v}"
                    for k, p_d in HarvardSentences.all_patient_maps.items()
                        for v in p_d.keys()]


@dataclass
class DebiasPretrainExperiments(JsonSerializable):
    experiment: SemiSupervisedExperiment = SemiSupervisedExperiment(
        result_output=ResultOptions(result_dir=RESULT_PATH, save_model_path=MODEL_PATH),
        dataset=HarvardSentencesDatasetOptions(#train_sets=None,#train_sets,
                                               batch_size=2048,
                                               batch_size_eval=2048,
                                               n_dl_workers=0,
                                               n_dl_eval_workers=0),
        task=SemisupervisedCodebookTaskOptions(lr_adjust_patience=3, early_stopping_patience=None, n_epochs=50))

    choose_n_for_pretrain: int = 2
    n_splits: int = 4
    this_split: int = 0

    device: Optional[str] = None

    @staticmethod
    def generate_pretrain_sets(choose):
        all_combos = list(combinations(all_patient_set_strs, choose))
        all_combo_cli_params = [','.join(s) for s in all_combos]
        return all_combo_cli_params

    def run(self):
        pretrain_sets = self.generate_pretrain_sets(self.choose_n_for_pretrain)
        train_set_splits = np.array_split(pretrain_sets, self.n_splits)
        set_of_sets_to_run = train_set_splits[self.this_split]

        print(f"Running {len(set_of_sets_to_run)}: {set_of_sets_to_run}")
        sleep(5)
        from copy import copy

        for train_sets in set_of_sets_to_run:
            print(f"RUNNING: {train_sets}")
            pretraining = copy(self.experiment)
            pretraining.dataset.train_sets = train_sets
#            pretraining = SemiSupervisedExperiment(
#                result_output=ResultOptions(result_dir=RESULT_PATH, save_model_path=MODEL_PATH),
#                dataset=HarvardSentencesDatasetOptions(train_sets=train_sets,
#                                                       batch_size=2048,
#                                                       batch_size_eval=2048,
#                                                       n_dl_workers=0,
#                                                       n_dl_eval_workers=0),
#                task=SemisupervisedCodebookTaskOptions(lr_adjust_patience=3, early_stopping_patience=None, n_epochs=50)
#            )

            if self.device is not None:
                pretraining.task.device = self.device
            pretraining.run()


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(DebiasPretrainExperiments, dest='debias')
    args = parser.parse_args()
    experiment: DebiasPretrainExperiments = args.debias
    logger.info(f"EXPERIMENT: {experiment}")
    experiment.run()
