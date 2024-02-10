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
from brain2vec.models.brain2vec import Brain2VecOptions
from brain2vec.experiments import info_leakage_eval as il
from brain2vec.experiments import fine_tune as ft

from mmz.experiments import Experiment
from itertools import combinations
from mmz.experiments import ResultOptions
from time import sleep
import os
from copy import deepcopy
from pathlib import Path


logger = get_logger(__name__)

RESULT_PATH = os.environ.get('RESULT_PATH')
MODEL_PATH = os.path.join(RESULT_PATH, 'models')
Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)
Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)


@dataclass
class ILGridSearch(JsonSerializable):
    experiment: il.InfoLeakageExperiment = il.InfoLeakageExperiment(
        #task=il.ShadowClassifierMembershipInferenceFineTuningTask(),
        result_output=ResultOptions(result_dir=RESULT_PATH, save_model_path=MODEL_PATH),
    )

    n_splits: int = 1
    this_split: int = 0

    def run(self):
        kws_l = [dict(n_layers=n_layers, linear_hidden_n=linear_hidden_n,
                      dropout=dropout)
                 for n_layers in [1, 2, 3, 4]
                 for linear_hidden_n in [32, 64, 128, 512, 1024, 2048]
                 for dropout in [0., 0.5]
                 ]
        kws_splits = np.array_split(kws_l, self.n_splits)
        kws_to_run = kws_splits[self.this_split]

        print(f"Preparing to run {len(kws_to_run)} experiments")
        for i, kws in enumerate(kws_to_run):
            print(f"-->>> RUNNING:\n{kws}")
            for jj, test_set in enumerate(combos_of_3_participants):
                exper = deepcopy(self.experiment)
                exper.task.dataset.test_sets = test_set
                exper.task.dataset.pipeline_params = '{"rnd_stim__n":2000}'
                exper.task.dataset.batch_size=1024
                exper.task.dataset.batch_size_eval = 2048
                exper.task.dataset.n_dl_workers = 0
                exper.task.dataset.n_dl_eval_workers = 0
                exper.task.method = '1d_linear'
                exper.task.n_epochs = 100
                exper.task.lr_adjust_patience = 15
                exper.task.dataset.flatten_sensors_to_samples = True
                exper.task.early_stopping_patience = 20
                exper.attacker_model = ft.FineTuningModel(**kws)
                exper.attacker_model.fine_tuning_method = '1d_linear'
                exper.run()
                del exper


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(ILGridSearch, dest='il_grid_search')
    args = parser.parse_args()
    experiment: ILGridSearch = args.il_grid_search
    logger.info(f"EXPERIMENT: {experiment}")
    experiment.run()
