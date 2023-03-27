from mmz.utils import get_logger
from dataclasses import dataclass
from simple_parsing.helpers import JsonSerializable
from typing import Optional
import numpy as np

from brain2vec.experiments.pretrain import SemiSupervisedExperiment, SemisupervisedCodebookTaskOptions
from brain2vec.datasets.harvard_sentences import HarvardSentencesDatasetOptions, HarvardSentences
from brain2vec.models.brain2vec import Brain2VecOptions
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
class PretrainGridSearch(JsonSerializable):
    experiment: SemiSupervisedExperiment = SemiSupervisedExperiment(
        result_output=ResultOptions(result_dir=RESULT_PATH, save_model_path=MODEL_PATH),
        dataset=HarvardSentencesDatasetOptions(train_sets='*',  # train_sets,
                                                 batch_size=2048,
                                                 batch_size_eval=4096,
                                                 n_dl_workers=0,
                                                 n_dl_eval_workers=0),
        task=SemisupervisedCodebookTaskOptions(lr_adjust_patience=5, early_stopping_patience=None, n_epochs=20))

    n_splits: int = 1
    this_split: int = 0

    def run(self):
        pos_opts = [
            # This config is not valid - temporal aspect combined into the ras encoding
            #dict(positional_encoding_method='combined', temporal_pos_encoding=True, ras_pos_encoding=True),
            # --

            dict(positional_encoding_method='combined', #temporal_pos_encoding=False,
                 ras_pos_encoding=True),
            #dict(positional_encoding_method='independent', temporal_pos_encoding=True, ras_pos_encoding=True),
            #dict(positional_encoding_method='independent', temporal_pos_encoding=False, ras_pos_encoding=True),

            #dict(positional_encoding_method='independent', temporal_pos_encoding=False, ras_pos_encoding=False),
            #dict(positional_encoding_method='independent', temporal_pos_encoding=False, ras_pos_encoding=True),
        ]
        kws_l = [dict(n_encoder_layers=n_enc_layers,
                      quant_num_vars=quant_num_vars,
                      **pos_opts_kws)
                 for pos_opts_kws in pos_opts
                 for n_enc_layers in [3, 4, 5]
                 for quant_num_vars in [40, 80]
                 ]

        kws_splits = np.array_split(kws_l, self.n_splits)
        kws_to_run = kws_splits[self.this_split]

        print(f"Preparing to run {len(kws_to_run)} experiments")
        for i, kws in enumerate(kws_to_run):
            print(f"-->>> RUNNING:\n{kws}")
            exper = deepcopy(self.experiment)
            exper.model = Brain2VecOptions(**kws)
            exper.run()


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(PretrainGridSearch, dest='pretrain_grid_search')
    args = parser.parse_args()
    experiment: PretrainGridSearch = args.pretrain_grid_search
    logger.info(f"EXPERIMENT: {experiment}")
    experiment.run()
