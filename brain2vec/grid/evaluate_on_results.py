

import pandas as pd
from mmz.utils import get_logger
import json
from dataclasses import dataclass, field
from simple_parsing.helpers import JsonSerializable
from typing import Optional, Dict, Union, List
import numpy as np
from copy import deepcopy

from brain2vec.experiments.pretrain import (
    SemiSupervisedExperiment,
    SemisupervisedCodebookTaskOptions,
)
from brain2vec.experiments.info_leakage_eval import InfoLeakageExperiment
from brain2vec.experiments.fine_tune import FineTuningExperiment
from brain2vec.datasets.harvard_sentences import (
    HarvardSentencesDatasetOptions,
    HarvardSentences,
)
from brain2vec.experiments import upack_result_options_to_columns, load_results_to_frame
from mmz.utils import SetParamsMixIn
from itertools import combinations
import os
from sklearn.model_selection import ParameterGrid
from simple_parsing import subgroups
from brain2vec.grid.base import BaseExperimentGrid
from brain2vec.grid import grid_on_results


logger = get_logger(__name__)


@dataclass
class EvaluateOnResults(BaseExperimentGrid):
    eval_process: str = 'feature_decomp'
    def feature_decomp(self):
        experiments = self.get_experiments()
        for ii, e in enumerate(experiments):
            plts = e.make_pretrain_inspection_plots()

    def run(self):
        if self.eval_process == 'feature_decomp': self.feature_decomp()

