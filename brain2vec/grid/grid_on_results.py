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


logger = get_logger(__name__)


@dataclass
class ExperimentGridOnResults(BaseExperimentGrid):
    """First identify relevant results, then run a grid for each result file found"""

    experiment_base_instance: Union[
        InfoLeakageExperiment, FineTuningExperiment
    ] = subgroups(
        {
            # TODO: might need to be implemented on info leakage for easier grid..?
            #"info_leakage": InfoLeakageExperiment,
            "fine_tune": FineTuningExperiment
        },
        default="fine_tune",
    )


    experiment_component_grids_d: Dict[str, Dict] = field(default=dict)
    existing_results_dir: Optional[str] = None

    n_splits: int = 1
    this_split: int = 0
    sample_tuples_for: Optional[str] = None
    sample_choose_n: Optional[int] = None

    # New params in this child class
    input_results_dir: Optional[str] = None
    input_results_query: Optional[str] = None

    def __post_init__(self):
        assert self.input_results_dir is not None, "input_results_dir must be set!"
        input_results_df = self.load_existing_results(self.input_results_dir)
        assert input_results_df is not None, f"no results found in {self.input_results_dir}"
        logger.info(f"Found {len(input_results_df)} result files in {self.input_results_dir}")
        if self.input_results_query is not None:
            assert isinstance(self.input_results_query, str),\
                f"input_results_query should be a string, but got {self.input_results_query}"
            logger.info(f"Running '{self.input_results_query}' on input results dataframe")
            self.input_results_df = input_results_df.query(self.input_results_query)
            logger.info(f"Query filters input results to {len(self.input_results_df)} result files")
        else:
            logger.info("No query provided, will grid on ALL results")
            self.input_results_df = input_results_df

        # Create the grid object as provided in CLI
        grid_d = eval(self.experiment_component_grids_str)
        input_result_file_paths: List[str] = list()
        for rix, row in self.input_results_df.iterrows():
            res_path = os.path.join(self.input_results_dir, row['name'])
            input_result_file_paths.append(res_path)
        # Update the grid object with new result_file assignments
        grid_d['pretrained_result_input.result_file'] = input_result_file_paths
        grid_d['pretrained_result_input.model_base_path'] = [os.path.join(self.input_results_dir, 'models')]

        self.experiment_component_grids_str = json.dumps(grid_d)

        super().__post_init__()

#    @classmethod
#    def filter_result_options_to_grid_spec(cls, p, experiments_l, experiments_gettr_d):
#        filtered_experiments_l, unfiltered_experiments_l = super().filter_result_options_to_grid_spec(p, experiments_l, experiments_gettr_d)


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(ExperimentGridOnResults, dest="grid")
    args = parser.parse_args()
    grid: BaseExperimentGrid = args.grid
    logger.info(f"EXPERIMENT: {grid}")
    grid.run()
