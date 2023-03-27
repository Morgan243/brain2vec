import pandas as pd
from mmz.utils import get_logger
from dataclasses import dataclass, field
from simple_parsing.helpers import JsonSerializable
from typing import Optional, Dict, Tuple, Union
import numpy as np
from copy import deepcopy

from brain2vec.experiments.pretrain import SemiSupervisedExperiment, SemisupervisedCodebookTaskOptions
from brain2vec.experiments.info_leakage_eval import InfoLeakageExperiment
from brain2vec.datasets.harvard_sentences import HarvardSentencesDatasetOptions, HarvardSentences
from brain2vec.models.brain2vec import Brain2VecOptions
from brain2vec.experiments import upack_result_options_to_columns, load_results_to_frame
from mmz.experiments import Experiment
from itertools import combinations
from mmz.experiments import ResultOptions
from time import sleep
import os
from copy import deepcopy
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from simple_parsing import subgroups


logger = get_logger(__name__)


all_patient_set_strs = [f"{k}-{v}"
                        for k, p_d in HarvardSentences.all_patient_maps.items()
                        for v in p_d.keys()]


@dataclass
class BaseExperimentGrid(JsonSerializable):
    experiment_component_grids_str: str
    experiment_base_instance: Union[InfoLeakageExperiment, SemiSupervisedExperiment] = subgroups({
        'info_leakage': InfoLeakageExperiment,
        'pretrain': SemiSupervisedExperiment
    },
    default='pretrain')

    experiment_component_grids_d: Dict[str, Dict] = field(default=dict)
    existing_results_dir: Optional[str] = None

    n_splits: int = 1
    this_split: int = 0
    sample_tuples_for: Optional[str] = None
    sample_choose_n: Optional[int] = None

    def __post_init__(self):
        if self.experiment_component_grids_str is not None:
            self.experiment_component_grids_d = eval(self.experiment_component_grids_str)

        if self.sample_tuples_for is not None:
            #self.experiment_component_grids_d['dataset'] = self.experiment_component_grids_d.get('dataset', dict())
            #self.experiment_component_grids_d['task'] = self.experiment_component_grids_d.get('task', dict())
            #self.experiment_component_grids_d['task']['dataset'] = self.experiment_component_grids_d['task'].get('dataset', dict())
            tuple_sets = self.generate_sets(self.sample_choose_n, all_patient_set_strs)

            logger.info(f"task.dataset.{self.sample_tuples_for} set to (N={len(tuple_sets)}) {tuple_sets}")
            #self.experiment_component_grids_d['task']['dataset'][self.sample_tuples_for] = tuple_sets

            # get any search for the task
            base_task_kws = self.experiment_component_grids_d.get('task', dict())
            if hasattr(self.experiment_base_instance.task, 'dataset'):
                logger.info(f"Parameterizing dataset from TASK")
                self.experiment_component_grids_d['task'] = [
                    dict(dataset=[{self.sample_tuples_for: t}])
                    for t in tuple_sets
                ]
            elif hasattr(self.experiment_base_instance, 'dataset'):
                logger.info(f"Parameterizing dataset from EXPERIMENT")
                self.experiment_component_grids_d['dataset'] = {self.sample_tuples_for: tuple_sets}


        self.experiment_grid_l, self.experiment_grid_gettr_map = self.experiment_grid_from_component_grids(
            self.experiment_base_instance,
            self.experiment_component_grids_d
        )

    @staticmethod
    def generate_sets(choose, input_sets):
        all_combos = list(combinations(input_sets, choose))
        all_combo_cli_params = [','.join(s) for s in all_combos]
        return all_combo_cli_params

    @staticmethod
    def load_existing_results(p):
        model_glob_p = os.path.join(p, '*.json')
        results_df = load_results_to_frame(model_glob_p)
        if len(results_df) == 0:
            return None
        return upack_result_options_to_columns(results_df)

    @classmethod
    def filter_result_options_to_grid_spec(cls, p, experiments_l, experiments_gettr_d):
        results_options_df = cls.load_existing_results(p)
        if results_options_df is None:
            return list(), experiments_l

        matching_experiments = list()
        unfiltered_experiments_l = list()
        filtered_experiments_l = list()
        for exper in experiments_l:
            m = pd.Series(True, index=results_options_df.index)
            for attr_dot_k, gettr_f in experiments_gettr_d.items():
                col = attr_dot_k.split('.')[-1]
                expers_val = gettr_f(exper)
                _m = results_options_df[col].eq(
                    #gettr_f(exper)
                    expers_val
                )
                logger.info(f"{_m.sum()} have {attr_dot_k} equal to {expers_val}")
                m = m & _m
            if m.any():
                filtered_experiments_l.append(exper)
            else:
                unfiltered_experiments_l.append(exper)

        return filtered_experiments_l, unfiltered_experiments_l

    @staticmethod
    def experiment_grid_from_component_grids(experiment_base_instance, experiment_component_grid_specs_d):
        exper_k_grids_d = {exper_k: list(ParameterGrid(grid_spec_d))
                           for exper_k, grid_spec_d in experiment_component_grid_specs_d.items()}
        exper_obj_l = list()
        grid_name_value_gettr_d = dict()
        for _g in ParameterGrid(exper_k_grids_d):
            from copy import deepcopy
            exp = deepcopy(experiment_base_instance)
            print('-'*20)
            print(f'Grid: {_g}')
            for exper_attr, exper_attr_params in _g.items():

                for sub_attr_k, sub_attr_val in exper_attr_params.items():
                    def _getattr(_exp, _exper_attr=exper_attr, _sub_attr_k=sub_attr_k):
                        _exper_attr_o = getattr(_exp, _exper_attr)
                        return getattr(_exper_attr_o, _sub_attr_k)

                    grid_name_value_gettr_d[f"{exper_attr}.{sub_attr_k}"] = _getattr

                exper_attr_o = getattr(exp, exper_attr)
                print(f"Setting: {exper_attr_params}")
                exper_attr_o.set_params(exper_attr_params)
                setattr(exp, exper_attr, exper_attr_o)

            exper_obj_l.append(exp)

        return exper_obj_l, grid_name_value_gettr_d

    @staticmethod
    def _split_and_get_ith_item(l, n_splits: int, this_split: int):
        splits = np.array_split(l, n_splits)
        ith_split = splits[this_split]
        return ith_split

    def get_experiments(self):
        set_of_sets_to_run = self._split_and_get_ith_item(self.experiment_grid_l, self.n_splits,
                                                          self.this_split)
        logger.info(f"Total of {len(set_of_sets_to_run)} experiments split {self.this_split + 1}/{self.n_splits} "
                    f"in grid of {len(self.experiment_grid_l)}")

        if self.existing_results_dir is not None:
            already_exists, not_seen = self.filter_result_options_to_grid_spec(self.existing_results_dir,
                                                                               set_of_sets_to_run,
                                                                               self.experiment_grid_gettr_map)
            set_of_sets_to_run = not_seen
            logger.info(f"{len(set_of_sets_to_run)} experiments remain after filtering {len(already_exists)} "
                        f"completed in {self.existing_results_dir}")

        return set_of_sets_to_run

    def run(self):
        experiments = list(self.get_experiments())
        while len(experiments) > 0:
            exper = experiments.pop(0)
            exper = deepcopy(exper)
            exper.run()
            del exper


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(BaseExperimentGrid, dest='grid')
    args = parser.parse_args()
    grid: BaseExperimentGrid = args.grid
    logger.info(f"EXPERIMENT: {grid}")
    grid.run()
