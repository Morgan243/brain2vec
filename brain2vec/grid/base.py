import pandas as pd
from mmz.utils import get_logger
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


logger = get_logger(__name__)


all_patient_set_strs = [
    f"{k}-{v}"
    for k, p_d in HarvardSentences.all_patient_maps.items()
    for v in p_d.keys()
]


@dataclass
class BaseExperimentGrid(JsonSerializable):
    experiment_component_grids_str: str
    experiment_base_instance: Union[
        InfoLeakageExperiment, SemiSupervisedExperiment
    ] = subgroups(
        {"info_leakage": InfoLeakageExperiment,
         "pretrain": SemiSupervisedExperiment,
         },
        default="pretrain",
    )

    experiment_component_grids_d: Dict[str, Dict] = field(default=dict)
    existing_results_dir: Optional[str] = None

    n_splits: int = 1
    this_split: int = 0
    sample_tuples_for: Optional[str] = None
    sample_choose_n: Optional[int] = None
    sample_to_skip: Optional[str] = None
    n_to_run: Optional[int] = None

    def __post_init__(self):
        if self.experiment_component_grids_str is None:
            self.experiment_component_grids_str = "{}"

        self.experiment_component_grids_d = eval(self.experiment_component_grids_str)

        if self.sample_tuples_for is not None:
            if self.sample_to_skip is not None:
                if isinstance(self.sample_to_skip, str):
                    self.sample_to_skip = [w.strip() for w in self.sample_to_skip.split(',')]
            tuple_sets = self.generate_sets(self.sample_choose_n, all_patient_set_strs,
                                            cli_params_to_skip=self.sample_to_skip)

            logger.info(
                f"task.dataset.{self.sample_tuples_for} set to (N={len(tuple_sets)}) {tuple_sets}"
            )

            # get any search for the task
            if hasattr(self.experiment_base_instance.task, "dataset"):
                logger.info("Parameterizing dataset from TASK")
                # Get out the list of dataset grid dictionaries
                self.experiment_component_grids_d[
                    f"task.dataset.{self.sample_tuples_for}"
                ] = tuple_sets

            elif hasattr(self.experiment_base_instance, "dataset"):
                logger.info("Parameterizing dataset from EXPERIMENT")
                self.experiment_component_grids_d[f"dataset.{self.sample_tuples_for}"] = tuple_sets

        self.experiment_grid_l = list()
        self.experiment_grid_gettr_map = dict()
        for trial in ParameterGrid(self.experiment_component_grids_d):
            exp = deepcopy(self.experiment_base_instance)
            for attr_k, attr_v in trial.items():
                try:
                    exp = exp.set_recursive_dot_attribute(attr_k, attr_v)
                except AssertionError as e:
                    print(f"FAILED trying to set {attr_k} to {attr_v} on {exp}")
                    raise

                def _gettr(_exp, _attr_k=attr_k):
                    return _exp.get_recursive_dot_attribute(_attr_k)

                self.experiment_grid_gettr_map[attr_k] = _gettr

            self.experiment_grid_l.append(exp)

    @staticmethod
    def generate_sets(choose, input_sets, cli_params_to_skip: Optional[List[str]] = None):
        all_combos = list(combinations(input_sets, choose))
        all_combo_cli_params = [",".join(s) for s in all_combos]
        if cli_params_to_skip is not None:
            all_combo_cli_params = [s for s in all_combo_cli_params if s not in cli_params_to_skip]
        return all_combo_cli_params

    @staticmethod
    def load_existing_results(p):
        model_glob_p = os.path.join(p, "*.json")
        results_df = load_results_to_frame(model_glob_p)
        if len(results_df) == 0:
            return None
        return upack_result_options_to_columns(results_df)

    @classmethod
    def filter_result_options_to_grid_spec(cls, p, experiments_l, experiments_gettr_d):
        results_options_df = cls.load_existing_results(p)
        if results_options_df is None:
            return list(), experiments_l

        results_options_df, opt_cols = results_options_df
        unfiltered_experiments_l = list()
        filtered_experiments_l = list()
        for exper in experiments_l:
            m = pd.Series(True, index=results_options_df.index)
            for attr_dot_k, gettr_f in experiments_gettr_d.items():
                col = attr_dot_k.split(".")[-1]
                expers_val = gettr_f(exper)

                if col == 'result_file':
                    pretrained_results_df = results_options_df.pretrained_results.apply(pd.Series)
                    pretrained_results_df['result_filename'] = pretrained_results_df.path.map(lambda s: os.path.split(s)[-1])
                    this_result_filename = os.path.split(expers_val)[-1]
                    logger.info("Checking result files for: " + this_result_filename)
                    _m = pretrained_results_df['result_filename'].eq(this_result_filename)
                    logger.info("Result filenames: " + (", ".join(sorted(pretrained_results_df.result_filename.unique()))))
                    logger.info(f"Result matching: {_m.sum()}")
                    #_m = pd.Series(True, index=results_options_df.index)
                    #logger.info(f"Result File path is passthrough")
                elif col == 'model_base_path':
                    _m = pd.Series(True, index=results_options_df.index)
                    logger.info(f"Model base path is passthrough")
                else:
                    try:
                        _m = results_options_df[col].eq(
                            # gettr_f(exper)
                            expers_val
                        )
                    except KeyError as e:
                        print(
                            f"{col} extracted from {attr_dot_k} no in results_options_df cols: {results_options_df.columns.tolist()}"
                        )
                        raise e
                    logger.info(f"{_m.sum()} have {attr_dot_k} equal to {expers_val}")
                m = m & _m
            if m.any():
                filtered_experiments_l.append(exper)
            else:
                unfiltered_experiments_l.append(exper)

        return filtered_experiments_l, unfiltered_experiments_l

    @classmethod
    def recurse_into_attributes_from_dict(cls, o: object, d: Dict, parents=None):
        """
        O is object with attribute assignments from dictionary d.

        for k through keys in d
            if all(
            if k is an attribute of o, recurse

        :param o:
        :param d:
        :param parents:
        :return:
        """
        parents = [] if parents is None else parents
        assert issubclass(o.__class__, SetParamsMixIn)

        this_o_params = dict()
        this_o_getters = dict()
        for k, v in d.items():
            assert hasattr(o, k)
            _next = getattr(o, k)

            if issubclass(_next.__class__, SetParamsMixIn):
                assert isinstance(v, dict)
                # _next.set_params()
                return BaseExperimentGrid.recurse_into_attributes_from_dict(
                    _next, v, parents=parents + [k]
                )
            else:
                this_o_params[k] = v

                def _getattr(_parent, _k=k):
                    _p = _parent
                    for _p_k in parents:
                        _p = getattr(_p, _p_k)
                    return getattr(_p, _k)

                getter_k = ".".join(parents + [k])
                this_o_getters[getter_k] = _getattr

        o.set_params(this_o_params)
        return o, this_o_getters

    @staticmethod
    def _split_and_get_ith_item(list_or_arr, n_splits: int, this_split: int):
        splits = np.array_split(list_or_arr, n_splits)
        ith_split = splits[this_split]
        return ith_split

    def get_experiments(self):
        set_of_sets_to_run = self._split_and_get_ith_item(
            self.experiment_grid_l, self.n_splits, self.this_split
        )
        logger.info(
            f"Total of {len(set_of_sets_to_run)} "
            f"experiments split {self.this_split + 1}/{self.n_splits} "
            f"in grid of {len(self.experiment_grid_l)}"
        )

        if self.existing_results_dir is not None:
            already_exists, not_seen = self.filter_result_options_to_grid_spec(
                self.existing_results_dir,
                set_of_sets_to_run,
                self.experiment_grid_gettr_map,
            )
            set_of_sets_to_run = not_seen
            logger.info(
                f"{len(set_of_sets_to_run)} experiments remain after "
                f"filtering {len(already_exists)} "
                f"completed in {self.existing_results_dir}"
            )
            for s in set_of_sets_to_run:
                logger.info(s)
            to_run_str = '\n'.join(map(str, set_of_sets_to_run))
            #logger.info(f"Sets to run: {to_run_str}")
        else:
            logger.info(
                "existing_results_dir not provided, not filtering based on prior results"
            )

        return set_of_sets_to_run

    def run(self):
        experiments = list(self.get_experiments())
        n_completed = 0
        while len(experiments) > 0:
            exper = experiments.pop(0)
            exper = deepcopy(exper)
            exper.run()
            del exper
            from joblib.externals.loky import get_reusable_executor
            print("Grid - Shutting down loky")
            get_reusable_executor().shutdown(wait=True)
            print("Grid - ..done")
            n_completed += 1
            if self.n_to_run is not None and n_completed >= self.n_to_run:
                print(f"Ending early - completed {n_completed}, which >= {self.n_to_run}")
                break


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(BaseExperimentGrid, dest="grid")
    args = parser.parse_args()
    grid: BaseExperimentGrid = args.grid
    logger.info(f"EXPERIMENT: {grid}")
    grid.run()
