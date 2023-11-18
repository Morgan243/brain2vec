import numpy as np
import pandas as pd
from glob import glob
import os
import json
from dataclasses import dataclass
from simple_parsing.helpers import JsonSerializable
from typing import Optional, Tuple, Union

from tqdm.auto import tqdm
import torch
from brain2vec import models


def load_results_to_frame(p, config_params=None):
    if isinstance(p, (list, tuple, np.ndarray)):
        return pd.concat([
            load_results_to_frame(_p, config_params=config_params)
            for _p in p
        ], axis=0).reset_index(drop=True)

    result_files = glob(p)

    json_result_data = [json.load(open(f)) for f in tqdm(result_files)]
    results_df = pd.DataFrame(json_result_data)
    #results_df['bw_reg_weight'] = results_df['bw_reg_weight'].fillna(-1)
    try:
        results_df['test_patient'] = results_df['test_sets'].str.split('-').apply(lambda l: '-'.join(l[:-1]))
        results_df['test_fold'] = results_df['test_sets'].str.split('-').apply(lambda l: l[-1])
    except:
        print("Unable to parse test patient - was there one?")

    ####
    if config_params is None:
        return results_df
    #elif isinstance(config_params, bool) and config_params:
    #    config_params = [n for n in experiments.all_model_hyperparam_names if n in results_df.columns.values]

    print("All config params to consider: " + ", ".join(config_params))
    #config_params = default_config_params if config_params is None else config_params
    nun_config_params = results_df[config_params].nunique()

    config_cols = nun_config_params[nun_config_params > 1].index.tolist()
    fixed_config_cols = nun_config_params[nun_config_params == 1].index.tolist()

    ###

    try:
        fixed_unique = results_df[fixed_config_cols].apply(pd.unique)
        if isinstance(fixed_unique, pd.DataFrame):
            fixed_d = fixed_unique.iloc[0].to_dict()
        else:
            fixed_d = fixed_unique.to_dict()

        fixed_d_str = "\n\t".join(f"{k}={v}" for k, v in fixed_d.items())
        #print(f"Fixed Params: {', '.join(fixed_config_cols)}")
        print(f"Fixed Params:\n------------\n\t{fixed_d_str}")
        print(f"Changing Params: {', '.join(config_cols)}\n-------------\n")
        print(results_df.groupby(config_cols).size().unstack(-1))
    except:
        print("Unable to summarize parameterization of result files... new result structure?")

    return fixed_config_cols, config_cols, results_df


def load_model_from_results(results, base_model_path=None, load_weights=True, **kws_update):
    model_kws = results['model_kws']

    if base_model_path is not None:
        _p = results['save_model_path']
        _p = _p if '\\' not in _p else _p.replace('\\', '/')

        model_filename = os.path.split(_p)[-1]
        model_path = os.path.join(base_model_path, model_filename)
        if not os.path.isfile(model_path):
            raise ValueError(f"Inferred model path does not exist: {model_path}")
    else:
        model_path = results['save_model_path']


    model_kws.update(kws_update)
    model, _ = models.make_model(model_name=results['model_name'], model_kws=model_kws)

    if load_weights:
        with open(model_path, 'rb') as f:
            model_state = torch.load(f)

        model.load_state_dict(model_state)
    return model
    #model.to(options.device)


def upack_result_options_to_columns(results_df: pd.DataFrame,
                                    options_cols_like='_options',
                                    parse_cv_and_epoch_results: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    opt_df_map = {opt_col: results_df[opt_col].apply(pd.Series) for opt_col in
                  results_df.filter(like=options_cols_like).columns}

    options_df = opt_df_map['model_options'].copy()
    for opt_name, _df in opt_df_map.items():
        keep_cols = list(set(_df.columns.tolist()) - set(options_df.columns.tolist()))
        options_df = options_df.join(_df[keep_cols].copy())

    keep_cols = list(set(options_df.columns.tolist()) - set(results_df.columns.tolist()))
    opt_cols = [c for c in options_df.columns.tolist() if c not in results_df.columns.tolist()]
    #opt_cols = options_df.drop(['model_name', 'save_model_path'], axis=1).columns.tolist()

    result_options_df = results_df.join(options_df[keep_cols])
    # When grid searching on results (i.e. pretrained models), make sure that we have a matching column
    # to how it will be parameterized in the experiment grid
    if 'model_base_path' not in result_options_df.columns:
        result_options_df['model_base_path'] = result_options_df.save_model_path.map(os.path.split).explode().reset_index(drop=True).iloc[::2].values
    if 'result_file' not in result_options_df.columns:
        result_options_df['result_file'] = result_options_df['path']

    if not parse_cv_and_epoch_results:
        return result_options_df, opt_cols

    return parse_epoch_results(result_options_df, opt_cols)


def parse_epoch_results(result_options_df, opt_cols):
    cv_results_df = None
    has_training_complete_col = 'training_complete' in result_options_df.columns
    if not has_training_complete_col or (has_training_complete_col and result_options_df.training_complete.fillna(True).any()):
        cv_results_df = pd.concat([
            pd.DataFrame(row.cv).assign(experiment_name=row['name'],
                                        uid=row.uid,
                                        model_name=row.model_name,
                                        # Options
                                        **{opt_name: row[opt_name] for opt_name in opt_cols}
                                        )
            for ix, row in result_options_df.iterrows()
            if not has_training_complete_col or row['training_complete']
        ]).reset_index(names='epoch_num')

    epoch_results_df = pd.concat([
        pd.DataFrame(row.epoch_outputs).T.assign(experiment_name=row['name'],
                                                 uid=row.uid,
                                                 model_name=row.model_name,
                                                 # Options
                                                 **{opt_name: row[opt_name] for opt_name in opt_cols}

                                                 )
        for ix, row in result_options_df.iterrows()]).reset_index(names='epoch_num')

    return result_options_df, cv_results_df, epoch_results_df, opt_cols


@dataclass
class ResultParsingOptions(JsonSerializable):
    result_file: str = None
    print_results: bool = False
    base_model_path: Optional[str] = None
    eval_sets: Optional[str] = None

    eval_win_step_size: int = 1
    pred_inspect_eval: bool = False
    base_output_path: Optional[str] = None
    eval_filter: Optional[str] = None
    device: str = 'cuda:0'
