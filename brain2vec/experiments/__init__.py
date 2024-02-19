import numpy as np
import pandas as pd
from pandas.api.types import is_hashable
from glob import glob
import os
import json
from dataclasses import dataclass
from simple_parsing.helpers import JsonSerializable
from typing import Optional, Tuple, Union, List, Dict, ClassVar
from dataclasses import field, make_dataclass


from tqdm.auto import tqdm
import torch
from brain2vec import models
from brain2vec.datasets.harvard_sentences import HarvardSentences
from brain2vec import utils


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

    all_tuples: ClassVar[List] = HarvardSentences.make_tuples_from_sets_str('*')
    all_set_strs: ClassVar[set] = {f"{t[0]}-{t[1]}" for t in all_tuples}
    pt_str_to_pub_id: ClassVar[Dict[str, int]] = {f"{t[0]}-{t[1]}": ii + 1 for ii, t in enumerate(sorted(all_tuples))}

    ParsedResults: ClassVar = make_dataclass("ParsedResults",
                                             ['result_options_df',
                                              #'pretrain_exper_df',
                                              'pretrain_parser',
                                              #'pretrain_results_df',
                                              'opt_cols',
                                              'mean_train_acc_ctab',
                                              'mean_cv_acc_ctab',
                                              'mean_test_acc_ctab'])

@dataclass
class PretrainResultParsingOptions(ResultParsingOptions):
    result_file: Optional[Union[str, List[str]]] = None
    results_df: Optional[pd.DataFrame] = None
    config_labels: Optional[dict] = None
    print_results: Optional[bool] = False

    #indicator_eval_d: Dict[str, str] = field(default_factory=lambda : {
    #    'Large SSL': "quant_num_vars == 160 and n_encoder_layers== 8",
    #    'Medium SSL': "quant_num_vars == 20 and n_encoder_layers== 8",
    #    'Small SSL': "quant_num_vars == 10 and n_encoder_layers== 4",
    #})

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

    renames = {
        'model_output_shape_str': 'Output Dimensions\n(Embed. x Time)',
        'feature_grad_mult': 'Feature Gradient Multiplier',
        'quant_num_vars': 'Codebook Size',
        'mask_length': 'Mask Length',
    }

    pos_order = ['Spatio-temporal from RAS', 'Spatial from RAS']
    output_shape_order = ['16x3', '32x6']
    mask_length_order = [1, 2]
    quant_vars_order = [20, 40, 80]

    feature_extractor_col: str = 'Feature Extractor'
    featenc_labels_d: ClassVar[Dict] = {
        # Original (must have)
        '5x32ch(original)': "feature_extractor_layers == '[(128, 7, 2)] + [(64, 3, 2)] * 2 + [(32, 3, 2)] * 2'",
        # Reducing channels
        '5x16ch(original)': "feature_extractor_layers == '[(128, 7, 2)] + [(64, 3, 2)] * 2 + [(32, 3, 2), (16, 3, 2)]'",
        # Reduce depth and widen first
        'wide-4x32ch(original)': "feature_extractor_layers == '[(128, 7, 7)] + [(64, 3, 2)] * 2 + [(32, 3, 2)]'",
        # Reduce depth and widen first two layers
        'wide2-4x32ch': "feature_extractor_layers == '[(128, 7, 7), (64, 5, 5), (64, 3, 1), (32, 3, 1)]'",
        # Reduce depth and widen first two layers and reduce channels
        'wide2-4x16ch': "feature_extractor_layers == '[(128, 7, 7), (64, 5, 5), (64, 3, 1), (16, 3, 1)]'",

        'wide2-3x16ch': "feature_extractor_layers == '[(128, 7, 7), (64, 5, 5), (16, 3, 2)]'",

        # -- Development --
        '4x32ch':  "feature_extractor_layers == '[(128, 7, 7)]  + [(128, 5, 5), (32,3,1), (32, 3, 1)]'",
        '4x16ch': "feature_extractor_layers == '[(128, 7, 7)]  + [(128, 5, 5), (32,3,1), (16, 3, 1)]'",
        '3x16ch': "feature_extractor_layers == '[(128, 7, 7)]  + [(128, 5, 5), (16, 3, 2)]'",
    }

    positional_encoding_col: str = 'Positional Encoding'
    posenc_labels_d: ClassVar[Dict] = {
        'Spatio-temporal from RAS':
            "positional_encoding_method == 'combined' and ras_pos_encoding and (ras_architecture == 'simple')",
        'CombinedRAS-Multihead':
            "positional_encoding_method == 'combined' and ras_pos_encoding and (ras_architecture == 'multihead')",
        'PosRAS-Multihead':
            "positional_encoding_method == 'ras_pos' and ras_pos_encoding and (ras_architecture == 'multihead')",
        'Positional-RAS':
            "positional_encoding_method == 'ras_pos' and ras_pos_encoding and (ras_architecture == 'simple')",

        'PosEmbed-Multihead':
            "positional_encoding_method == 'pos_embedding' and ras_pos_encoding and (ras_architecture == 'multihead')",
        'Spatial from RAS':
            "positional_encoding_method == 'pos_embedding' and ras_pos_encoding and (ras_architecture == 'simple')",

    }

    approach_labels_d: ClassVar[Dict] = {
        #'Original': "`Positional Encoding`.eq('CombinedRAS-Simple') & `Feature Extractor`.eq('5x32ch(original)')",
        #'Small Embedding': "`Positional Encoding`.eq('CombinedRAS-Multihead') & `Feature Extractor`.eq('5x16ch(original)')",
        #'Small Embedding with Wide Field': "`Positional Encoding`.eq('PosRAS-Multihead') & `Feature Extractor`.eq('wide2-4x16ch(original)')",

        #'Small Embedding v2': "`Positional Encoding`.eq('CombinedRAS-Multihead') & `Feature Extractor`.eq('5x16ch(original)')",
        #'Small Embedding with Wide Field v2': "`Positional Encoding`.eq('PosRAS-Multihead') & `Feature Extractor`.eq('wide2-3x16ch(original)')",
        #'': "`Positional Encoding`.eq('CombinedRAS-Multihead') & `Feature Extractor`.eq('wide2-3x16ch(original)')",
    }

    base_approach_eval_str: ClassVar[Optional[str]] = None


    def __post_init__(self):
        if self.results_df is None:
            self.results_df = load_results_to_frame(
                self.result_file
            )

        self.opt_df_map = {opt_col: self.results_df[opt_col].apply(pd.Series)
                      for opt_col in self.results_df.filter(like='_options').columns}

        # --
        options_df = self.opt_df_map['model_options'].copy()
        for opt_name, _df in self.opt_df_map.items():
            if opt_name == 'model_options':
                continue
            keep_cols = list(set(_df.columns.tolist()) - set(options_df.columns.tolist()))
            options_df = options_df.join(_df[keep_cols].copy())

        self.options_df = options_df
        self.opt_cols = options_df.drop(['model_name', 'save_model_path'], axis=1).columns.tolist()

        self.result_options_df = self.results_df.join(options_df[self.opt_cols])

        self.result_options_df['pretrain_name'] = self.result_options_df['name']

        self.result_options_df['pretrain_sets'] = self.result_options_df['train_sets']
        self.result_options_df['pretrain_n_sets'] = self.result_options_df.pretrain_sets.str.split(',').map(len)
        self.result_options_df['Fine Tuning Pt.'] = self.result_options_df.train_sets.map(self.pt_str_to_pub_id)
        test_sets_s = self.result_options_df.pretrain_sets.str.split(',').map(set).apply(lambda s: self.all_set_strs - s)
        test_sizes = test_sets_s.map(len)
        self.result_options_df['pretrain_n_holdout_sets'] = test_sizes
        for test_s in test_sizes.unique():
            test_s_m = self.result_options_df['pretrain_n_holdout_sets'].eq(test_s)
            m_test_sets_s = test_sets_s[test_s_m]
            m_train_sets_s = self.result_options_df['pretrain_sets'][test_s_m]
            # 6 pair has holdout of 1
            if test_s == 1:
                #assert test_sets_s.map(len).max() == 1
                self.result_options_df.loc[
                    test_s_m,
                    'pretraining_holdout_pub_id'] = m_test_sets_s.explode().map(self.pt_str_to_pub_id)
                #ctab_cols.append('pretraining_holdout_pub_id')

                self.result_options_df.loc[
                    test_s_m,
                    'pretraining_train_pub_id'] = self.result_options_df.loc[test_s_m].pretrain_sets.map(
                    lambda s: ", ".join(str(pub_id) for pt_str, pub_id in self.pt_str_to_pub_id.items() if pt_str in s)
                )

                self.result_options_df['Pretraining Pt.'] = self.result_options_df['pretraining_train_pub_id']

                #self.result_options_df.loc[
                #    test_s_m,
                #    'pt_transfer_type'] = self.result_options_df.loc[test_s_m].apply(
                #    lambda r: r.train_sets in r.pretrain_sets,
                #    axis=1).replace({True: "Same Pt.", False: "Different Pt."})

            # 1 pair has holdout of 6
            elif test_s > 1:
                self.result_options_df.loc[
                    test_s_m,
                    #'pretraining_holdout_pub_id'] = m_test_sets_s.explode().map(self.pt_str_to_pub_id)
                    'pretraining_holdout_pub_id'] = m_test_sets_s.map(lambda l: ", ".join(str(self.pt_str_to_pub_id[_l]) for _l in l))#.explode().map(cls.pt_str_to_pub_id)

                self.result_options_df.loc[
                    test_s_m,
                    'pretraining_train_pub_id'] = self.result_options_df.loc[test_s_m].pretrain_sets.map(
                    lambda s: ", ".join(str(pub_id) for pt_str, pub_id in self.pt_str_to_pub_id.items() if pt_str in s)
                )
                self.result_options_df['Pretraining Pt.'] = self.result_options_df['pretraining_train_pub_id']

                #ctab_cols.append('Pretraining Pt.')

                #self.result_options_df.loc[test_s_m, 'pt_transfer_type'] = self.result_options_df.loc[test_s_m].apply(
                #    lambda r: r.train_sets in r.pretrain_sets,
                #    axis=1).replace({True: "Same Pt.", False: "Different Pt."})

                #result_options_df['pt_transfer_type'] = result_options_df.eval('train_sets in pretrain_sets').replace(
                #    {True: "Same Pt.", False: "Different Pt."})
            else:
                raise ValueError(f"test_s  is {test_s}")

        #self.result_options_df['Transfer Type'] = self.result_options_df.pt_transfer_type.eq("Same Pt.").replace(
        #    {True: "Pretrain Pt.", False: "Non-Pretrain Pt."}
        #)

        #mean_test_acc_ctab = result_options_df.groupby(ctab_cols).test_accuracy.mean().unstack().round(2).T
        #mean_cv_acc_ctab = result_options_df.groupby(ctab_cols).cv_accuracy.mean().unstack().round(2).T
        #mean_train_acc_ctab = result_options_df.groupby(ctab_cols).train_accuracy.mean().unstack().round(2).T


        #pretrain_set_size_df = cls.map_sensor_count_columns_to_set_str(result_options_df.pretrain_sets)
        #result_options_df = result_options_df.join(pretrain_set_size_df)
        #print(result_options_df['task_name_pub_fmt_short'].value_counts())

        self.result_options_df['parent_dir'] = self.result_options_df.path.apply(lambda s: os.path.split(s)[0])
        self.result_options_df['feature_extractor_layers_str'] = (self.result_options_df
                                                                  .feature_extractor_layers
                                                                  .apply(eval)
                                                                  .apply(lambda l: "|".join(".".join(str(_t) for _t in t) for t in l))
                                                                  )

        self.opt_cols += ['pretrain_n_sets',
                          'pretrain_n_holdout_sets',
                          'pretraining_holdout_pub_id',
                          'pretraining_train_pub_id',
                          #'Transfer Type',
                          'Pretraining Pt.',
                          'feature_extractor_layers_str']

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
            self.result_options_df['model_output_shape_str'] = self.result_options_df.model_output_shape \
                .apply(lambda s: tuple(s) if isinstance(s, list) else tuple()) \
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

        self.result_options_df, self.feature_cols = self.create_labels_for_feature_encoder(self.result_options_df)
        self.result_options_df, self.posenc_cols = self.create_labels_for_posenc(self.result_options_df)


        unique_opt_counts_s = self.result_options_df[self.opt_cols].apply(
            lambda c: c.nunique() if c.map(is_hashable).all() else np.nan, axis=0)

        self.varied_opts_s: pd.Series = unique_opt_counts_s[unique_opt_counts_s > 1]
        self.hashable_opts_l: list = unique_opt_counts_s.index.tolist()


        # Setup publication descriptive text
        approch_s_map = dict(
            full_approach=(self.result_options_df['Positional Encoding']
                           + '\nMask Length = ' + self.result_options_df.mask_length.astype(str)
                           + '\nCodebook Size = ' + self.result_options_df.quant_num_vars.astype(str)
                           + '\nNum. Encoder Layers = ' + self.result_options_df.n_encoder_layers.astype(str)
                           + '\nOutput Dim. = ' + self.result_options_df.model_output_shape_str.astype(str)
                           ),

            no_depth_approach=(self.result_options_df['Positional Encoding']
                               + '\nMask Length = ' + self.result_options_df.mask_length.astype(str)
                               + '\nCodebook Size = ' + self.result_options_df.quant_num_vars.astype(str)
                               # + '\nNum. Encoder Layers = ' + result_options_df.n_encoder_layers.astype(str)
                               + '\nOutput Dim. = ' + self.result_options_df.model_output_shape_str.astype(str)
                               ),

            no_mask_approach=(self.result_options_df['Positional Encoding']
                              # + '\nMask Length = ' + result_options_df.mask_length.astype(str)
                              + '\nCodebook Size = ' + self.result_options_df.quant_num_vars.astype(str)
                              + '\nOutput Dim. = ' + self.result_options_df.model_output_shape_str.astype(str)
                              ),

            codebook_and_pos_approach=(self.result_options_df['Positional Encoding']
                                       # + '\nMask Length = ' + result_options_df.mask_length.astype(str)
                                       + '\nCodebook Size = ' + self.result_options_df.quant_num_vars.astype(str)
                                       # + '\nOutput Dim. = ' + result_options_df.model_output_shape_str.astype(str)
                                       ),

            codebook_pos_depth_approach=(self.result_options_df['Positional Encoding']
                                         # + '\nMask Length = ' + result_options_df.mask_length.astype(str)
                                         + '\nCodebook Size = ' + self.result_options_df.quant_num_vars.astype(str)
                                         + '\nNum. Encoder Layers = ' + self.result_options_df.n_encoder_layers.astype(str)
                                         ),

            depth_no_mask_approach=(self.result_options_df['Positional Encoding']
                                    # + '\nMask Length = ' + result_options_df.mask_length.astype(str)
                                    + '\nCodebook Size = ' + self.result_options_df.quant_num_vars.astype(str)
                                    + '\nNum. Encoder Layers = ' + self.result_options_df.n_encoder_layers.astype(str)
                                    + '\nOutput Dim. = ' + self.result_options_df.model_output_shape_str.astype(str)
                                    ),
        )

        for n, s in approch_s_map.items():
            self.result_options_df[n] = s

        self.opt_cols = self.opt_cols + self.feature_cols + self.posenc_cols + list(approch_s_map.keys())

        self.renames_full_approach = {'full_approach': 'Approach'}
        self.renames_nomask_approach = {'no_mask_approach': 'Approach'}
        self.renames_nodepth_approach = {'no_depth_approach': 'Approach'}
        self.renames_depth_nomask_approach = {'depth_no_mask_approach': 'Approach'}
        self.renames_codebook_and_pos_approach = {'codebook_and_pos_approach': 'Approach'}
        self.renames_codebook_pos_depth_approach = {'codebook_pos_depth_approach': 'Approach'}

        self.epoch_results_df, self.cv_epoch_results_df = self.extract_epoch_results(self.result_options_df,
                                                                                     self.opt_cols)

        self.melt_epoch_df = self.epoch_results_df.melt(
            ['epoch_num', 'experiment_name'] + self.opt_cols,
            [  # 'bce_loss', 'cv_bce_loss',
                'cv_bce_loss', 'cv_perplexity',
                # 'codewords_used',
                'cv_feature_pen',
                # 'cv_loss', 'total_loss',
            ])

    @classmethod
    def from_paths(cls, paths: List[str], **kws):
        return cls(results_df=pd.concat(
            (load_results_to_frame(p).assign(results_loaded_from_path=p) for p in paths),
            ignore_index=True
        ), **kws)


    @classmethod
    def extract_epoch_results(cls, result_options_df, option_cols):
        cv_epoch_results_df = None

        if 'training_complete' not in result_options_df.columns or result_options_df.training_complete.any():
            cv_epoch_results_df = pd.concat([
                pd.DataFrame(row.cv).assign(experiment_name=row['name'],
                                            uid=row.uid,
                                            model_name=row.model_name,
                                            # Options
                                            **{opt_name: row[opt_name] for opt_name in option_cols if opt_name in row}
                                            )
                for ix, row in result_options_df.iterrows()
                if 'training_complete' not in row or row['training_complete'] and not pd.isnull(row['cv'])
            ]).reset_index(names='epoch_num')

        epoch_results_df = pd.concat([
            pd.DataFrame(row.epoch_outputs).T.assign(experiment_name=row['name'],
                                                     uid=row.uid,
                                                     model_name=row.model_name,
                                                     # Options
                                                     **{opt_name: row[opt_name] for opt_name in option_cols if opt_name in row}

                                                     )
            for ix, row in result_options_df.iterrows()]).reset_index(names='epoch_num')

        cv_epoch_res_df = epoch_results_df.cv_loss_d.apply(pd.Series)
        cv_epoch_res_df.columns = "cv_" + cv_epoch_res_df.columns

        epoch_results_df = cv_epoch_res_df.join(epoch_results_df)

        cv_epoch_res_df['cv_loss'] = cv_epoch_res_df[['cv_bce_loss', 'cv_perplexity', 'cv_feature_pen']].sum(axis=1)
        epoch_results_df['loss_delta'] = epoch_results_df.eval("total_loss - cv_loss")
        epoch_results_df['codewords_used'] = (1 - epoch_results_df.cv_perplexity) * (
                epoch_results_df.quant_num_vars ** epoch_results_df.quant_num_groups)

        return epoch_results_df, cv_epoch_results_df

    @classmethod
    @utils.return_new_columns
    def create_labels_for_feature_encoder(cls, result_options_df: pd.DataFrame):

        featenc_lables_d = {k: result_options_df.eval(s).rename(k)
                            for k, s in cls.featenc_labels_d.items()}
        featenc_labels_df = pd.concat(featenc_lables_d.values(), axis=1)
        print(featenc_labels_df.value_counts().to_frame())
        result_options_df = utils.check_and_join_labels(result_options_df, featenc_labels_df,
                                                      from_dummies_name=cls.feature_extractor_col)
        return result_options_df


    @classmethod
    @utils.return_new_columns
    def create_labels_for_posenc(cls, result_options_df: pd.DataFrame):
        posenc_lables_d = {k: result_options_df.eval(s).rename(k) for k, s in cls.posenc_labels_d.items()}
        posenc_labels_df = pd.concat(posenc_lables_d.values(), axis=1)
        print(posenc_labels_df.value_counts().to_frame())

        result_options_df = utils.check_and_join_labels(result_options_df, posenc_labels_df,
                                                        from_dummies_name=cls.positional_encoding_col)
        return result_options_df

    @classmethod
    @utils.return_new_columns
    def create_labels_for_approach(cls, result_options_df: pd.DataFrame,
                                   base_approach_eval_str: str = None):

        base_approach_eval_str = cls.base_approach_eval_str if base_approach_eval_str is None else base_approach_eval_str

        if isinstance(base_approach_eval_str, str) and len(base_approach_eval_str.strip()) > 0:
            base_approach_m = result_options_df.eval(base_approach_eval_str).rename('base_approach')
        else:
            base_approach_m = pd.Series(True, index=result_options_df.index, name='base_approach')

        approach_labels_d = {k: (base_approach_m & result_options_df.eval(s)).rename(k)
                             for k, s in cls.approach_labels_d.items()}

        if len(approach_labels_d) == 0:
            return result_options_df.assign(Approach=np.nan)

        approach_labels_df = pd.concat(approach_labels_d.values(), axis=1)
        print(approach_labels_df.value_counts().to_frame())
        result_options_df = utils.check_and_join_labels(result_options_df, approach_labels_df,
                                                      from_dummies_name='Approach')
        return result_options_df



# Override to make the result parsing options optional in this script
@dataclass
class FineTuneResultParsingOptions(ResultParsingOptions):
    result_file: Optional[Union[str, List[str]]] = None
    results_df: Optional[pd.DataFrame] = None
    results_filter_query: Optional[str] = None
    print_results: Optional[bool] = False

    indicator_eval_d: Dict[str, str] = field(default_factory=lambda : {
        'Large SSL': "quant_num_vars == 160 and n_encoder_layers== 8",
        'Medium SSL': "quant_num_vars == 20 and n_encoder_layers== 8",
        'Small SSL': "quant_num_vars == 10 and n_encoder_layers== 4",
    })

    #pretrained_results_df['Large SSL'] = pretrained_results_df.eval("quant_num_vars == 160 and n_encoder_layers== 8")
    #pretrained_results_df['Medium SSL'] = pretrained_results_df.eval("quant_num_vars == 20 and n_encoder_layers== 8")
    #pretrained_results_df['Small SSL'] = pretrained_results_df.eval("quant_num_vars == 10 and n_encoder_layers== 4")

    # - Fine Tune specific static attrs -
    task_pub_d: ClassVar[Dict[str, str]] = {'word_classification_fine_tuning': 'Word Classification',
                                            'speech_classification_fine_tuning': 'Speech Activity Detection',
                                            'region_classification_fine_tuning': 'Speech-related Behavior Recognition'}

    task_pub_fmt_d: ClassVar[Dict[str, str]] = {'word_classification_fine_tuning': 'Word\nClassification',
                                                'speech_classification_fine_tuning': 'Speech Activity\nDetection',
                                                'region_classification_fine_tuning': 'Speech-related\nBehavior Recognition'}

    task_pub_fmt_rev_d: ClassVar[Dict[str, str]] = {v: k for k, v in task_pub_fmt_d.items()}

    task_pub_fmt_short_d: ClassVar[Dict[str, str]] = {'word_classification_fine_tuning': 'Word',
                                                      'speech_classification_fine_tuning': 'Speech',
                                                      'region_classification_fine_tuning': 'Behavior'}


    task_rate_d: ClassVar[Dict[str, float]] = {'word_classification_fine_tuning': .1,
                                               'speech_classification_fine_tuning': .5,
                                               'region_classification_fine_tuning': .25}
    def __post_init__(self):
        if self.results_df is None:
            self.results_df = load_results_to_frame(
                self.result_file
            )

        # result_options_df, pretrain_exper_df, mean_train_acc_ctab, mean_cv_acc_ctab, mean_test_acc_ctab = FineTuneResultParsingOptions.preprocess_results_frame(results_df)
        #self.parsed_results = self.preprocess_results_frame(self.results_df, indicator_eval_d=self.indicator_eval_d)
        self.result_options_df, opt_cols = upack_result_options_to_columns(self.results_df,
                                                                      parse_cv_and_epoch_results=False)


        pretrain_col = 'pretrained_results'
        if pretrain_col not in self.result_options_df.columns:
            pretrain_col = 'pretrained_model_opts'

        if pretrain_col in self.result_options_df.columns:

            self.result_options_df['pretrain_result_file'] = self.result_options_df[pretrain_col].apply(lambda d: d.get('name'))
        #pretrain_col = pretrain_col if pretrain_col in self.result_options_df.columns else 'pretrained_model_opts'
            self.pretrain_parser = self.preprocess_pretraining_results(self.result_options_df.drop_duplicates('pretrain_result_file').reset_index(),
                                                                       pretrained_results_col=pretrain_col,
                                                                       indicator_eval_d=self.indicator_eval_d
                                                                   )
        else:
            self.pretrain_parser = None


        self.result_options_df['parent_dir'] = self.result_options_df.path.apply(lambda s: os.path.split(s)[0])
        if self.results_filter_query is not None:
            self.result_options_df = self.result_options_df.query(self.results_filter_query)

        def parse_perf_col(ro_df, col):
            _perf_df = ro_df[col].apply(pd.Series)
            _perf_df.columns = f'{col}_' + _perf_df.columns
            return _perf_df

        perf_df_map = dict(train=parse_perf_col(self.result_options_df, 'train'),
                           cv=parse_perf_col(self.result_options_df, 'cv'),
                           test=parse_perf_col(self.result_options_df, 'test'))

        self.result_options_df = self.result_options_df.join(
            perf_df_map.values()
        )

        self.result_options_df['task_name_pub'] = self.result_options_df.task_name.map(self.task_pub_d)
        self.result_options_df['task_name_pub_fmt'] = self.result_options_df.task_name.map(self.task_pub_fmt_d)
        self.result_options_df['task_name_pub_fmt_short'] = self.result_options_df.task_name.map(self.task_pub_fmt_short_d)
        print("Adding short form task name")
        print(self.result_options_df['task_name_pub_fmt_short'].value_counts())


        train_set_size_df = self.map_sensor_count_columns_to_set_str(self.result_options_df.train_sets)
        self.result_options_df = self.result_options_df.join(train_set_size_df)

        self.result_options_df['train_cv_delta'] = self.result_options_df.cv_accuracy - self.result_options_df.train_accuracy

        print(self.result_options_df['task_name_pub_fmt_short'].value_counts())


        # -
        opt_df_map = {opt_col: self.result_options_df[opt_col].apply(pd.Series)
                      for opt_col in self.result_options_df.filter(like='_options').columns}
        print(opt_df_map.keys())

        self.model_option_cols = opt_df_map['model_options'].columns.tolist()

        options_df = opt_df_map['model_options'].copy()
        for opt_name, _df in opt_df_map.items():
            keep_cols = list(set(_df.columns.tolist()) - set(options_df.columns.tolist()))
            options_df = options_df.join(_df[keep_cols].copy())

        self.options_df = options_df
        self.opt_df_map = opt_df_map
        self.opt_cols = options_df.drop([#'model_name',
                                         'save_model_path'], axis=1).columns.tolist()
        unique_opt_counts_s = self.result_options_df[self.opt_cols].apply(
            lambda c: c.nunique() if c.map(is_hashable).all() else np.nan, axis=0)

        self.varied_opts_s: pd.Series = unique_opt_counts_s[unique_opt_counts_s > 1]


        if self.pretrain_parser is not None:
            self.pretrain_perf_w_ro_df = self.pretrain_parser.result_options_df.set_index('pretrain_name').merge(
                self.result_options_df,
                #left_on='result_options_ix',
                left_index=True,
                #right_index=True,
                right_on='pretrain_result_file',
                suffixes=('_pretrain', ''),
                indicator=True
            ).sort_index()

            assert self.pretrain_perf_w_ro_df._merge.eq('both').all()

            self.pretrain_perf_w_ro_df['pt_transfer_type'] = self.pretrain_perf_w_ro_df.apply(
                lambda r: r.train_sets in r.pretrain_sets,
                axis=1).replace({True: "Same Pt.", False: "Different Pt."})

            self.pretrain_perf_w_ro_df['Transfer Type'] = self.pretrain_perf_w_ro_df.pt_transfer_type.eq("Same Pt.").replace(
                {True: "Pretrain Pt.", False: "Non-Pretrain Pt."}
            )

            self.opt_cols += ['Transfer Type']

        self.opt_cols += ['task_name_pub', 'task_name_pub_fmt', 'task_name_pub_fmt_short']

        self.extract_epoch_results()

    def extract_epoch_results(self):
        import warnings
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

        opt_cols = self.opt_cols
        if self.pretrain_parser is not None:
            opt_cols += self.pretrain_parser.opt_cols
            _df = self.pretrain_perf_w_ro_df
        else:
            _df = self.result_options_df

        self.epoch_results_df = pd.concat([
            pd.DataFrame(row.epoch_outputs).T.assign(
                experiment_name=row['name'],
                uid=row.uid,
                model_name=row.model_name,
                # Options
                **{opt_name: row[opt_name] for opt_name in opt_cols if opt_name in row}
            )
            for ix, row in _df.iterrows()]).reset_index(names='epoch_num').copy()

        #cv_epoch_res_df = epoch_results_df.cv_loss_d.apply(pd.Series)
        #cv_epoch_res_df.columns = "cv_" + cv_epoch_res_df.columns

        #self.epoch_results_df = cv_epoch_res_df.join(epoch_results_df)

    @classmethod
    def map_sensor_count_columns_to_set_str(cls, pt_set_str_s):
        out_d = dict()
        out_d['num_sensors'] = pt_set_str_s.map(cls.n_sensors_d)
        out_d['more_than_sensors'] = out_d['num_sensors'].ge(100).map(
            {False: '$<$ 100 Sensors', True: '$\geq$ 100 Sensors'})
        out_d['num_sensor_cat'] = out_d['num_sensors'].pipe(pd.cut, bins=[70, 93, 105, 235],
                                                            labels=['small', 'med', 'large'], right=False)
        df = pd.DataFrame(out_d)
        df.columns = pt_set_str_s.name + '_' + df.columns
        return df

    @classmethod
    def preprocess_pretraining_results(cls, result_options_df: pd.DataFrame,
                                       pretrained_results_col: str = 'pretrained_results',
                                       indicator_eval_d: Optional[Dict[str, str]] = None):
        # - Parse out pretrained results -
        pretrained_results_df = result_options_df[pretrained_results_col].apply(pd.Series)
        return PretrainResultParsingOptions(results_df=pretrained_results_df)

    @classmethod
    def from_results_paths(cls, paths: List[str], **kws):
        return cls(results_df=pd.concat(
            (load_results_to_frame(p).assign(results_loaded_from_path=p) for p in paths),
            ignore_index=True
        ), **kws)


@dataclass
class InfoLeakageResultParsingOptions(FineTuneResultParsingOptions):
    #def __post_init__(self):
    #    super().__post_init__()

    def extract_epoch_results(self):
        epoch_fields = ['total_loss', 'cv_loss']
        _epoch_results_l = list()
        for ix, r in self.result_options_df.iterrows():
            m_results_d = r.attacker_model_train_results

            m_epoch_df = pd.DataFrame([pd.Series({k: _d[k] for k in epoch_fields}, name=epoch_n)
                                       for epoch_n, _d in m_results_d.items()])
            _epoch_results_l.append(
                m_epoch_df.assign(experiment_name=r['name'],
                                  uid=r.uid,
                                  # **{opt_name: r[opt_name] for opt_name in opt_cols}
                                  **{opt_name: r[opt_name] for opt_name in self.opt_df_map['model_options'].columns}
                                  )
            )

        self.epoch_results_df = pd.concat(_epoch_results_l).reset_index(names='epoch_num')

