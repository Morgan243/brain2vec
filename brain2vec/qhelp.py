import pandas as pd
import numpy as np
import os


def return_new_columns(f):
    def _f(*args, **kws):
        start_cols = args[1].columns.tolist()
        r = f(*args, **kws)
        end_cols = r.columns.tolist()
        new_cols = list(set(end_cols) - set(start_cols))
        # Order them the same
        return r, [col for col in end_cols if col in new_cols]
    return _f


class QHelpFinetune:

    @classmethod
    def create_pretrain_labels(cls):
        pass


class QHelpPretrain:
    featenc_labels_d = {
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

    posenc_labels_d = {
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

    approach_labels_d = {
        #'Original': "`Positional Encoding`.eq('CombinedRAS-Simple') & `Feature Extractor`.eq('5x32ch(original)')",
        #'Small Embedding': "`Positional Encoding`.eq('CombinedRAS-Multihead') & `Feature Extractor`.eq('5x16ch(original)')",
        #'Small Embedding with Wide Field': "`Positional Encoding`.eq('PosRAS-Multihead') & `Feature Extractor`.eq('wide2-4x16ch(original)')",

        #'Small Embedding v2': "`Positional Encoding`.eq('CombinedRAS-Multihead') & `Feature Extractor`.eq('5x16ch(original)')",
        #'Small Embedding with Wide Field v2': "`Positional Encoding`.eq('PosRAS-Multihead') & `Feature Extractor`.eq('wide2-3x16ch(original)')",
        #'': "`Positional Encoding`.eq('CombinedRAS-Multihead') & `Feature Extractor`.eq('wide2-3x16ch(original)')",
    }

    #base_approach_eval_str = """n_encoder_layers.eq(6) & quant_num_vars.eq(20) & feature_grad_mult.eq(1) & parent_dir.ne('../results_local/results_pretrain_1pair_230507')"""
    base_approach_eval_str = """n_encoder_layers.eq(6) & quant_num_vars.eq(20) & parent_dir.ne('../results_local/results_pretrain_1pair_230507')"""

    @classmethod
    def extract_epoch_results(cls, result_options_df, option_cols):
        if 'training_complete' not in result_options_df.columns or result_options_df.training_complete.any():
            cv_results_df = pd.concat([
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
        #epoch_results_df['pct_loss_delta'] = epoch_results_df.eval("(total_loss - cv_loss) / total_loss")
        epoch_results_df['codewords_used'] = (1 - epoch_results_df.cv_perplexity) * (
                    epoch_results_df.quant_num_vars ** epoch_results_df.quant_num_groups)

        return epoch_results_df

#    @classmethod
#    def extract_extra(cls, result_options_df, option_cols):
#        if 'model_output_shape' in result_options_df.columns:
#            result_options_df['model_output_shape_str'] = result_options_df.model_output_shape \
#                .apply(lambda s: tuple(s) if isinstance(s, list) else tuple()) \
#                .apply(lambda l: "x".join(str(t) for t in l[1:]) if len(l) > 0 else "NA"
#                       )
#            if 'model_output_shape_str' not in option_cols:
#                option_cols.append('model_output_shape_str')
#
#        result_options_df['parent_dir'] = result_options_df.path.apply(lambda s: os.path.split(s)[0])
#        result_options_df['feature_extractor_layers_str'] = result_options_df.feature_extractor_layers.apply(eval).apply(lambda l: "|".join(".".join(str(_t) for _t in t) for t in l))
#        if 'feature_extractor_layers_str' not in option_cols:
#            option_cols.append('feature_extractor_layers_str')
#
#        return result_options_df, option_cols


    @classmethod
    @return_new_columns
    def create_labels_for_feature_encoder(cls, result_options_df: pd.DataFrame):

        featenc_lables_d = {k: result_options_df.eval(s).rename(k)
                            for k, s in cls.featenc_labels_d.items()}
        featenc_labels_df = pd.concat(featenc_lables_d.values(), axis=1)
        print(featenc_labels_df.value_counts().to_frame())
        result_options_df = cls.check_and_join_labels(result_options_df, featenc_labels_df,
                                                      from_dummies_name='Feature Extractor')
        return result_options_df


    @classmethod
    @return_new_columns
    def create_labels_for_posenc(cls, result_options_df: pd.DataFrame):
        posenc_lables_d = {k: result_options_df.eval(s).rename(k) for k, s in cls.posenc_labels_d.items()}
        posenc_labels_df = pd.concat(posenc_lables_d.values(), axis=1)
        print(posenc_labels_df.value_counts().to_frame())

        result_options_df = cls.check_and_join_labels(result_options_df, posenc_labels_df,
                                                      from_dummies_name='Positional Encoding')
        return result_options_df
        #return posenc_labels_df

    @classmethod
    @return_new_columns
    def create_labels_for_approach(cls, result_options_df: pd.DataFrame,
                                   base_approach_eval_str: str = None):

        base_approach_eval_str = cls.base_approach_eval_str if base_approach_eval_str is None else base_approach_eval_str
        if isinstance(base_approach_eval_str, str) and len(base_approach_eval_str.strip()) > 0:
            base_approach_m = result_options_df.eval(base_approach_eval_str).rename('base_approach')
        else:
            base_approach_m = pd.Series(True, index=result_options_df.index, name='base_approach')

        #base_approach_m = result_options_df.n_encoder_layers.eq(6) & result_options_df.quant_num_vars.eq(
        #    20) & result_options_df.feature_grad_mult.eq(1)
        #base_approach_m &= ~result_options_df.parent_dir.eq('../results_local/results_pretrain_1pair_230507')

#        approach_labels_d = {
#            'Original':
#                result_options_df['Positional Encoding'].eq('CombinedRAS-Simple')
#                & result_options_df['Feature Extractor'].eq('5x32ch(original)')
#            # & ~result_options_df['name'].eq('20230511_0411_b5068fd8-82ff-40ad-b89a-b31346413d82.json')
#            ,
#
#            'Small Embedding':
#                result_options_df['Positional Encoding'].eq('CombinedRAS-Multihead')
#                & result_options_df['Feature Extractor'].eq('5x16ch(original)')
#            ,
#
#            # 'Wide Field with Smaller Embedding':
#            # result_options_df['Positional Encoding'].eq('CombinedRAS-Multihead')
#            # & result_options_df['Feature Extractor'].eq('wide2-4x16ch(original)'),
#
#            'Small Embedding with Wide Field':
#                result_options_df['Positional Encoding'].eq('PosRAS-Multihead')
#                & result_options_df['Feature Extractor'].eq('wide2-4x16ch(original)')
#            ,
#
#            # 'CombinedRAS-Multihead':result_options_df.eval("positional_encoding_method == 'combined' and ras_pos_encoding and (ras_architecture == 'multihead')"),
#            # 'PosRAS-Multihead':result_options_df.eval("positional_encoding_method == 'ras_pos' and ras_pos_encoding and (ras_architecture == 'multihead')"),
#            # 'PosRAS-Simple':result_options_df.eval("positional_encoding_method == 'ras_pos' and ras_pos_encoding and (ras_architecture == 'simple')")
#        }
        approach_labels_d = {k: (base_approach_m & result_options_df.eval(s)).rename(k)
                             for k, s in cls.approach_labels_d.items()}
        if len(approach_labels_d) == 0:
            return result_options_df.assign(Approach=np.nan)
        approach_labels_df = pd.concat(approach_labels_d.values(), axis=1)
        print(approach_labels_df.value_counts().to_frame())
        result_options_df = cls.check_and_join_labels(result_options_df, approach_labels_df,
                                                      from_dummies_name='Approach')
        return result_options_df


    @classmethod
    def check_and_join_labels(cls, result_options_df: pd.DataFrame,
                              _labels_df: pd.DataFrame,
                              from_dummies_name: str
                              ):
        over_defined = _labels_df.sum(1).gt(1)
        od_df = _labels_df.loc[over_defined]
        if len(od_df) > 0:
            msg = f"{len(od_df)} samples have duplicate membership across labels in {from_dummies_name}"
            print(msg)
            print(od_df.loc[:, od_df.any()])
            print("-"*30)

        #assert _labels_df.sum(1).le(1).all(), msg

        result_options_df.loc[:, _labels_df.columns] = _labels_df

        result_options_df[from_dummies_name] = pd.from_dummies(_labels_df, default_category=np.nan)['']
        # move the column to the end
        cols = result_options_df.columns.drop(from_dummies_name)
        return result_options_df[cols.tolist() + [from_dummies_name]]


