import pandas as pd
import numpy as np


def return_new_columns(f):
    def _f(*args, **kws):
        start_cols = args[1].columns.tolist()
        r = f(*args, **kws)
        end_cols = r.columns.tolist()
        new_cols = list(set(end_cols) - set(start_cols))
        # Order them the same
        return r, [col for col in end_cols if col in new_cols]
    return _f

def check_and_join_labels(result_options_df: pd.DataFrame,
                          _labels_df: pd.DataFrame,
                          from_dummies_name: str):
    over_defined = _labels_df.sum(axis=1).gt(1)
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


def confusion_matrix_for(cm_df, c_id):
    TP = cm_df.loc[c_id, c_id]
    FN = cm_df.loc[c_id, :].sum() - TP
    FP = cm_df.loc[:, c_id].sum() - TP
    TN = cm_df.values.sum() - TP - FN - FP
    return TP, TN, FP, FN

#def class_tp_from_cm(cm_df, c_id):
#    return cm_df.loc[c_id, c_id]
#
#def class_tn_from_cm(cm_df, c_id):
#    return cm_df.drop(c_id, axis=0).loc[c_id, c_id]
#
#def class_fp_from_cm(cm_df, c_id):
#    return cm_df.drop(c_id).loc[:, c_id].sum()
#
#
#def class_fn_from_cm(cm_df, c_id):
#    return cm_df.drop(c_id, axis=1).loc[c_id, :].sum()


def class_precision_from_cm(cm_df, c_id):
    #tp = class_tp_from_cm(cm_df, c_id)
    #fp = class_fp_from_cm(cm_df, c_id)
    tp, tn, fp, fn = confusion_matrix_for(cm_df, c_id)
    return tp / (tp + fp)


def class_recall_from_cm(cm_df, c_id):
    #tp = class_tp_from_cm(cm_df, c_id)
    #fn = class_fn_from_cm(cm_df, c_id)
    tp, tn, fp, fn = confusion_matrix_for(cm_df, c_id)
    return tp / (tp + fn)


def class_f1_from_cm(cm_df, c_id):
    pr = class_precision_from_cm(cm_df, c_id)
    re = class_recall_from_cm(cm_df, c_id)

    return (2 * (pr * re)) / (pr + re)

def class_acc_from_cm(cm_df, c_id):
    tp, tn, fp, fn = confusion_matrix_for(cm_df, c_id)
    return (tp + tn) / (tp + tn + fp + fn)
