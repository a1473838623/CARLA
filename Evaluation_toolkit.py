import numpy as np
import pandas as pd
import os

from metrics.f1_score_f1_pa import *
from metrics.fc_score import *
from metrics.precision_at_k import *
from metrics.customizable_f1_score import *
from metrics.AUC import *
from metrics.Matthews_correlation_coefficient import *
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events


from pate.PATE_metric import PATE

from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve


def combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores):
    events_pred = convert_vector_to_events(pred_labels)
    events_gt = convert_vector_to_events(y_test)
    Trange = (0, len(y_test))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    true_events = get_events(y_test)

    score_list = {
        "Affiliation precision": affiliation['precision'],
        "Affiliation recall": affiliation['recall'],
    }

    return score_list


def adjust_predicts(label, predict=None, calc_latency=False):
    label = np.asarray(label)
    latency = 0

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(actual)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True

    MCM = metrics.multilabel_confusion_matrix(actual, predict, labels=[1, 0])

    pa_tn = MCM[0][0, 0]
    pa_tp = MCM[0][1, 1]
    pa_fp = MCM[0][0, 1]
    pa_fn = MCM[0][1, 0]

    prec = pa_tp / (pa_tp + pa_fp)
    rec = pa_tp / (pa_tp + pa_fn)
    if (prec + rec) == 0:
        f1_score = 0
    else:
        f1_score = 2 * (prec * rec) / (prec + rec)

    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4), pa_tp, pa_tn, pa_fp, pa_fn, prec, rec, f1_score
    else:
        return predict, prec, rec, f1_score, pa_tp, pa_tn, pa_fp, pa_fn


def add_summary_statistics(res_df):
    # Compute the sum of 'best_tp', 'best_tn', 'best_fp', 'best_fn'
    sum_best_tp = res_df['best_tp'].sum()
    sum_best_tn = res_df['best_tn'].sum()
    sum_best_fp = res_df['best_fp'].sum()
    sum_best_fn = res_df['best_fn'].sum()

    # Calculate precision, recall and f1 score
    precision = sum_best_tp / (sum_best_tp + sum_best_fp) if (sum_best_tp + sum_best_fp) > 0 else 0
    recall = sum_best_tp / (sum_best_tp + sum_best_fn) if (sum_best_tp + sum_best_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate the average and std of 'roc' and 'pr'
    roc_avg = res_df['roc'].mean()
    roc_std = res_df['roc'].std()
    pr_avg = res_df['pr'].mean()
    pr_std = res_df['pr'].std()

    sum_pa_tp = res_df['pa_tp'].sum()
    sum_pa_tn = res_df['pa_tn'].sum()
    sum_pa_fp = res_df['pa_fp'].sum()
    sum_pa_fn = res_df['pa_fn'].sum()

    # Calculate precision, recall and f1 score
    pa_precision = sum_pa_tp / (sum_pa_tp + sum_pa_fp) if (sum_pa_tp + sum_pa_fp) > 0 else 0
    pa_recall = sum_pa_tp / (sum_pa_tp + sum_pa_fn) if (sum_pa_tp + sum_pa_fn) > 0 else 0
    pa_f1_score = 2 * (pa_precision * pa_recall) / (pa_precision + pa_recall) if (pa_precision + pa_recall) > 0 else 0

    aff_f1_mean = res_df['aff_f1'].mean()
    aff_f1_std = res_df['aff_f1'].std()

    R_AUC_ROC_mean = res_df['R_AUC_ROC'].mean()
    R_AUC_ROC_std = res_df['R_AUC_ROC'].std()

    R_AUC_PR_mean = res_df['R_AUC_PR'].mean()
    R_AUC_PR_std = res_df['R_AUC_PR'].std()

    VUS_ROC_mean = res_df['VUS_ROC'].mean()
    VUS_ROC_std = res_df['VUS_ROC'].std()

    VUS_PR_mean = res_df['VUS_PR'].mean()
    VUS_PR_std = res_df['VUS_PR'].std()

    PATE_mean = res_df['PATE'].mean()
    PATE_std = res_df['PATE'].std()

    PATE_F1_mean = res_df['PATE_F1'].mean()
    PATE_F1_std = res_df['PATE_F1'].std()

    summary_row = pd.Series({
        'best_pre': precision,
        'best_rec': recall,
        'b_f_1': f1_score,
        'roc': roc_avg,
        'roc_std': roc_std,
        'pr': pr_avg,
        'pr_std': pr_std,
        'aff_f1_mean': aff_f1_mean,
        'aff_f1_std': aff_f1_std,
        'PATE_mean': PATE_mean,
        'PATE_std': PATE_std,
        'PATE_F1_mean': PATE_F1_mean,
        'PATE_F1_std': PATE_F1_std
    })

    return summary_row


if __name__ == "__main__":
    res_df = pd.DataFrame(
        columns=['name', 'roc', 'pr', 'best_tp', 'best_tn', 'best_fp', 'best_fn', 'best_pre', 'best_rec',
                 'b_f_1', 'pa_tp', 'pa_tn', 'pa_fp', 'pa_fn', 'pa_pre', 'pa_rec', 'pa_f1', 'latency',
                 'aff_pre', 'aff_rec', 'aff_f1', 'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 'PATE', 'PATE_F1'])

    ds_name = 'MSL'
    path = os.path.join('results/', ds_name)
    file_list = os.listdir(path)
    file_list = sorted(file_list)

    for filename in file_list:
        if filename != 'GECCO' and filename != '.json':
            print(filename)
            df_train = pd.read_csv(f"results/{ds_name}/{filename}/classification/classification_trainprobs.csv")
            df_test = pd.read_csv(f"results/{ds_name}/{filename}/classification/classification_testprobs.csv")
            cl_num = df_train.shape[1] - 1

            df_train['Class'] = np.where((df_train['Class'] == 0), 0, 1)
            df_train['pred'] = df_train[df_train.columns[0:cl_num]].idxmax(axis=1)

            score_col = df_train['pred'].value_counts().idxmax()

            df_test['Class'] = np.where((df_test['Class'] == 0), 0, 1)
            df_test['pred'] = df_test[df_test.columns[0:cl_num]].idxmax(axis=1)

            roc_auc, pr_auc, best_tn, best_tp, best_fp, best_fn, best_pre, best_rec, best_f1 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            aff_pre, aff_rec, aff_f1 = 0, 0, 0
            R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR = 0, 0, 0, 0
            pate, pate_f1 = 0, 0
            latency, pa_tp, pa_tn, pa_fp, pa_fn, pa_pre, pa_rec, pa_f1 = 0, 0, 0, 0, 0, 0, 0, 0
            thresholds = []
            scores = None

            try:
                df_test['pred'] = np.where((df_test['pred'] == score_col), 0, 1)

                MCM = metrics.multilabel_confusion_matrix(df_test['Class'], df_test['pred'], labels=[1, 0])

                tn = MCM[0][0, 0]
                tp = MCM[0][1, 1]
                fp = MCM[0][0, 1]
                fn = MCM[0][1, 0]

                if (tp + fp) == 0:
                    pre = 0
                else:
                    pre = tp / (tp + fp)
                if (tp + fn) == 0:
                    recall = 0
                else:
                    recall = tp / (tp + fn)
                if (pre + recall) == 0:
                    f_1 = 0
                else:
                    f_1 = 2 * pre * recall / (pre + recall)

                scores = 1 - df_test[score_col]

                roc_auc = roc_auc_score(df_test['Class'], scores)
                pr_auc = average_precision_score(df_test['Class'], scores)

                fpr, tpr, thresholds_roc = roc_curve(df_test['Class'], scores, pos_label=1)
                precision, recall, thresholds = precision_recall_curve(df_test['Class'], scores, pos_label=1)

                res = pd.DataFrame()
                res['pre'] = precision
                res['rec'] = recall
                res['f1'] = 2 * res['pre'] * res['rec'] / (res['pre'] + res['rec'])
                best_idx = res['f1'].argmax()
                best_f1 = res['f1'][best_idx]
                best_pre = res['pre'][best_idx]
                best_rec = res['rec'][best_idx]
                best_thr = thresholds[best_idx]
                print('Best f1 : ', best_f1, 'best_thr', best_thr)
                anomalies = [True if s >= best_thr else False for s in scores]

                best_tn, best_fp, best_fn, best_tp = confusion_matrix(df_test['Class'], anomalies).ravel()

                all_evals = combine_all_evaluation_scores(df_test['Class'].values, np.array(anomalies), scores.values)

                aff_pre = all_evals['Affiliation precision']
                aff_rec = all_evals['Affiliation recall']
                aff_f1 = 2 * (aff_pre * aff_rec) / (aff_pre + aff_rec) if (aff_pre + aff_rec) > 0 else 0

                R_AUC_ROC = 0
                R_AUC_PR = 0
                VUS_ROC = 0
                VUS_PR = 0

                pate = PATE(df_test['Class'].values, scores.values, binary_scores=False, n_jobs=-1)
                pate_f1 = PATE(df_test['Class'].values, np.array(anomalies), binary_scores=True, n_jobs=-1)
                print('PATE: ', pate, '\t PATE_F1: ', pate_f1)

            except ValueError:
                print('ERROR ++++++++++++++')
                pass

            # PA F1 search
            pa_f1 = -1
            if scores is not None and len(thresholds) > 0:
                for thr in thresholds:
                    preds_pa = [True if s >= thr else False for s in scores]
                    pa_prediction, t_latency, t_tp, t_tn, t_fp, t_fn, t_pre, t_rec, t_f1 = adjust_predicts(
                        df_test['Class'], preds_pa, True)
                    if t_f1 > pa_f1:
                        latency, pa_tp, pa_tn, pa_fp, pa_fn, pa_pre, pa_rec, pa_f1 = t_latency, t_tp, t_tn, t_fp, t_fn, t_pre, t_rec, t_f1

            new_row = pd.Series(
                [filename, roc_auc, pr_auc, best_tp, best_tn, best_fp, best_fn, best_pre, best_rec, best_f1,
                 pa_tp, pa_tn, pa_fp, pa_fn, pa_pre, pa_rec, pa_f1, latency, aff_pre, aff_rec, aff_f1,
                 R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, pate, pate_f1],
                index=['name', 'roc', 'pr', 'best_tp', 'best_tn', 'best_fp', 'best_fn', 'best_pre', 'best_rec',
                       'b_f_1', 'pa_tp', 'pa_tn', 'pa_fp', 'pa_fn', 'pa_pre', 'pa_rec', 'pa_f1', 'latency',
                       'aff_pre', 'aff_rec', 'aff_f1', 'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 'PATE', 'PATE_F1'])

            res_df = res_df._append(new_row, ignore_index=True)

    final_results = add_summary_statistics(res_df)
    output_file = os.path.join(path, f'{ds_name}_cuda_allmetrics.csv')
    res_df.to_csv(output_file)
    print(f"Results saved to {output_file}")
    print("\nSummary Statistics:")
    print(final_results)