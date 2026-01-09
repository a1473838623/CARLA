import numpy as np
import pandas as pd
import os

os.environ['NUMBA_DISABLE_CUDA'] = '1'
import argparse

from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, auc
)
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.metrics import get_range_vus_roc


def point_adjustment(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).copy()
    anomaly_state = False

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if y_true[j] == 0:
                    break
                y_pred[j] = 1
        elif y_true[i] == 0:
            anomaly_state = False

        if anomaly_state:
            y_pred[i] = 1

    return y_pred


def find_best_threshold(scores, labels, apply_pa=False):
    gt = labels.astype(int)
    percentiles = [(90 + (i / 10)) for i in range(100)]
    thresholds = np.percentile(scores, percentiles)

    best_f1, best_threshold = -1, thresholds[0]

    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        if apply_pa:
            pred = point_adjustment(gt, pred)
        _, _, f1, _ = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, thresh

    return best_threshold


def compute_label_based_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {
        'Precision': precision, 'Recall': recall, 'F-score': f_score, 'ACC': acc,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }


def compute_affiliation_metrics(y_true, y_pred):
    events_pred = convert_vector_to_events(y_pred)
    events_gt = convert_vector_to_events(y_true)

    if len(events_pred) == 0 or len(events_gt) == 0:
        return {'Aff-Pre': 0.0, 'Aff-Rec': 0.0, 'Aff-F1': 0.0}

    affiliation = pr_from_events(events_pred, events_gt, (0, len(y_true)))
    aff_pre, aff_rec = affiliation['precision'], affiliation['recall']
    aff_f1 = 2 * aff_pre * aff_rec / (aff_pre + aff_rec) if (aff_pre + aff_rec) > 0 else 0
    return {'Aff-Pre': aff_pre, 'Aff-Rec': aff_rec, 'Aff-F1': aff_f1}


def compute_score_based_metrics(y_true, scores, slidingWindow=100):
    result = {}
    try:
        result['AUC-ROC'] = roc_auc_score(y_true, scores)
        result['AUC-PR'] = average_precision_score(y_true, scores)
    except:
        result['AUC-ROC'] = result['AUC-PR'] = 0

    try:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        f1_scores = np.divide(2 * precision * recall, precision + recall,
                              out=np.zeros_like(precision), where=(precision + recall) != 0)
        result['F1_AUC'] = auc(recall[np.argsort(recall)], f1_scores[np.argsort(recall)])
    except:
        result['F1_AUC'] = 0

    try:
        vus = get_range_vus_roc(score=scores, labels=y_true, slidingWindow=slidingWindow)
        result['R_AUC_ROC'], result['R_AUC_PR'] = vus['R_AUC_ROC'], vus['R_AUC_PR']
        result['VUS_ROC'], result['VUS_PR'] = vus['VUS_ROC'], vus['VUS_PR']
    except:
        result['R_AUC_ROC'] = result['R_AUC_PR'] = result['VUS_ROC'] = result['VUS_PR'] = 0

    result['slidingWindow'] = slidingWindow
    return result


def compute_micro_label_metrics(res_df):
    """Micro 平均：汇总所有文件的 TP/FP/FN/TN 后再计算指标"""
    sum_tp, sum_tn = res_df['TP'].sum(), res_df['TN'].sum()
    sum_fp, sum_fn = res_df['FP'].sum(), res_df['FN'].sum()

    pre = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    acc = (sum_tp + sum_tn) / (sum_tp + sum_tn + sum_fp + sum_fn)

    return {
        'Precision': pre, 'Recall': rec, 'F-score': f1, 'ACC': acc,
        'TP': int(sum_tp), 'TN': int(sum_tn), 'FP': int(sum_fp), 'FN': int(sum_fn)
    }


def extract_scores_and_labels(df_train, df_test):
    """从 DataFrame 提取 scores 和 labels"""
    cl_num = df_train.shape[1] - 1
    df_train['pred'] = df_train[df_train.columns[0:cl_num]].idxmax(axis=1)
    score_col = df_train['pred'].value_counts().idxmax()

    scores = (1 - df_test[score_col]).values
    labels = np.where(df_test['Class'] == 0, 0, 1).astype(int)
    return scores, labels


def evaluate_single_file(df_train, df_test, use_pa=None, slidingWindow=100):
    results = {}
    scores, gt = extract_scores_and_labels(df_train, df_test)

    if use_pa is None or use_pa is True:
        thresh = find_best_threshold(scores, gt, apply_pa=True)
        pred = point_adjustment(gt, (scores >= thresh).astype(int))
        results['with_pa'] = {
            'threshold': thresh,
            'label_based': compute_label_based_metrics(gt, pred),
            'affiliation': compute_affiliation_metrics(gt, pred)
        }

    if use_pa is None or use_pa is False:
        thresh = find_best_threshold(scores, gt, apply_pa=False)
        pred = (scores >= thresh).astype(int)
        results['without_pa'] = {
            'threshold': thresh,
            'label_based': compute_label_based_metrics(gt, pred),
            'affiliation': compute_affiliation_metrics(gt, pred)
        }

    results['score_based'] = compute_score_based_metrics(gt, scores, slidingWindow)
    return results


def find_result_files(ds_name, fname, result_root='results'):
    for p in [
        os.path.join(result_root, ds_name, fname, 'classification'),
        os.path.join(result_root, ds_name, 'classification'),
    ]:
        train_file = os.path.join(p, 'classification_trainprobs.csv')
        test_file = os.path.join(p, 'classification_testprobs.csv')
        if os.path.exists(train_file) and os.path.exists(test_file):
            return train_file, test_file
    return None, None


def get_file_list(ds_name, result_root='results'):
    path = os.path.join(result_root, ds_name)
    if not os.path.exists(path):
        return []
    return [item for item in sorted(os.listdir(path))
            if os.path.isdir(os.path.join(path, item))
            and item not in ['All', 'classification']
            and os.path.exists(os.path.join(path, item, 'classification'))]


def print_results(results, name):
    print(f"\n{'=' * 70}")
    print(f"  Result: {name}")
    print(f"{'=' * 70}")

    for key, label in [('with_pa', 'WITH PA'), ('without_pa', 'WITHOUT PA')]:
        if key not in results:
            continue
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")
        print(f"  Threshold: {results[key]['threshold']:.6f}")

        lm = results[key]['label_based']
        print(f"\n  [Label-based metrics]")
        print(f"    Precision : {lm['Precision']:.4f}")
        print(f"    Recall    : {lm['Recall']:.4f}")
        print(f"    F-score   : {lm['F-score']:.4f}")
        print(f"    ACC       : {lm['ACC']:.4f}")
        print(f"    (TP={lm['TP']}, TN={lm['TN']}, FP={lm['FP']}, FN={lm['FN']})")

        am = results[key]['affiliation']
        print(f"\n  [Affiliation metrics]")
        print(f"    Aff-Pre   : {am['Aff-Pre']:.4f}")
        print(f"    Aff-Rec   : {am['Aff-Rec']:.4f}")
        print(f"    Aff-F1    : {am['Aff-F1']:.4f}")

    sm = results['score_based']
    print(f"\n{'─' * 70}")
    print(f"  SCORE-BASED METRICS (PA not applicable)")
    print(f"{'─' * 70}")
    print(f"    slidingWindow : {sm.get('slidingWindow', 'N/A')}")
    print(f"    AUC-ROC       : {sm['AUC-ROC']:.4f}")
    print(f"    AUC-PR        : {sm['AUC-PR']:.4f}")
    print(f"    R_AUC_ROC     : {sm['R_AUC_ROC']:.4f}")
    print(f"    R_AUC_PR      : {sm['R_AUC_PR']:.4f}")
    print(f"    VUS_ROC       : {sm['VUS_ROC']:.4f}")
    print(f"    VUS_PR        : {sm['VUS_PR']:.4f}")
    print(f"    F1_AUC        : {sm['F1_AUC']:.4f}")
    print(f"{'=' * 70}\n")


def evaluate_dataset(ds_name, fname='All', use_pa=None, slidingWindow=100, mode='auto', result_root='results'):
    print(f"{'=' * 70}")
    print(f"Dataset: {ds_name}, Fname: {fname}, Mode: {mode}")
    print(f"{'=' * 70}")

    # 自动检测模式
    if mode == 'auto':
        has_all = find_result_files(ds_name, 'All', result_root)[0] is not None
        has_single = len(get_file_list(ds_name, result_root)) > 0
        mode = 'all' if (fname == 'All' and has_all) else 'single'
        print(f"Selected mode: {mode}")

    # All 模式
    if mode == 'all':
        train_path, test_path = find_result_files(ds_name, 'All', result_root)
        if train_path is None:
            print("Error: Cannot find All mode result files")
            return None
        results = evaluate_single_file(pd.read_csv(train_path), pd.read_csv(test_path), use_pa, slidingWindow)
        print_results(results, f"{ds_name}/All")
        return results

    # Single 模式：逐文件评估，然后拼接所有数据计算汇总指标
    file_list = [fname] if fname != 'All' else get_file_list(ds_name, result_root)
    if not file_list:
        print("Error: No files found")
        return None

    print(f"Files: {len(file_list)}")

    # 收集每个文件的 scores 和 labels
    all_scores, all_labels = [], []
    pa_records, nopa_records = [], []

    for filename in file_list:
        train_path, test_path = find_result_files(ds_name, filename, result_root)
        if train_path is None:
            continue
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            # 提取 scores 和 labels
            scores, labels = extract_scores_and_labels(df_train, df_test)
            all_scores.append(scores)
            all_labels.append(labels)

            # 单文件评估（用于收集混淆矩阵）
            result = evaluate_single_file(df_train, df_test, use_pa, slidingWindow)

            if 'with_pa' in result:
                pa_records.append(result['with_pa']['label_based'])
            if 'without_pa' in result:
                nopa_records.append(result['without_pa']['label_based'])

            f_pa = result['with_pa']['label_based']['F-score'] if 'with_pa' in result else '-'
            f_nopa = result['without_pa']['label_based']['F-score'] if 'without_pa' in result else '-'
            print(f"  {filename}: F1(PA)={f_pa if f_pa == '-' else f'{f_pa:.4f}'}, "
                  f"F1(noPA)={f_nopa if f_nopa == '-' else f'{f_nopa:.4f}'}")
        except Exception as e:
            print(f"  {filename}: ERROR - {e}")

    if not all_scores:
        print("No valid results!")
        return None

    # 拼接所有文件的 scores 和 labels
    concat_scores = np.concatenate(all_scores)
    concat_labels = np.concatenate(all_labels)

    print(f"\n{'=' * 70}")
    print(f"  AGGREGATED RESULTS ({len(file_list)} files, {len(concat_labels)} samples)")
    print(f"{'=' * 70}")

    results = {}

    # WITH PA
    if use_pa is None or use_pa is True:
        print(f"\n{'─' * 70}")
        print(f"  WITH PA")
        print(f"{'─' * 70}")

        # Label-based: Micro 汇总
        micro_pa = compute_micro_label_metrics(pd.DataFrame(pa_records))
        print(f"\n  [Label-based metrics] (Micro: aggregate TP/FP/FN/TN)")
        print(f"    Precision : {micro_pa['Precision']:.4f}")
        print(f"    Recall    : {micro_pa['Recall']:.4f}")
        print(f"    F-score   : {micro_pa['F-score']:.4f}")
        print(f"    ACC       : {micro_pa['ACC']:.4f}")
        print(f"    (TP={micro_pa['TP']}, TN={micro_pa['TN']}, FP={micro_pa['FP']}, FN={micro_pa['FN']})")

        # Affiliation: 在拼接数据上重新计算
        thresh_pa = find_best_threshold(concat_scores, concat_labels, apply_pa=True)
        pred_pa = point_adjustment(concat_labels, (concat_scores >= thresh_pa).astype(int))
        aff_pa = compute_affiliation_metrics(concat_labels, pred_pa)
        print(f"\n  [Affiliation metrics] (on concatenated data)")
        print(f"    Aff-Pre   : {aff_pa['Aff-Pre']:.4f}")
        print(f"    Aff-Rec   : {aff_pa['Aff-Rec']:.4f}")
        print(f"    Aff-F1    : {aff_pa['Aff-F1']:.4f}")

        results['with_pa'] = {'label_based': micro_pa, 'affiliation': aff_pa}

    # WITHOUT PA
    if use_pa is None or use_pa is False:
        print(f"\n{'─' * 70}")
        print(f"  WITHOUT PA")
        print(f"{'─' * 70}")

        # Label-based: Micro 汇总
        micro_nopa = compute_micro_label_metrics(pd.DataFrame(nopa_records))
        print(f"\n  [Label-based metrics] (Micro: aggregate TP/FP/FN/TN)")
        print(f"    Precision : {micro_nopa['Precision']:.4f}")
        print(f"    Recall    : {micro_nopa['Recall']:.4f}")
        print(f"    F-score   : {micro_nopa['F-score']:.4f}")
        print(f"    ACC       : {micro_nopa['ACC']:.4f}")
        print(f"    (TP={micro_nopa['TP']}, TN={micro_nopa['TN']}, FP={micro_nopa['FP']}, FN={micro_nopa['FN']})")

        # Affiliation: 在拼接数据上重新计算
        thresh_nopa = find_best_threshold(concat_scores, concat_labels, apply_pa=False)
        pred_nopa = (concat_scores >= thresh_nopa).astype(int)
        aff_nopa = compute_affiliation_metrics(concat_labels, pred_nopa)
        print(f"\n  [Affiliation metrics] (on concatenated data)")
        print(f"    Aff-Pre   : {aff_nopa['Aff-Pre']:.4f}")
        print(f"    Aff-Rec   : {aff_nopa['Aff-Rec']:.4f}")
        print(f"    Aff-F1    : {aff_nopa['Aff-F1']:.4f}")

        results['without_pa'] = {'label_based': micro_nopa, 'affiliation': aff_nopa}

    # Score-based: 在拼接数据上计算
    print(f"\n{'─' * 70}")
    print(f"  SCORE-BASED METRICS (PA not applicable)")
    print(f"{'─' * 70}")
    score_metrics = compute_score_based_metrics(concat_labels, concat_scores, slidingWindow)
    print(f"    slidingWindow : {score_metrics['slidingWindow']}")
    print(f"    AUC-ROC       : {score_metrics['AUC-ROC']:.4f}")
    print(f"    AUC-PR        : {score_metrics['AUC-PR']:.4f}")
    print(f"    R_AUC_ROC     : {score_metrics['R_AUC_ROC']:.4f}")
    print(f"    R_AUC_PR      : {score_metrics['R_AUC_PR']:.4f}")
    print(f"    VUS_ROC       : {score_metrics['VUS_ROC']:.4f}")
    print(f"    VUS_PR        : {score_metrics['VUS_PR']:.4f}")
    print(f"    F1_AUC        : {score_metrics['F1_AUC']:.4f}")
    print(f"{'=' * 70}\n")

    results['score_based'] = score_metrics
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MSL')
    parser.add_argument('--fname', type=str, default='All')
    parser.add_argument('--use_pa', type=str, default='both', choices=['true', 'false', 'both'])
    parser.add_argument('--sliding_window', type=int, default=100)
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'single', 'all'])
    parser.add_argument('--result_root', type=str, default='results')
    args = parser.parse_args()

    evaluate_dataset(
        ds_name=args.dataset,
        fname=args.fname,
        use_pa=None if args.use_pa == 'both' else (args.use_pa == 'true'),
        slidingWindow=args.sliding_window,
        mode=args.mode,
        result_root=args.result_root
    )