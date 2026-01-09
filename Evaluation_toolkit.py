import numpy as np
import pandas as pd
import os
os.environ['NUMBA_DISABLE_CUDA'] = '1'
import argparse

from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.metrics import get_range_vus_roc

from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, auc
)


# ============================================================
# PA (Point Adjustment) 函数
# ============================================================
def point_adjustment(y_true, y_pred):
    """
    Point Adjustment: 如果一个异常段内有任意一个点被检测到，
    则将该异常段内的所有点都标记为检测到。
    """
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


# ============================================================
# 指标计算函数
# ============================================================
def compute_affiliation_metrics(y_true, y_pred):
    """计算 Affiliation 指标"""
    events_pred = convert_vector_to_events(y_pred)
    events_gt = convert_vector_to_events(y_true)
    Trange = (0, len(y_true))
    affiliation = pr_from_events(events_pred, events_gt, Trange)

    aff_pre = affiliation['precision']
    aff_rec = affiliation['recall']
    aff_f1 = 2 * aff_pre * aff_rec / (aff_pre + aff_rec) if (aff_pre + aff_rec) > 0 else 0

    return aff_pre, aff_rec, aff_f1


def compute_f1_auc(y_true, scores):
    """计算 F1-AUC"""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0
    )
    sorted_indices = np.argsort(recall)
    return auc(recall[sorted_indices], f1_scores[sorted_indices])


def compute_label_based_metrics(y_true, y_pred):
    """计算基于标签的指标：Precision, Recall, F-score, ACC"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)

    return {
        'Precision': precision,
        'Recall': recall,
        'F-score': f_score,
        'ACC': acc,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }


def compute_score_based_metrics(y_true, scores, slidingWindow=100):
    """计算基于分数的指标（不受 PA 影响）"""
    result = {}

    result['AUC-ROC'] = roc_auc_score(y_true, scores)
    result['AUC-PR'] = average_precision_score(y_true, scores)
    result['F1_AUC'] = compute_f1_auc(y_true, scores)

    try:
        vus_metrics = get_range_vus_roc(
            score=scores,
            labels=y_true,
            slidingWindow=slidingWindow
        )
        result['R_AUC_ROC'] = vus_metrics['R_AUC_ROC']
        result['R_AUC_PR'] = vus_metrics['R_AUC_PR']
        result['VUS_ROC'] = vus_metrics['VUS_ROC']
        result['VUS_PR'] = vus_metrics['VUS_PR']
    except Exception as e:
        print(f"  Warning: VUS/R_AUC calculation failed: {e}")
        result['R_AUC_ROC'] = 0
        result['R_AUC_PR'] = 0
        result['VUS_ROC'] = 0
        result['VUS_PR'] = 0

    return result


def find_best_threshold(y_true, scores):
    """找到最佳 F1 阈值"""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    return best_threshold


# ============================================================
# 单通道评估函数（同时返回 PA 和 noPA）
# ============================================================
def evaluate_single_channel(df_train, df_test, slidingWindow):
    """评估单个通道，同时返回 PA 和 noPA 结果"""
    result = {}

    cl_num = df_train.shape[1] - 1

    # 确定正常类别（原始逻辑）
    df_train['pred'] = df_train[df_train.columns[0:cl_num]].idxmax(axis=1)
    score_col = df_train['pred'].value_counts().idxmax()

    # 准备数据
    y_true = np.where(df_test['Class'] == 0, 0, 1)
    scores = (1 - df_test[score_col]).values

    # 找最佳阈值
    best_threshold = find_best_threshold(y_true, scores)
    y_pred_raw = (scores >= best_threshold).astype(int)

    result['threshold'] = best_threshold

    # 基于分数的指标（不受 PA 影响）
    score_metrics = compute_score_based_metrics(y_true, scores, slidingWindow)
    result.update(score_metrics)

    # Without PA
    label_nopa = compute_label_based_metrics(y_true, y_pred_raw)
    aff_pre_nopa, aff_rec_nopa, aff_f1_nopa = compute_affiliation_metrics(y_true, y_pred_raw)
    result['Precision_noPA'] = label_nopa['Precision']
    result['Recall_noPA'] = label_nopa['Recall']
    result['F-score_noPA'] = label_nopa['F-score']
    result['ACC_noPA'] = label_nopa['ACC']
    result['TP_noPA'] = label_nopa['TP']
    result['TN_noPA'] = label_nopa['TN']
    result['FP_noPA'] = label_nopa['FP']
    result['FN_noPA'] = label_nopa['FN']
    result['Aff-Pre_noPA'] = aff_pre_nopa
    result['Aff-Rec_noPA'] = aff_rec_nopa
    result['Aff-F1_noPA'] = aff_f1_nopa

    # With PA
    y_pred_pa = point_adjustment(y_true, y_pred_raw)
    label_pa = compute_label_based_metrics(y_true, y_pred_pa)
    aff_pre_pa, aff_rec_pa, aff_f1_pa = compute_affiliation_metrics(y_true, y_pred_pa)
    result['Precision_PA'] = label_pa['Precision']
    result['Recall_PA'] = label_pa['Recall']
    result['F-score_PA'] = label_pa['F-score']
    result['ACC_PA'] = label_pa['ACC']
    result['TP_PA'] = label_pa['TP']
    result['TN_PA'] = label_pa['TN']
    result['FP_PA'] = label_pa['FP']
    result['FN_PA'] = label_pa['FN']
    result['Aff-Pre_PA'] = aff_pre_pa
    result['Aff-Rec_PA'] = aff_rec_pa
    result['Aff-F1_PA'] = aff_f1_pa

    return result


# ============================================================
# 统计汇总函数
# ============================================================
def compute_macro_metrics(res_df):
    """计算 Macro 平均"""
    metric_cols = [
        'Precision_noPA', 'Recall_noPA', 'F-score_noPA', 'ACC_noPA',
        'Aff-Pre_noPA', 'Aff-Rec_noPA', 'Aff-F1_noPA',
        'Precision_PA', 'Recall_PA', 'F-score_PA', 'ACC_PA',
        'Aff-Pre_PA', 'Aff-Rec_PA', 'Aff-F1_PA',
        'AUC-ROC', 'AUC-PR', 'R_AUC_ROC', 'R_AUC_PR',
        'VUS_ROC', 'VUS_PR', 'F1_AUC'
    ]

    macro = {}
    n_samples = len(res_df)

    for col in metric_cols:
        if col in res_df.columns:
            macro[f'{col}_mean'] = res_df[col].mean()
            macro[f'{col}_std'] = res_df[col].std() if n_samples > 1 else 0.0

    return macro


def compute_micro_metrics(res_df):
    """计算 Micro 平均"""
    result = {}

    for suffix in ['_noPA', '_PA']:
        tp_col, tn_col = f'TP{suffix}', f'TN{suffix}'
        fp_col, fn_col = f'FP{suffix}', f'FN{suffix}'

        if not all(col in res_df.columns for col in [tp_col, tn_col, fp_col, fn_col]):
            continue

        sum_tp = res_df[tp_col].sum()
        sum_tn = res_df[tn_col].sum()
        sum_fp = res_df[fp_col].sum()
        sum_fn = res_df[fn_col].sum()

        micro_pre = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
        micro_rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
        micro_f1 = 2 * micro_pre * micro_rec / (micro_pre + micro_rec) if (micro_pre + micro_rec) > 0 else 0
        micro_acc = (sum_tp + sum_tn) / (sum_tp + sum_tn + sum_fp + sum_fn)

        result[f'Precision{suffix}'] = micro_pre
        result[f'Recall{suffix}'] = micro_rec
        result[f'F-score{suffix}'] = micro_f1
        result[f'ACC{suffix}'] = micro_acc
        result[f'TP{suffix}'] = int(sum_tp)
        result[f'TN{suffix}'] = int(sum_tn)
        result[f'FP{suffix}'] = int(sum_fp)
        result[f'FN{suffix}'] = int(sum_fn)

    return result


# ============================================================
# 打印函数
# ============================================================
def print_single_result(result, name):
    """打印单个结果"""
    print(f"\n{'=' * 70}")
    print(f"  Result: {name}")
    print(f"{'=' * 70}")

    print(f"\n{'─' * 70}")
    print(f"  WITHOUT Point Adjustment")
    print(f"{'─' * 70}")
    print(f"  [Label-based]")
    print(f"    Precision : {result['Precision_noPA']:.4f}")
    print(f"    Recall    : {result['Recall_noPA']:.4f}")
    print(f"    F-score   : {result['F-score_noPA']:.4f}")
    print(f"    ACC       : {result['ACC_noPA']:.4f}")
    print(f"    (TP={result['TP_noPA']}, TN={result['TN_noPA']}, FP={result['FP_noPA']}, FN={result['FN_noPA']})")
    print(f"  [Affiliation]")
    print(f"    Aff-Pre   : {result['Aff-Pre_noPA']:.4f}")
    print(f"    Aff-Rec   : {result['Aff-Rec_noPA']:.4f}")
    print(f"    Aff-F1    : {result['Aff-F1_noPA']:.4f}")

    print(f"\n{'─' * 70}")
    print(f"  WITH Point Adjustment")
    print(f"{'─' * 70}")
    print(f"  [Label-based]")
    print(f"    Precision : {result['Precision_PA']:.4f}")
    print(f"    Recall    : {result['Recall_PA']:.4f}")
    print(f"    F-score   : {result['F-score_PA']:.4f}")
    print(f"    ACC       : {result['ACC_PA']:.4f}")
    print(f"    (TP={result['TP_PA']}, TN={result['TN_PA']}, FP={result['FP_PA']}, FN={result['FN_PA']})")
    print(f"  [Affiliation]")
    print(f"    Aff-Pre   : {result['Aff-Pre_PA']:.4f}")
    print(f"    Aff-Rec   : {result['Aff-Rec_PA']:.4f}")
    print(f"    Aff-F1    : {result['Aff-F1_PA']:.4f}")

    print(f"\n{'─' * 70}")
    print(f"  SCORE-BASED (PA not applicable)")
    print(f"{'─' * 70}")
    print(f"    AUC-ROC   : {result['AUC-ROC']:.4f}")
    print(f"    AUC-PR    : {result['AUC-PR']:.4f}")
    print(f"    R_AUC_ROC : {result['R_AUC_ROC']:.4f}")
    print(f"    R_AUC_PR  : {result['R_AUC_PR']:.4f}")
    print(f"    VUS_ROC   : {result['VUS_ROC']:.4f}")
    print(f"    VUS_PR    : {result['VUS_PR']:.4f}")
    print(f"    F1_AUC    : {result['F1_AUC']:.4f}")
    print(f"{'=' * 70}\n")


def print_multi_channel_summary(macro, micro, n_channels):
    """打印多通道汇总结果"""
    print(f"\n{'=' * 70}")
    print(f"  AGGREGATED RESULTS ({n_channels} channels)")
    print(f"{'=' * 70}")

    print(f"\n{'─' * 70}")
    print(f"  WITHOUT Point Adjustment")
    print(f"{'─' * 70}")

    print(f"\n  [MACRO] (mean ± std)")
    for metric in ['Precision', 'Recall', 'F-score', 'ACC']:
        mean_val = macro.get(f'{metric}_noPA_mean', 0)
        std_val = macro.get(f'{metric}_noPA_std', 0)
        print(f"    {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
    for metric in ['Aff-Pre', 'Aff-Rec', 'Aff-F1']:
        mean_val = macro.get(f'{metric}_noPA_mean', 0)
        std_val = macro.get(f'{metric}_noPA_std', 0)
        print(f"    {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\n  [MICRO] (aggregate TP/FP/FN/TN)")
    print(f"    Precision   : {micro['Precision_noPA']:.4f}")
    print(f"    Recall      : {micro['Recall_noPA']:.4f}")
    print(f"    F-score     : {micro['F-score_noPA']:.4f}")
    print(f"    ACC         : {micro['ACC_noPA']:.4f}")
    print(f"    (TP={micro['TP_noPA']}, TN={micro['TN_noPA']}, FP={micro['FP_noPA']}, FN={micro['FN_noPA']})")

    print(f"\n{'─' * 70}")
    print(f"  WITH Point Adjustment")
    print(f"{'─' * 70}")

    print(f"\n  [MACRO] (mean ± std)")
    for metric in ['Precision', 'Recall', 'F-score', 'ACC']:
        mean_val = macro.get(f'{metric}_PA_mean', 0)
        std_val = macro.get(f'{metric}_PA_std', 0)
        print(f"    {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
    for metric in ['Aff-Pre', 'Aff-Rec', 'Aff-F1']:
        mean_val = macro.get(f'{metric}_PA_mean', 0)
        std_val = macro.get(f'{metric}_PA_std', 0)
        print(f"    {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\n  [MICRO] (aggregate TP/FP/FN/TN)")
    print(f"    Precision   : {micro['Precision_PA']:.4f}")
    print(f"    Recall      : {micro['Recall_PA']:.4f}")
    print(f"    F-score     : {micro['F-score_PA']:.4f}")
    print(f"    ACC         : {micro['ACC_PA']:.4f}")
    print(f"    (TP={micro['TP_PA']}, TN={micro['TN_PA']}, FP={micro['FP_PA']}, FN={micro['FN_PA']})")

    print(f"\n{'─' * 70}")
    print(f"  SCORE-BASED (PA not applicable)")
    print(f"{'─' * 70}")
    print(f"  [MACRO] (mean ± std)")
    for metric in ['AUC-ROC', 'AUC-PR', 'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 'F1_AUC']:
        mean_val = macro.get(f'{metric}_mean', 0)
        std_val = macro.get(f'{metric}_std', 0)
        print(f"    {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
    print(f"{'=' * 70}\n")


# ============================================================
# 查找结果文件
# ============================================================
def find_all_mode_files(ds_name):
    """查找 All 模式的结果文件"""
    possible_paths = [
        (f"results/{ds_name}/All/classification/classification_trainprobs.csv",
         f"results/{ds_name}/All/classification/classification_testprobs.csv"),
        (f"results/{ds_name}/classification/classification_trainprobs.csv",
         f"results/{ds_name}/classification/classification_testprobs.csv"),
        (f"results/{ds_name}_All/classification/classification_trainprobs.csv",
         f"results/{ds_name}_All/classification/classification_testprobs.csv"),
    ]

    for train_path, test_path in possible_paths:
        if os.path.exists(train_path) and os.path.exists(test_path):
            return train_path, test_path

    return None, None


def detect_result_structure(ds_name):
    """检测结果目录的结构"""
    path = os.path.join('results/', ds_name)

    if not os.path.exists(path):
        return 'none'

    train_all, test_all = find_all_mode_files(ds_name)
    has_all = train_all is not None

    has_single = False
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path) and item not in ['All', 'classification']:
                class_path = os.path.join(item_path, 'classification')
                if os.path.exists(class_path):
                    has_single = True
                    break

    if has_all and has_single:
        return 'both'
    elif has_all:
        return 'all_only'
    elif has_single:
        return 'single_only'
    else:
        return 'none'


# ============================================================
# 主评估函数
# ============================================================
def evaluate_dataset(ds_name, slidingWindow=100, mode='auto', fname='All'):
    """评估数据集（同时输出 PA 和 noPA 结果）"""
    print("=" * 70)
    print(f"  Dataset: {ds_name}")
    print(f"  Fname: {fname}")
    print(f"  Sliding Window: {slidingWindow}")
    print(f"  Mode: {mode}")
    print("=" * 70)

    path = os.path.join('results/', ds_name)

    if mode == 'auto':
        structure = detect_result_structure(ds_name)
        print(f"  Detected structure: {structure}")

        if fname == 'All':
            if structure in ['all_only', 'both']:
                mode = 'all'
            else:
                print(f"  Warning: fname='All' but no All mode results found, falling back to single mode")
                mode = 'single'
        else:
            mode = 'single'

        print(f"  Selected mode: {mode}")

    # All 模式
    if mode == 'all':
        print("\n[All Mode] Evaluating aggregated multi-channel result...")

        train_path, test_path = find_all_mode_files(ds_name)

        if train_path is None:
            print(f"Error: Cannot find All mode result files for {ds_name}")
            return None, None

        print(f"  Found train file: {train_path}")
        print(f"  Found test file: {test_path}")

        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
        except Exception as e:
            print(f"Error reading files: {e}")
            return None, None

        result = evaluate_single_channel(df_train, df_test, slidingWindow)
        result['name'] = 'All'

        print_single_result(result, f"{ds_name} (All channels aggregated)")

        res_df = pd.DataFrame([result])
        output_file = os.path.join(path, f'{ds_name}_All_evaluation.csv')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        res_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

        return res_df, result

    # Single 模式
    else:
        print("\n[Single Mode] Evaluating each channel separately...")

        res_list = []

        if fname != 'All':
            file_list = [fname]
        else:
            file_list = sorted(os.listdir(path))

        for filename in file_list:
            full_path = os.path.join(path, filename)
            if not os.path.isdir(full_path):
                continue
            if filename in ['All', 'GECCO', '.json', 'classification']:
                continue

            try:
                df_train = pd.read_csv(f"{full_path}/classification/classification_trainprobs.csv")
                df_test = pd.read_csv(f"{full_path}/classification/classification_testprobs.csv")
            except FileNotFoundError:
                continue

            try:
                result = evaluate_single_channel(df_train, df_test, slidingWindow)
                result['name'] = filename
                print(f"  {filename}: F1(PA)={result['F-score_PA']:.4f}, F1(noPA)={result['F-score_noPA']:.4f}")
            except Exception as e:
                print(f"  {filename}: ERROR - {e}")
                continue

            res_list.append(result)

        if not res_list:
            print("\nNo valid results found!")
            return None, None

        res_df = pd.DataFrame(res_list)
        n_channels = len(res_df)

        macro = compute_macro_metrics(res_df)
        micro = compute_micro_metrics(res_df)

        print_multi_channel_summary(macro, micro, n_channels)

        detail_file = os.path.join(path, f'{ds_name}_channels_evaluation.csv')
        res_df.to_csv(detail_file, index=False)
        print(f"Channel details saved to {detail_file}")

        summary_data = {
            'Aggregation': ['Macro_noPA', 'Micro_noPA', 'Macro_PA', 'Micro_PA'],
            'Precision': [
                macro['Precision_noPA_mean'], micro['Precision_noPA'],
                macro['Precision_PA_mean'], micro['Precision_PA']
            ],
            'Recall': [
                macro['Recall_noPA_mean'], micro['Recall_noPA'],
                macro['Recall_PA_mean'], micro['Recall_PA']
            ],
            'F-score': [
                macro['F-score_noPA_mean'], micro['F-score_noPA'],
                macro['F-score_PA_mean'], micro['F-score_PA']
            ],
            'ACC': [
                macro['ACC_noPA_mean'], micro['ACC_noPA'],
                macro['ACC_PA_mean'], micro['ACC_PA']
            ],
        }

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(path, f'{ds_name}_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")

        return res_df, {'macro': macro, 'micro': micro}


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anomaly Detection Evaluation')
    parser.add_argument('--dataset', type=str, default='MSL', help='Dataset name')
    parser.add_argument('--fname', type=str, default='All', help='File name or "All"')
    parser.add_argument('--sliding_window', type=int, default=100, help='Sliding window for VUS/R_AUC')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'single', 'all'],
                        help='Evaluation mode')

    args = parser.parse_args()

    evaluate_dataset(
        ds_name=args.dataset,
        slidingWindow=args.sliding_window,
        mode=args.mode,
        fname=args.fname
    )