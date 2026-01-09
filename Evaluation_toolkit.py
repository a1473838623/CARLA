import numpy as np
import pandas as pd
import os

os.environ['NUMBA_DISABLE_CUDA'] = '1'
import argparse
import ast

from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, auc
)
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.metrics import get_range_vus_roc

# ============================================================
# 数据集配置
# ============================================================
DATASET_CONFIG = {
    'MSL': {
        'window_size': 200,
        'stride': 1,
        'label_type': 'npy',
        'label_file': './datasets/MSL_SMAP/MSL_test_label.npy',
    },
    'SMAP': {
        'window_size': 200,
        'stride': 1,
        'label_type': 'npy',
        'label_file': './datasets/MSL_SMAP/SMAP_test_label.npy',
    },
    'smd': {
        'window_size': 200,
        'stride': 5,
        'label_type': 'npy',
        'label_file': './datasets/SMD/SMD_test_label.npy',
    },
    'psm': {
        'window_size': 100,
        'stride': 10,
        'label_type': 'csv_column',
        'label_file': './datasets/PSM/test_label.csv',
        'label_column': 1,
    },
    'swat': {
        'window_size': 200,
        'stride': 10,
        'label_type': 'csv_embedded',
        'label_file': './datasets/SWAT/All_test.csv',
        'label_column': 'Normal/Attack',
    },
    'nips_ts_swan': {
        'window_size': 100,
        'stride': 10,
        'label_type': 'npy',
        'label_file': './datasets/NIPS_TS_Swan/NIPS_TS_Swan_test_label.npy',
    },
    'nips_ts_water': {
        'window_size': 100,
        'stride': 10,
        'label_type': 'npy',
        'label_file': './datasets/NIPS_TS_Water/NIPS_TS_Water_test_label.npy',
    },
}


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
# 阈值搜索函数（核心修正）
# ============================================================
def find_best_threshold(scores, labels, apply_pa=False):
    """
    搜索最佳阈值

    Args:
        scores: 异常分数
        labels: 真实标签
        apply_pa: 是否在搜索过程中应用 PA

    Returns:
        best_threshold: 最佳阈值
    """
    gt = labels.astype(int)

    # 使用百分位数作为候选阈值（90% - 99.9%）
    percentiles = [(90 + (i / 10)) for i in range(100)]
    thresholds = np.percentile(scores, percentiles)

    best_f1 = -1
    best_threshold = thresholds[0]

    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)

        if apply_pa:
            pred = point_adjustment(gt, pred)

        _, _, f1, _ = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold


# ============================================================
# 窗口级别 -> 点级别 转换函数
# ============================================================
def window_to_point_scores(window_scores, window_size, stride, total_points, method='mean'):
    """
    将窗口级别的预测分数转换为点级别的预测分数
    """
    n_windows = len(window_scores)

    if method == 'last':
        point_scores = np.zeros(total_points)
        counts = np.zeros(total_points)

        for i in range(n_windows):
            end_point = i * stride + window_size - 1
            if end_point < total_points:
                point_scores[end_point] = window_scores[i]
                counts[end_point] = 1

        for i in range(total_points):
            if counts[i] == 0:
                for j in range(i + 1, total_points):
                    if counts[j] > 0:
                        point_scores[i] = point_scores[j]
                        break

        return point_scores

    elif method == 'first':
        point_scores = np.zeros(total_points)
        counts = np.zeros(total_points)

        for i in range(n_windows):
            start_point = i * stride
            if start_point < total_points:
                point_scores[start_point] = window_scores[i]
                counts[start_point] = 1

        for i in range(total_points - 1, -1, -1):
            if counts[i] == 0:
                for j in range(i - 1, -1, -1):
                    if counts[j] > 0:
                        point_scores[i] = point_scores[j]
                        break

        return point_scores

    elif method == 'center':
        point_scores = np.zeros(total_points)
        counts = np.zeros(total_points)

        for i in range(n_windows):
            center_point = i * stride + window_size // 2
            if center_point < total_points:
                point_scores[center_point] += window_scores[i]
                counts[center_point] += 1

        mask = counts > 0
        point_scores[mask] /= counts[mask]

        first_valid = np.argmax(counts > 0)
        last_valid = total_points - 1 - np.argmax(counts[::-1] > 0)

        for i in range(first_valid):
            point_scores[i] = point_scores[first_valid]
        for i in range(last_valid + 1, total_points):
            point_scores[i] = point_scores[last_valid]

        return point_scores

    else:  # 'mean' or 'max'
        if method == 'mean':
            point_scores = np.zeros(total_points)
            counts = np.zeros(total_points)

            for i in range(n_windows):
                start = i * stride
                end = min(start + window_size, total_points)
                point_scores[start:end] += window_scores[i]
                counts[start:end] += 1

            counts[counts == 0] = 1
            point_scores /= counts

        else:  # 'max'
            point_scores = np.full(total_points, -np.inf)

            for i in range(n_windows):
                start = i * stride
                end = min(start + window_size, total_points)
                point_scores[start:end] = np.maximum(
                    point_scores[start:end],
                    window_scores[i]
                )

            point_scores[point_scores == -np.inf] = 0

        return point_scores


# ============================================================
# 加载点级别真实标签
# ============================================================
def load_point_labels(ds_name, fname='All', data_root='data'):
    """从原始数据集加载点级别的真实标签"""
    config = DATASET_CONFIG.get(ds_name)
    if config is None:
        raise ValueError(f"Unknown dataset: {ds_name}. Available: {list(DATASET_CONFIG.keys())}")

    label_type = config['label_type']

    if label_type == 'npy':
        label_path = config['label_file']
        if os.path.exists(label_path):
            labels = np.load(label_path)
            return labels.astype(int), len(labels)
        raise FileNotFoundError(f"Label file not found: {label_path}")

    elif label_type == 'csv_column':
        label_path = config['label_file']
        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
            col_idx = config.get('label_column', 1)
            if isinstance(col_idx, int):
                labels = df.iloc[:, col_idx].values
            else:
                labels = df[col_idx].values
            labels = np.nan_to_num(labels).flatten().astype(int)
            return labels, len(labels)
        raise FileNotFoundError(f"Label file not found: {label_path}")

    elif label_type == 'csv_embedded':
        label_path = config['label_file']
        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
            label_col = config.get('label_column', 'Normal/Attack')
            labels = df[label_col].values
            if labels.dtype == object:
                labels = np.where(labels == 'Normal', 0, 1)
            return labels.astype(int), len(labels)
        raise FileNotFoundError(f"Label file not found: {label_path}")

    else:
        raise ValueError(f"Unknown label_type: {label_type}")


# ============================================================
# slidingWindow 辅助函数
# ============================================================
def get_default_sliding_window(y_true):
    """根据异常段的平均长度自动计算 slidingWindow"""
    events = convert_vector_to_events(y_true)
    if len(events) == 0:
        return 100
    avg_length = np.mean([e[1] - e[0] for e in events])
    return max(10, int(avg_length))


# ============================================================
# 指标计算函数
# ============================================================
def compute_label_based_metrics(y_true, y_pred):
    """计算基于标签的指标"""
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


def compute_affiliation_metrics(y_true, y_pred):
    """计算 Affiliation 指标"""
    events_pred = convert_vector_to_events(y_pred)
    events_gt = convert_vector_to_events(y_true)
    Trange = (0, len(y_true))

    if len(events_pred) == 0 or len(events_gt) == 0:
        return {'Aff-Pre': 0.0, 'Aff-Rec': 0.0, 'Aff-F1': 0.0}

    affiliation = pr_from_events(events_pred, events_gt, Trange)

    aff_pre = affiliation['precision']
    aff_rec = affiliation['recall']
    aff_f1 = 2 * aff_pre * aff_rec / (aff_pre + aff_rec) if (aff_pre + aff_rec) > 0 else 0

    return {'Aff-Pre': aff_pre, 'Aff-Rec': aff_rec, 'Aff-F1': aff_f1}


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


def compute_score_based_metrics(y_true, scores, slidingWindow=None):
    """计算基于分数的指标（不受 PA 影响）"""
    result = {}

    if slidingWindow is None:
        slidingWindow = get_default_sliding_window(y_true)

    try:
        result['AUC-ROC'] = roc_auc_score(y_true, scores)
        result['AUC-PR'] = average_precision_score(y_true, scores)
    except ValueError as e:
        print(f"  Warning: AUC calculation failed: {e}")
        result['AUC-ROC'] = 0
        result['AUC-PR'] = 0

    try:
        result['F1_AUC'] = compute_f1_auc(y_true, scores)
    except Exception as e:
        print(f"  Warning: F1_AUC calculation failed: {e}")
        result['F1_AUC'] = 0

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

    result['slidingWindow'] = slidingWindow
    return result


# ============================================================
# 核心评估函数（同时计算 PA 和非 PA）
# ============================================================
def evaluate_all_metrics(scores, labels, use_pa=None, slidingWindow=None):
    """
    完整评估：计算 PA 和/或非 PA 的所有指标

    Args:
        scores: 异常分数
        labels: 真实标签
        use_pa: 是否使用 PA
            - None: 同时输出 PA 和非 PA 结果
            - True: 只输出 PA 结果
            - False: 只输出非 PA 结果
        slidingWindow: 滑动窗口大小，None 则自动计算

    Returns:
        dict: 根据 use_pa 返回不同结构
    """
    gt = labels.astype(int)

    results = {}

    # ========== 1. 使用 PA 的评估（独立搜索阈值）==========
    if use_pa is None or use_pa is True:
        thresh_pa = find_best_threshold(scores, gt, apply_pa=True)
        pred_pa = (scores >= thresh_pa).astype(int)
        pred_pa = point_adjustment(gt, pred_pa)

        results['with_pa'] = {
            'threshold': thresh_pa,
            'label_based': compute_label_based_metrics(gt, pred_pa),
            'affiliation': compute_affiliation_metrics(gt, pred_pa)
        }

    # ========== 2. 不使用 PA 的评估（独立搜索阈值）==========
    if use_pa is None or use_pa is False:
        thresh_no_pa = find_best_threshold(scores, gt, apply_pa=False)
        pred_no_pa = (scores >= thresh_no_pa).astype(int)

        results['without_pa'] = {
            'threshold': thresh_no_pa,
            'label_based': compute_label_based_metrics(gt, pred_no_pa),
            'affiliation': compute_affiliation_metrics(gt, pred_no_pa)
        }

    # ========== 3. Score-based 指标（不依赖阈值和 PA）==========
    results['score_based'] = compute_score_based_metrics(gt, scores, slidingWindow)

    return results


# ============================================================
# 点级别评估
# ============================================================
def evaluate_point_level(window_scores, point_labels, window_size, stride,
                         use_pa=None, slidingWindow=None, score_method='mean'):
    """
    点级别评估

    Args:
        use_pa: None=同时输出两者, True=只PA, False=只非PA
    """
    total_points = len(point_labels)

    # 转换分数到点级别
    point_scores = window_to_point_scores(
        window_scores, window_size, stride, total_points, method=score_method
    )

    # 使用统一的评估函数
    results = evaluate_all_metrics(point_scores, point_labels, use_pa, slidingWindow)
    results['score_method'] = score_method

    return results


# ============================================================
# 窗口级别评估
# ============================================================
def evaluate_window_level(window_scores, window_labels, use_pa=None, slidingWindow=None):
    """
    窗口级别评估

    Args:
        use_pa: None=同时输出两者, True=只PA, False=只非PA
    """
    return evaluate_all_metrics(window_scores, window_labels, use_pa, slidingWindow)


# ============================================================
# 打印函数
# ============================================================
def print_all_results(results, name, eval_level, use_pa=None):
    """格式化打印所有评估结果"""
    print(f"\n{'=' * 70}")
    print(f"  Dataset: {name}")
    print(f"  Evaluation Level: {eval_level.upper()}")
    if 'score_method' in results:
        print(f"  Score Method: {results['score_method']}")
    print(f"{'=' * 70}")

    # WITH PA
    if 'with_pa' in results:
        print(f"\n{'─' * 70}")
        print("  WITH PA (Point Adjustment)")
        print(f"{'─' * 70}")
        print(f"  Threshold: {results['with_pa']['threshold']:.6f}")

        print("\n  [Label-based metrics]")
        lm = results['with_pa']['label_based']
        print(f"    Precision : {lm['Precision']:.4f}")
        print(f"    Recall    : {lm['Recall']:.4f}")
        print(f"    F-score   : {lm['F-score']:.4f}")
        print(f"    ACC       : {lm['ACC']:.4f}")
        print(f"    (TP={lm['TP']}, TN={lm['TN']}, FP={lm['FP']}, FN={lm['FN']})")

        print("\n  [Affiliation metrics]")
        am = results['with_pa']['affiliation']
        print(f"    Aff-Pre   : {am['Aff-Pre']:.4f}")
        print(f"    Aff-Rec   : {am['Aff-Rec']:.4f}")
        print(f"    Aff-F1    : {am['Aff-F1']:.4f}")

    # WITHOUT PA
    if 'without_pa' in results:
        print(f"\n{'─' * 70}")
        print("  WITHOUT PA (Point Adjustment)")
        print(f"{'─' * 70}")
        print(f"  Threshold: {results['without_pa']['threshold']:.6f}")

        print("\n  [Label-based metrics]")
        lm = results['without_pa']['label_based']
        print(f"    Precision : {lm['Precision']:.4f}")
        print(f"    Recall    : {lm['Recall']:.4f}")
        print(f"    F-score   : {lm['F-score']:.4f}")
        print(f"    ACC       : {lm['ACC']:.4f}")
        print(f"    (TP={lm['TP']}, TN={lm['TN']}, FP={lm['FP']}, FN={lm['FN']})")

        print("\n  [Affiliation metrics]")
        am = results['without_pa']['affiliation']
        print(f"    Aff-Pre   : {am['Aff-Pre']:.4f}")
        print(f"    Aff-Rec   : {am['Aff-Rec']:.4f}")
        print(f"    Aff-F1    : {am['Aff-F1']:.4f}")

    # SCORE-BASED
    print(f"\n{'─' * 70}")
    print("  SCORE-BASED METRICS (PA not applicable)")
    print(f"{'─' * 70}")
    sm = results['score_based']
    print(f"    slidingWindow : {sm.get('slidingWindow', 'N/A')}")
    print(f"    AUC-ROC       : {sm['AUC-ROC']:.4f}")
    print(f"    AUC-PR        : {sm['AUC-PR']:.4f}")
    print(f"    R_AUC_ROC     : {sm['R_AUC_ROC']:.4f}")
    print(f"    R_AUC_PR      : {sm['R_AUC_PR']:.4f}")
    print(f"    VUS_ROC       : {sm['VUS_ROC']:.4f}")
    print(f"    VUS_PR        : {sm['VUS_PR']:.4f}")
    print(f"    F1_AUC        : {sm['F1_AUC']:.4f}")
    print(f"{'=' * 70}\n")


# ============================================================
# 结果转换为 DataFrame
# ============================================================
def results_to_dataframe(results, name, eval_level):
    """将结果转换为 DataFrame 格式"""
    row = {'name': name, 'eval_level': eval_level}

    # PA 结果
    if 'with_pa' in results:
        row['PA_threshold'] = results['with_pa']['threshold']
        for k, v in results['with_pa']['label_based'].items():
            row[f'PA_{k}'] = v
        for k, v in results['with_pa']['affiliation'].items():
            row[f'PA_{k}'] = v

    # 非 PA 结果
    if 'without_pa' in results:
        row['nPA_threshold'] = results['without_pa']['threshold']
        for k, v in results['without_pa']['label_based'].items():
            row[f'nPA_{k}'] = v
        for k, v in results['without_pa']['affiliation'].items():
            row[f'nPA_{k}'] = v

    # Score-based 结果
    for k, v in results['score_based'].items():
        row[k] = v

    return pd.DataFrame([row])


# ============================================================
# 主评估函数
# ============================================================
def evaluate_dataset(
        ds_name,
        fname='All',
        use_pa=None,
        slidingWindow=None,
        eval_level='point',
        score_method='mean',
        data_root='data',
        result_root='results'
):
    """
    评估数据集

    Args:
        ds_name: 数据集名称
        fname: 文件名
        use_pa: 是否使用 PA
            - None: 同时输出 PA 和非 PA 结果
            - True: 只输出 PA 结果
            - False: 只输出非 PA 结果
        slidingWindow: VUS/R_AUC 的滑动窗口大小，None 则自动计算
        eval_level: 评估级别 ('point' 或 'window')
        score_method: 分数转换方法
        data_root: 数据根目录
        result_root: 结果根目录
    """
    # 获取数据集配置
    config = DATASET_CONFIG.get(ds_name)
    if config is None:
        print(f"Warning: Unknown dataset {ds_name}, using default config")
        config = {'window_size': 200, 'stride': 1}

    window_size = config['window_size']
    stride = config['stride']

    print("=" * 70)
    print(f"Dataset: {ds_name}")
    print(f"Fname: {fname}")
    print(f"Evaluation Level: {eval_level}")
    print(f"Window Size: {window_size}, Stride: {stride}")
    print(f"Score Method: {score_method}")
    print("=" * 70)

    # 查找结果文件
    possible_paths = [
        os.path.join(result_root, ds_name, fname, 'classification'),
        os.path.join(result_root, ds_name, 'classification'),
        os.path.join(result_root, f'{ds_name}_{fname}', 'classification'),
    ]

    result_path = None
    for p in possible_paths:
        train_file = os.path.join(p, 'classification_trainprobs.csv')
        test_file = os.path.join(p, 'classification_testprobs.csv')
        if os.path.exists(train_file) and os.path.exists(test_file):
            result_path = p
            break

    if result_path is None:
        print(f"Error: Cannot find result files for {ds_name}/{fname}")
        print("Searched paths:")
        for p in possible_paths:
            print(f"  - {p}")
        return None

    print(f"Found results at: {result_path}")

    # 加载预测结果
    df_train = pd.read_csv(os.path.join(result_path, 'classification_trainprobs.csv'))
    df_test = pd.read_csv(os.path.join(result_path, 'classification_testprobs.csv'))

    print(f"Train windows: {len(df_train)}, Test windows: {len(df_test)}")

    # 提取窗口分数
    cl_num = df_train.shape[1] - 1
    df_train['pred'] = df_train[df_train.columns[0:cl_num]].idxmax(axis=1)
    score_col = df_train['pred'].value_counts().idxmax()

    window_scores = (1 - df_test[score_col]).values
    window_labels = np.where(df_test['Class'] == 0, 0, 1)

    # 根据评估级别进行评估
    if eval_level == 'point':
        try:
            point_labels, total_points = load_point_labels(ds_name, fname, data_root)
            print(
                f"Loaded point labels: {total_points} points, {point_labels.sum()} anomalies ({100 * point_labels.mean():.2f}%)")

            results = evaluate_point_level(
                window_scores=window_scores,
                point_labels=point_labels,
                window_size=window_size,
                stride=stride,
                use_pa=use_pa,
                slidingWindow=slidingWindow,
                score_method=score_method
            )
        except Exception as e:
            print(f"Warning: Could not load point labels: {e}")
            print("Falling back to window-level evaluation")
            eval_level = 'window'
            results = evaluate_window_level(
                window_scores=window_scores,
                window_labels=window_labels,
                use_pa=use_pa,
                slidingWindow=slidingWindow
            )
    else:
        results = evaluate_window_level(
            window_scores=window_scores,
            window_labels=window_labels,
            use_pa=use_pa,
            slidingWindow=slidingWindow
        )

    # 打印结果
    print_all_results(results, f"{ds_name}/{fname}", eval_level, use_pa)

    # 保存结果
    res_df = results_to_dataframe(results, fname, eval_level)
    output_dir = os.path.join(result_root, ds_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{ds_name}_{fname}_{eval_level}_evaluation.csv')
    res_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    return results


# ============================================================
# 批量评估
# ============================================================
def evaluate_all_datasets(
        datasets=None,
        eval_level='point',
        score_method='mean',
        use_pa=None,
        slidingWindow=None,
        data_root='data',
        result_root='results'
):
    """批量评估多个数据集"""
    if datasets is None:
        datasets = list(DATASET_CONFIG.keys())

    all_results = []

    for ds_name in datasets:
        print(f"\n{'#' * 70}")
        print(f"# Evaluating: {ds_name}")
        print(f"{'#' * 70}")

        try:
            result = evaluate_dataset(
                ds_name=ds_name,
                fname='All',
                use_pa=use_pa,
                slidingWindow=slidingWindow,
                eval_level=eval_level,
                score_method=score_method,
                data_root=data_root,
                result_root=result_root
            )
            if result:
                result['dataset'] = ds_name
                all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总表格
    if all_results:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        summary_rows = []
        for r in all_results:
            row = {'dataset': r['dataset']}
            if 'with_pa' in r:
                row['PA_F-score'] = r['with_pa']['label_based']['F-score']
                row['PA_Aff-F1'] = r['with_pa']['affiliation']['Aff-F1']
            if 'without_pa' in r:
                row['nPA_F-score'] = r['without_pa']['label_based']['F-score']
                row['nPA_Aff-F1'] = r['without_pa']['affiliation']['Aff-F1']
            row['AUC-ROC'] = r['score_based']['AUC-ROC']
            row['AUC-PR'] = r['score_based']['AUC-PR']
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))

        pa_suffix = '' if use_pa is None else ('_PA' if use_pa else '_noPA')
        summary_file = os.path.join(result_root, f'all_datasets_{eval_level}_summary{pa_suffix}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to {summary_file}")

    return all_results


# ============================================================
# 对比不同转换方法
# ============================================================
def compare_methods(
        ds_name,
        fname='All',
        use_pa=None,
        slidingWindow=None,
        data_root='data',
        result_root='results'
):
    """对比不同的窗口到点转换方法"""

    methods = ['mean', 'max', 'last', 'center']
    all_results = []

    for method in methods:
        print(f"\n{'#' * 70}")
        print(f"# Method: {method}")
        print(f"{'#' * 70}")

        result = evaluate_dataset(
            ds_name=ds_name,
            fname=fname,
            use_pa=use_pa,
            slidingWindow=slidingWindow,
            eval_level='point',
            score_method=method,
            data_root=data_root,
            result_root=result_root
        )

        if result:
            result['method'] = method
            all_results.append(result)

    # 打印对比表格
    if all_results:
        print("\n" + "=" * 100)
        print("Method Comparison")
        print("=" * 100)

        # 根据 use_pa 决定打印哪些列
        if use_pa is None:
            print(
                f"{'Method':<10} {'PA_F1':>10} {'nPA_F1':>10} {'PA_Aff-F1':>12} {'nPA_Aff-F1':>12} {'AUC-ROC':>10} {'AUC-PR':>10}")
            print("-" * 100)
            for r in all_results:
                print(f"{r['method']:<10} "
                      f"{r['with_pa']['label_based']['F-score']:>10.4f} "
                      f"{r['without_pa']['label_based']['F-score']:>10.4f} "
                      f"{r['with_pa']['affiliation']['Aff-F1']:>12.4f} "
                      f"{r['without_pa']['affiliation']['Aff-F1']:>12.4f} "
                      f"{r['score_based']['AUC-ROC']:>10.4f} "
                      f"{r['score_based']['AUC-PR']:>10.4f}")
        elif use_pa:
            print(f"{'Method':<10} {'PA_F1':>10} {'PA_Aff-F1':>12} {'AUC-ROC':>10} {'AUC-PR':>10}")
            print("-" * 60)
            for r in all_results:
                print(f"{r['method']:<10} "
                      f"{r['with_pa']['label_based']['F-score']:>10.4f} "
                      f"{r['with_pa']['affiliation']['Aff-F1']:>12.4f} "
                      f"{r['score_based']['AUC-ROC']:>10.4f} "
                      f"{r['score_based']['AUC-PR']:>10.4f}")
        else:
            print(f"{'Method':<10} {'nPA_F1':>10} {'nPA_Aff-F1':>12} {'AUC-ROC':>10} {'AUC-PR':>10}")
            print("-" * 60)
            for r in all_results:
                print(f"{r['method']:<10} "
                      f"{r['without_pa']['label_based']['F-score']:>10.4f} "
                      f"{r['without_pa']['affiliation']['Aff-F1']:>12.4f} "
                      f"{r['score_based']['AUC-ROC']:>10.4f} "
                      f"{r['score_based']['AUC-PR']:>10.4f}")

    return all_results


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified Anomaly Detection Evaluation')
    parser.add_argument('--dataset', type=str, default='MSL',
                        help='Dataset name')
    parser.add_argument('--fname', type=str, default='All',
                        help='File name')
    parser.add_argument('--use_pa', type=str, default='both',
                        choices=['true', 'false', 'both', None],
                        help='Use Point Adjustment: true, false, or both (default)')
    parser.add_argument('--sliding_window', type=int, default=None,
                        help='Sliding window for VUS/R_AUC (None for auto)')
    parser.add_argument('--eval_level', type=str, default='window',
                        choices=['point', 'window'],
                        help='Evaluation level')
    parser.add_argument('--score_method', type=str, default='mean',
                        choices=['mean', 'max', 'last', 'first', 'center'],
                        help='Method for converting window scores to point scores')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Data root directory')
    parser.add_argument('--result_root', type=str, default='results',
                        help='Results root directory')
    parser.add_argument('--compare_methods', action='store_true',
                        help='Compare different conversion methods')
    parser.add_argument('--eval_all', action='store_true',
                        help='Evaluate all datasets')

    args = parser.parse_args()

    # 解析 use_pa 参数
    if args.use_pa is None or args.use_pa == 'both':
        use_pa = None  # 同时输出 PA 和非 PA
    elif args.use_pa == 'true':
        use_pa = True
    else:
        use_pa = False

    if args.eval_all:
        evaluate_all_datasets(
            eval_level=args.eval_level,
            score_method=args.score_method,
            use_pa=use_pa,
            slidingWindow=args.sliding_window,
            data_root=args.data_root,
            result_root=args.result_root
        )
    elif args.compare_methods:
        compare_methods(
            ds_name=args.dataset,
            fname=args.fname,
            use_pa=use_pa,
            slidingWindow=args.sliding_window,
            data_root=args.data_root,
            result_root=args.result_root
        )
    else:
        evaluate_dataset(
            ds_name=args.dataset,
            fname=args.fname,
            use_pa=use_pa,
            slidingWindow=args.sliding_window,
            eval_level=args.eval_level,
            score_method=args.score_method,
            data_root=args.data_root,
            result_root=args.result_root
        )