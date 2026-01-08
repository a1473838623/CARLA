import numpy as np
import pandas as pd
import os

os.environ['NUMBA_DISABLE_CUDA'] = '1'
import argparse
import ast

from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.metrics import get_range_vus_roc

from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, auc
)

# ============================================================
# 数据集配置
# ============================================================
DATASET_CONFIG = {
    'MSL': {
        'window_size': 200,
        'stride': 1,
        'label_type': 'npy',  # MSL_test_label.npy
        'label_file': '{dataset}_test_label.npy',
        'data_subdir': '',
    },
    'SMAP': {
        'window_size': 200,
        'stride': 1,
        'label_type': 'npy',
        'label_file': '{dataset}_test_label.npy',
        'data_subdir': '',
    },
    'SMD': {
        'window_size': 200,
        'stride': 5,
        'label_type': 'npy',  # SMD_test_label.npy for All mode
        'label_file': '{dataset}_test_label.npy',
        'data_subdir': '',
    },
    'PSM': {
        'window_size': 100,
        'stride': 10,
        'label_type': 'csv_column',
        'label_file': 'test_label.csv',
        'label_column': 1,  # 第二列（跳过时间戳）
        'data_subdir': '',
    },
    'SWAT': {
        'window_size': 200,
        'stride': 10,
        'label_type': 'csv_embedded',  # 标签嵌入在测试数据文件中
        'label_file': '{fname}_test.csv',
        'label_column': 'Normal/Attack',
        'data_subdir': '',
    },
    'NIPS_TS_Swan': {
        'window_size': 100,
        'stride': 10,
        'label_type': 'npy',
        'label_file': 'NIPS_TS_Swan_test_label.npy',
        'data_subdir': '',
    },
    'NIPS_TS_Water': {
        'window_size': 100,
        'stride': 10,
        'label_type': 'npy',
        'label_file': 'NIPS_TS_Water_test_label.npy',
        'data_subdir': '',
    },
}


# ============================================================
# 窗口级别 -> 点级别 转换函数
# ============================================================
def window_to_point_scores(window_scores, window_size, stride, total_points, method='mean'):
    """
    将窗口级别的预测分数转换为点级别的预测分数

    Args:
        window_scores: 每个窗口的预测分数，shape=(n_windows,)
        window_size: 滑动窗口大小
        stride: 滑动步长
        total_points: 原始时间序列的总点数
        method: 转换方法 ('mean', 'max', 'last', 'first', 'center')

    Returns:
        point_scores: 每个点的预测分数，shape=(total_points,)
    """
    n_windows = len(window_scores)

    if method == 'last':
        # 每个窗口的预测分配给窗口的最后一个点
        point_scores = np.zeros(total_points)
        counts = np.zeros(total_points)

        for i in range(n_windows):
            end_point = i * stride + window_size - 1
            if end_point < total_points:
                point_scores[end_point] = window_scores[i]
                counts[end_point] = 1

        # 填充没有覆盖到的点
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

        # 填充边界点
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
# 加载点级别真实标签（支持多种数据集格式）
# ============================================================
def load_point_labels(ds_name, fname='All', data_root='data'):
    """
    从原始数据集加载点级别的真实标签

    Args:
        ds_name: 数据集名称
        fname: 文件名
        data_root: 数据根目录

    Returns:
        point_labels: 点级别的真实标签
        total_points: 总点数
    """
    config = DATASET_CONFIG.get(ds_name)
    if config is None:
        raise ValueError(f"Unknown dataset: {ds_name}. Available: {list(DATASET_CONFIG.keys())}")

    # 确定数据目录
    ds_dir = os.path.join(data_root, ds_name.lower())
    if not os.path.exists(ds_dir):
        # 尝试其他可能的目录名
        alt_dirs = [
            os.path.join(data_root, ds_name),
            os.path.join(data_root, ds_name.upper()),
            os.path.join(data_root, ds_name.replace('_', '')),
        ]
        for alt in alt_dirs:
            if os.path.exists(alt):
                ds_dir = alt
                break

    label_type = config['label_type']

    # ============================================================
    # NPY 格式：MSL, SMAP, SMD, NIPS_TS_Swan, NIPS_TS_Water
    # ============================================================
    if label_type == 'npy':
        label_file = config['label_file'].format(dataset=ds_name)
        label_path = os.path.join(ds_dir, label_file)

        if os.path.exists(label_path):
            labels = np.load(label_path)
            return labels.astype(int), len(labels)

        # 备选：对于 MSL/SMAP，尝试从 labeled_anomalies.csv 解析
        if ds_name in ['MSL', 'SMAP'] and fname != 'All':
            csv_path = os.path.join(ds_dir, 'labeled_anomalies.csv')
            if os.path.exists(csv_path):
                csv_reader = pd.read_csv(csv_path)
                data_info = csv_reader[csv_reader['chan_id'] == fname]

                if len(data_info) == 0:
                    raise ValueError(f"Channel {fname} not found in {csv_path}")

                labels = []
                for index, row in data_info.iterrows():
                    anomalies = ast.literal_eval(row['anomaly_sequences'])
                    length = row.iloc[-1]
                    label = np.zeros([int(length)], dtype=int)
                    for anomaly in anomalies:
                        label[anomaly[0]:anomaly[1] + 1] = 1
                    labels.extend(label)

                return np.asarray(labels), len(labels)

        # 备选：对于 SMD，尝试从 test_label 目录加载
        if ds_name == 'SMD' and fname != 'All':
            label_path = os.path.join(ds_dir, 'test_label', fname)
            if os.path.exists(label_path):
                labels = pd.read_csv(label_path, header=None).values.flatten()
                return labels.astype(int), len(labels)

        raise FileNotFoundError(f"Label file not found: {label_path}")

    # ============================================================
    # CSV 列格式：PSM
    # ============================================================
    elif label_type == 'csv_column':
        label_file = config['label_file']
        label_path = os.path.join(ds_dir, label_file)

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

    # ============================================================
    # CSV 嵌入格式：SWAT
    # ============================================================
    elif label_type == 'csv_embedded':
        label_file = config['label_file'].format(fname=fname)
        label_path = os.path.join(ds_dir, label_file)

        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
            label_col = config.get('label_column', 'Normal/Attack')
            labels = df[label_col].values
            # SWAT 的标签可能是字符串 'Normal'/'Attack' 或数字
            if labels.dtype == object:
                labels = np.where(labels == 'Normal', 0, 1)
            return labels.astype(int), len(labels)

        raise FileNotFoundError(f"Label file not found: {label_path}")

    else:
        raise ValueError(f"Unknown label_type: {label_type}")


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

    if len(events_pred) == 0 or len(events_gt) == 0:
        return 0.0, 0.0, 0.0

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
    """计算基于分数的指标"""
    result = {}

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
# 评估函数
# ============================================================
def evaluate_point_level(
        window_scores,
        point_labels,
        window_size,
        stride,
        use_pa=False,
        slidingWindow=100,
        score_method='mean'
):
    """点级别评估"""
    result = {}
    total_points = len(point_labels)

    # 转换分数到点级别
    point_scores = window_to_point_scores(
        window_scores, window_size, stride, total_points, method=score_method
    )

    # 找最佳阈值
    best_threshold = find_best_threshold(point_labels, point_scores)
    result['threshold'] = best_threshold

    # 生成点级别预测标签
    point_pred_raw = (point_scores >= best_threshold).astype(int)

    # 基于分数的指标
    score_metrics = compute_score_based_metrics(point_labels, point_scores, slidingWindow)
    result.update(score_metrics)

    # 基于标签的指标（可选 PA）
    if use_pa:
        point_pred = point_adjustment(point_labels, point_pred_raw)
    else:
        point_pred = point_pred_raw

    label_metrics = compute_label_based_metrics(point_labels, point_pred)
    result.update(label_metrics)

    # Affiliation 指标
    aff_pre, aff_rec, aff_f1 = compute_affiliation_metrics(point_labels, point_pred)
    result['Aff-Pre'] = aff_pre
    result['Aff-Rec'] = aff_rec
    result['Aff-F1'] = aff_f1

    return result


def evaluate_window_level(
        window_scores,
        window_labels,
        use_pa=False,
        slidingWindow=100
):
    """窗口级别评估（原有逻辑）"""
    result = {}

    best_threshold = find_best_threshold(window_labels, window_scores)
    result['threshold'] = best_threshold

    window_pred_raw = (window_scores >= best_threshold).astype(int)

    score_metrics = compute_score_based_metrics(window_labels, window_scores, slidingWindow)
    result.update(score_metrics)

    if use_pa:
        window_pred = point_adjustment(window_labels, window_pred_raw)
    else:
        window_pred = window_pred_raw

    label_metrics = compute_label_based_metrics(window_labels, window_pred)
    result.update(label_metrics)

    aff_pre, aff_rec, aff_f1 = compute_affiliation_metrics(window_labels, window_pred)
    result['Aff-Pre'] = aff_pre
    result['Aff-Rec'] = aff_rec
    result['Aff-F1'] = aff_f1

    return result


# ============================================================
# 主评估函数
# ============================================================
def evaluate_dataset(
        ds_name,
        fname='All',
        use_pa=False,
        slidingWindow=100,
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
        use_pa: 是否使用 Point Adjustment
        slidingWindow: VUS/R_AUC 的滑动窗口大小
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

    print("=" * 60)
    print(f"Dataset: {ds_name}")
    print(f"Fname: {fname}")
    print(f"Evaluation Level: {eval_level}")
    print(f"Window Size: {window_size}, Stride: {stride}")
    print(f"Score Method: {score_method}")
    print(f"Point Adjustment (PA): {'Enabled' if use_pa else 'Disabled'}")
    print("=" * 60)

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

    # 加载点级别标签（如果需要）
    point_labels = None
    if eval_level == 'point':
        try:
            point_labels, total_points = load_point_labels(ds_name, fname, data_root)
            print(
                f"Loaded point labels: {total_points} points, {point_labels.sum()} anomalies ({100 * point_labels.mean():.2f}%)")
        except Exception as e:
            print(f"Warning: Could not load point labels: {e}")
            print("Falling back to window-level evaluation")
            eval_level = 'window'

    # 评估
    if eval_level == 'point' and point_labels is not None:
        result = evaluate_point_level(
            window_scores=window_scores,
            point_labels=point_labels,
            window_size=window_size,
            stride=stride,
            use_pa=use_pa,
            slidingWindow=slidingWindow,
            score_method=score_method
        )
    else:
        result = evaluate_window_level(
            window_scores=window_scores,
            window_labels=window_labels,
            use_pa=use_pa,
            slidingWindow=slidingWindow
        )

    result['name'] = fname
    result['eval_level'] = eval_level

    # 打印结果
    print_result(result, f"{ds_name}/{fname}", use_pa, eval_level)

    # 保存结果
    output_cols = [
        'name', 'eval_level', 'Precision', 'Recall', 'F-score',
        'AUC-ROC', 'AUC-PR', 'ACC',
        'Aff-Pre', 'Aff-Rec', 'Aff-F1',
        'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 'F1_AUC'
    ]

    res_df = pd.DataFrame([result])
    pa_suffix = '_PA' if use_pa else '_noPA'
    output_dir = os.path.join(result_root, ds_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{ds_name}_{fname}_{eval_level}_evaluation{pa_suffix}.csv')

    res_df[[c for c in output_cols if c in res_df.columns]].to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    return result


def print_result(result, name, use_pa, eval_level):
    """打印评估结果"""
    print(f"\n{'=' * 60}")
    print(f"Result: {name}")
    print(f"Evaluation Level: {eval_level.upper()}")
    print(f"Point Adjustment: {'Enabled' if use_pa else 'Disabled'}")
    print(f"{'=' * 60}")

    print(f"\n[Label-based metrics" + (" (PA applied)]" if use_pa else " (PA not applied)]"))
    print(f"  Precision   : {result['Precision']:.4f}")
    print(f"  Recall      : {result['Recall']:.4f}")
    print(f"  F-score     : {result['F-score']:.4f}")
    print(f"  ACC         : {result['ACC']:.4f}")
    print(f"  (TP={result['TP']}, TN={result['TN']}, FP={result['FP']}, FN={result['FN']})")

    print(f"\n[Affiliation metrics" + (" (PA applied)]" if use_pa else " (PA not applied)]"))
    print(f"  Aff-Pre     : {result['Aff-Pre']:.4f}")
    print(f"  Aff-Rec     : {result['Aff-Rec']:.4f}")
    print(f"  Aff-F1      : {result['Aff-F1']:.4f}")

    print("\n[Score-based metrics]")
    print(f"  AUC-ROC     : {result['AUC-ROC']:.4f}")
    print(f"  AUC-PR      : {result['AUC-PR']:.4f}")
    print(f"  R_AUC_ROC   : {result['R_AUC_ROC']:.4f}")
    print(f"  R_AUC_PR    : {result['R_AUC_PR']:.4f}")
    print(f"  VUS_ROC     : {result['VUS_ROC']:.4f}")
    print(f"  VUS_PR      : {result['VUS_PR']:.4f}")
    print(f"  F1_AUC      : {result['F1_AUC']:.4f}")


# ============================================================
# 批量评估
# ============================================================
def evaluate_all_datasets(
        datasets=None,
        eval_level='point',
        score_method='mean',
        use_pa=False,
        data_root='data',
        result_root='results'
):
    """批量评估多个数据集"""
    if datasets is None:
        datasets = list(DATASET_CONFIG.keys())

    all_results = []

    for ds_name in datasets:
        print(f"\n{'#' * 60}")
        print(f"# Evaluating: {ds_name}")
        print(f"{'#' * 60}")

        try:
            result = evaluate_dataset(
                ds_name=ds_name,
                fname='All',
                use_pa=use_pa,
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

        summary_df = pd.DataFrame(all_results)
        cols = ['dataset', 'F-score', 'AUC-ROC', 'AUC-PR', 'Aff-F1', 'Precision', 'Recall']
        cols = [c for c in cols if c in summary_df.columns]

        print(summary_df[cols].to_string(index=False))

        # 保存汇总
        pa_suffix = '_PA' if use_pa else '_noPA'
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
        use_pa=False,
        data_root='data',
        result_root='results'
):
    """对比不同的窗口到点转换方法"""

    methods = ['mean', 'max', 'last', 'center']
    results = []

    for method in methods:
        print(f"\n{'#' * 60}")
        print(f"# Method: {method}")
        print(f"{'#' * 60}")

        result = evaluate_dataset(
            ds_name=ds_name,
            fname=fname,
            use_pa=use_pa,
            eval_level='point',
            score_method=method,
            data_root=data_root,
            result_root=result_root
        )

        if result:
            result['method'] = method
            results.append(result)

    # 打印对比表格
    if results:
        print("\n" + "=" * 80)
        print("Method Comparison")
        print("=" * 80)
        print(f"{'Method':<10} {'F-score':>10} {'AUC-ROC':>10} {'AUC-PR':>10} {'Aff-F1':>10}")
        print("-" * 50)
        for r in results:
            print(
                f"{r['method']:<10} {r['F-score']:>10.4f} {r['AUC-ROC']:>10.4f} {r['AUC-PR']:>10.4f} {r['Aff-F1']:>10.4f}")

    return results


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified Anomaly Detection Evaluation')
    parser.add_argument('--dataset', type=str, default='MSL',
                        help='Dataset name (MSL, SMAP, SMD, PSM, SWAT, NIPS_TS_Swan, NIPS_TS_Water)')
    parser.add_argument('--fname', type=str, default='All',
                        help='File name')
    parser.add_argument('--use_pa', action='store_true',
                        help='Enable Point Adjustment')
    parser.add_argument('--sliding_window', type=int, default=100,
                        help='Sliding window for VUS/R_AUC')
    parser.add_argument('--eval_level', type=str, default='point',
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

    if args.eval_all:
        evaluate_all_datasets(
            eval_level=args.eval_level,
            score_method=args.score_method,
            use_pa=args.use_pa,
            data_root=args.data_root,
            result_root=args.result_root
        )
    elif args.compare_methods:
        compare_methods(
            ds_name=args.dataset,
            fname=args.fname,
            use_pa=args.use_pa,
            data_root=args.data_root,
            result_root=args.result_root
        )
    else:
        evaluate_dataset(
            ds_name=args.dataset,
            fname=args.fname,
            use_pa=args.use_pa,
            slidingWindow=args.sliding_window,
            eval_level=args.eval_level,
            score_method=args.score_method,
            data_root=args.data_root,
            result_root=args.result_root
        )