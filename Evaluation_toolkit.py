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
    """
    计算基于分数的指标（不受 PA 影响）：
    AUC-ROC, AUC-PR, R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, F1_AUC
    """
    result = {}

    # AUC-ROC 和 AUC-PR
    result['AUC-ROC'] = roc_auc_score(y_true, scores)
    result['AUC-PR'] = average_precision_score(y_true, scores)

    # F1-AUC
    result['F1_AUC'] = compute_f1_auc(y_true, scores)

    # R_AUC 和 VUS
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
# 单通道评估函数
# ============================================================
def evaluate_single_channel(df_train, df_test, use_pa, slidingWindow):
    """评估单个通道，返回所有指标"""
    result = {}

    cl_num = df_train.shape[1] - 1

    # 确定正常类别
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

    # 基于标签的指标（可选 PA）
    if use_pa:
        y_pred = point_adjustment(y_true, y_pred_raw)
    else:
        y_pred = y_pred_raw

    label_metrics = compute_label_based_metrics(y_true, y_pred)
    result.update(label_metrics)

    # Affiliation 指标
    aff_pre, aff_rec, aff_f1 = compute_affiliation_metrics(y_true, y_pred)
    result['Aff-Pre'] = aff_pre
    result['Aff-Rec'] = aff_rec
    result['Aff-F1'] = aff_f1

    return result


# ============================================================
# 统计汇总函数
# ============================================================
def compute_macro_metrics(res_df):
    """计算 Macro 平均（每通道平均）"""
    metric_cols = [
        'Precision', 'Recall', 'F-score', 'ACC',
        'Aff-Pre', 'Aff-Rec', 'Aff-F1',
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
    """计算 Micro 平均（汇总 TP/FP/FN/TN）"""
    if not all(col in res_df.columns for col in ['TP', 'TN', 'FP', 'FN']):
        return {}

    sum_tp = res_df['TP'].sum()
    sum_tn = res_df['TN'].sum()
    sum_fp = res_df['FP'].sum()
    sum_fn = res_df['FN'].sum()

    micro_pre = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    micro_rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    micro_f1 = 2 * micro_pre * micro_rec / (micro_pre + micro_rec) \
        if (micro_pre + micro_rec) > 0 else 0
    micro_acc = (sum_tp + sum_tn) / (sum_tp + sum_tn + sum_fp + sum_fn)

    return {
        'Precision': micro_pre,
        'Recall': micro_rec,
        'F-score': micro_f1,
        'ACC': micro_acc,
        'TP': int(sum_tp),
        'TN': int(sum_tn),
        'FP': int(sum_fp),
        'FN': int(sum_fn)
    }


# ============================================================
# 打印函数
# ============================================================
def print_single_result(result, name, use_pa):
    """打印单个结果（All 模式或单通道）"""
    print(f"\n{'=' * 60}")
    print(f"Result: {name}")
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

    print("\n[Score-based metrics (PA not applicable)]")
    print(f"  AUC-ROC     : {result['AUC-ROC']:.4f}")
    print(f"  AUC-PR      : {result['AUC-PR']:.4f}")
    print(f"  R_AUC_ROC   : {result['R_AUC_ROC']:.4f}")
    print(f"  R_AUC_PR    : {result['R_AUC_PR']:.4f}")
    print(f"  VUS_ROC     : {result['VUS_ROC']:.4f}")
    print(f"  VUS_PR      : {result['VUS_PR']:.4f}")
    print(f"  F1_AUC      : {result['F1_AUC']:.4f}")


def print_multi_channel_summary(macro, micro, n_channels, use_pa):
    """打印多通道汇总结果"""
    print(f"\n{'=' * 60}")
    print(f"Multi-Channel Summary ({n_channels} channels)")
    print(f"Point Adjustment: {'Enabled' if use_pa else 'Disabled'}")
    print(f"{'=' * 60}")

    # Macro 平均
    print(f"\n[MACRO-averaged metrics (mean ± std across {n_channels} channels)]")

    print(f"  --- Label-based" + (" (PA applied)" if use_pa else " (PA not applied)") + " ---")
    for metric in ['Precision', 'Recall', 'F-score', 'ACC']:
        mean_val = macro.get(f'{metric}_mean', 0)
        std_val = macro.get(f'{metric}_std', 0)
        if n_channels > 1:
            print(f"  {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"  {metric:12s}: {mean_val:.4f}")

    print(f"  --- Affiliation" + (" (PA applied)" if use_pa else " (PA not applied)") + " ---")
    for metric in ['Aff-Pre', 'Aff-Rec', 'Aff-F1']:
        mean_val = macro.get(f'{metric}_mean', 0)
        std_val = macro.get(f'{metric}_std', 0)
        if n_channels > 1:
            print(f"  {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"  {metric:12s}: {mean_val:.4f}")

    print("  --- Score-based (PA not applicable) ---")
    for metric in ['AUC-ROC', 'AUC-PR', 'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 'F1_AUC']:
        mean_val = macro.get(f'{metric}_mean', 0)
        std_val = macro.get(f'{metric}_std', 0)
        if n_channels > 1:
            print(f"  {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"  {metric:12s}: {mean_val:.4f}")

    # Micro 平均
    print(f"\n[MICRO-averaged metrics (aggregate TP/FP/FN/TN)" + \
          (" (PA applied)]" if use_pa else " (PA not applied)]"))
    print(f"  Precision   : {micro['Precision']:.4f}")
    print(f"  Recall      : {micro['Recall']:.4f}")
    print(f"  F-score     : {micro['F-score']:.4f}")
    print(f"  ACC         : {micro['ACC']:.4f}")
    print(f"  (Total: TP={micro['TP']}, TN={micro['TN']}, FP={micro['FP']}, FN={micro['FN']})")


# ============================================================
# 【新增】查找 All 模式结果文件的函数
# ============================================================
def find_all_mode_files(ds_name):
    """
    查找 All 模式的结果文件，支持多种可能的路径结构
    返回 (train_path, test_path) 或 (None, None)
    """
    possible_paths = [
        # 路径格式 1: results/{ds_name}/All/classification/
        (f"results/{ds_name}/All/classification/classification_trainprobs.csv",
         f"results/{ds_name}/All/classification/classification_testprobs.csv"),
        # 路径格式 2: results/{ds_name}/classification/ (直接在数据集目录下)
        (f"results/{ds_name}/classification/classification_trainprobs.csv",
         f"results/{ds_name}/classification/classification_testprobs.csv"),
        # 路径格式 3: results/{ds_name}_All/classification/
        (f"results/{ds_name}_All/classification/classification_trainprobs.csv",
         f"results/{ds_name}_All/classification/classification_testprobs.csv"),
    ]

    for train_path, test_path in possible_paths:
        if os.path.exists(train_path) and os.path.exists(test_path):
            return train_path, test_path

    return None, None


# ============================================================
# 【新增】检测数据集的结果结构
# ============================================================
def detect_result_structure(ds_name):
    """
    检测结果目录的结构
    返回: 'all_only', 'single_only', 'both', 'none'
    """
    path = os.path.join('results/', ds_name)

    if not os.path.exists(path):
        return 'none'

    # 检查是否有 All 模式结果
    train_all, test_all = find_all_mode_files(ds_name)
    has_all = train_all is not None

    # 检查是否有单通道结果
    has_single = False
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path) and item not in ['All', 'classification']:
                # 检查是否有 classification 子目录
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
def evaluate_dataset(ds_name, use_pa=False, slidingWindow=100, mode='auto', fname='All'):
    """
    评估数据集

    Args:
        ds_name: 数据集名称 (如 'MSL', 'SMAP')
        use_pa: 是否对基于标签的指标使用 PA 调整
        slidingWindow: VUS/R_AUC 的滑动窗口大小
        mode: 评估模式
            - 'auto': 自动检测（根据目录结构）
            - 'single': 单通道模式（逐个通道评估）
            - 'all': All 模式（多通道聚合后的单一结果）
        fname: 文件名，'All' 表示评估所有通道聚合结果
    """
    print("=" * 60)
    print(f"Dataset: {ds_name}")
    print(f"Fname: {fname}")
    print(f"Point Adjustment (PA): {'Enabled' if use_pa else 'Disabled'}")
    print(f"Sliding Window: {slidingWindow}")
    print(f"Mode: {mode}")
    print("=" * 60)

    path = os.path.join('results/', ds_name)

    # ============================================================
    # 【修改】改进的模式检测逻辑
    # ============================================================
    if mode == 'auto':
        structure = detect_result_structure(ds_name)
        print(f"Detected structure: {structure}")

        if fname == 'All':
            # 用户指定了 All，优先使用 all 模式
            if structure in ['all_only', 'both']:
                mode = 'all'
            else:
                print(f"Warning: fname='All' but no All mode results found, falling back to single mode")
                mode = 'single'
        else:
            # 用户指定了具体文件名，使用 single 模式
            mode = 'single'

        print(f"Selected mode: {mode}")

    # ============================================================
    # All 模式：评估多通道聚合后的单一结果
    # ============================================================
    if mode == 'all':
        print("\n[All Mode] Evaluating aggregated multi-channel result...")

        # 【修改】使用新的文件查找函数
        train_path, test_path = find_all_mode_files(ds_name)

        if train_path is None:
            print(f"Error: Cannot find All mode result files for {ds_name}")
            print("Searched paths:")
            print(f"  - results/{ds_name}/All/classification/")
            print(f"  - results/{ds_name}/classification/")
            print(f"  - results/{ds_name}_All/classification/")
            return None, None

        print(f"Found train file: {train_path}")
        print(f"Found test file: {test_path}")

        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
        except Exception as e:
            print(f"Error reading files: {e}")
            return None, None

        result = evaluate_single_channel(df_train, df_test, use_pa, slidingWindow)
        result['name'] = 'All'

        # 打印结果
        print_single_result(result, f"{ds_name} (All channels aggregated)", use_pa)

        # 保存结果
        output_cols = [
            'name', 'Precision', 'Recall', 'F-score',
            'AUC-ROC', 'AUC-PR', 'ACC',
            'Aff-Pre', 'Aff-Rec', 'Aff-F1',
            'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 'F1_AUC'
        ]

        res_df = pd.DataFrame([result])
        pa_suffix = '_PA' if use_pa else '_noPA'
        output_file = os.path.join(path, f'{ds_name}_All_evaluation{pa_suffix}.csv')

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        res_df[output_cols].to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        return res_df, result

    # ============================================================
    # Single 模式：逐个通道评估，计算 Macro 和 Micro
    # ============================================================
    else:
        print("\n[Single Mode] Evaluating each channel separately...")

        columns = [
            'name',
            'Precision', 'Recall', 'F-score', 'ACC',
            'Aff-Pre', 'Aff-Rec', 'Aff-F1',
            'AUC-ROC', 'AUC-PR',
            'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 'F1_AUC',
            'TP', 'TN', 'FP', 'FN', 'threshold'
        ]

        res_df = pd.DataFrame(columns=columns)

        # 【修改】如果指定了具体的 fname（非 All），只评估该文件
        if fname != 'All':
            file_list = [fname]
        else:
            file_list = sorted(os.listdir(path))

        for filename in file_list:
            # 跳过非目录和特殊文件
            full_path = os.path.join(path, filename)
            if not os.path.isdir(full_path):
                continue
            if filename in ['All', 'GECCO', '.json', 'classification']:
                continue

            print(f"\nProcessing: {filename}")

            try:
                df_train = pd.read_csv(f"{full_path}/classification/classification_trainprobs.csv")
                df_test = pd.read_csv(f"{full_path}/classification/classification_testprobs.csv")
            except FileNotFoundError:
                print(f"  Skipping: file not found")
                continue

            try:
                result = evaluate_single_channel(df_train, df_test, use_pa, slidingWindow)
                result['name'] = filename

                print(f"  Threshold: {result['threshold']:.4f}")
                print(f"  F-score: {result['F-score']:.4f}, AUC-ROC: {result['AUC-ROC']:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                result = {'name': filename}
                for col in columns[1:]:
                    result[col] = 0

            res_df = res_df._append(pd.Series(result), ignore_index=True)

        n_channels = len(res_df)

        if n_channels == 0:
            print("\nNo valid results found!")
            return None, None

        # 计算 Macro 和 Micro
        macro = compute_macro_metrics(res_df)
        micro = compute_micro_metrics(res_df)

        # 打印汇总
        print_multi_channel_summary(macro, micro, n_channels, use_pa)

        # 保存详细结果
        output_cols = [
            'name', 'Precision', 'Recall', 'F-score',
            'AUC-ROC', 'AUC-PR', 'ACC',
            'Aff-Pre', 'Aff-Rec', 'Aff-F1',
            'R_AUC_ROC', 'R_AUC_PR', 'VUS_ROC', 'VUS_PR', 'F1_AUC'
        ]

        pa_suffix = '_PA' if use_pa else '_noPA'

        # 详细结果（每通道）
        detail_file = os.path.join(path, f'{ds_name}_channels_evaluation{pa_suffix}.csv')
        res_df[output_cols].to_csv(detail_file, index=False)
        print(f"\nChannel details saved to {detail_file}")

        # 汇总结果
        summary_data = {
            'Aggregation': ['Macro', 'Micro'],
            'Precision': [macro['Precision_mean'], micro['Precision']],
            'Recall': [macro['Recall_mean'], micro['Recall']],
            'F-score': [macro['F-score_mean'], micro['F-score']],
            'ACC': [macro['ACC_mean'], micro['ACC']],
            'Aff-Pre': [macro['Aff-Pre_mean'], '-'],
            'Aff-Rec': [macro['Aff-Rec_mean'], '-'],
            'Aff-F1': [macro['Aff-F1_mean'], '-'],
            'AUC-ROC': [macro['AUC-ROC_mean'], '-'],
            'AUC-PR': [macro['AUC-PR_mean'], '-'],
            'R_AUC_ROC': [macro['R_AUC_ROC_mean'], '-'],
            'R_AUC_PR': [macro['R_AUC_PR_mean'], '-'],
            'VUS_ROC': [macro['VUS_ROC_mean'], '-'],
            'VUS_PR': [macro['VUS_PR_mean'], '-'],
            'F1_AUC': [macro['F1_AUC_mean'], '-'],
        }

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(path, f'{ds_name}_summary{pa_suffix}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")

        return res_df, {'macro': macro, 'micro': micro}


# ============================================================
# 对比函数：同时运行 PA 和非 PA 评估
# ============================================================
def compare_pa_effect(ds_name, slidingWindow=100, mode='auto', fname='All'):
    """对比 PA 调整的效果"""
    print("\n" + "#" * 60)
    print("# Running evaluation WITHOUT Point Adjustment")
    print("#" * 60)
    res_nopa, summary_nopa = evaluate_dataset(ds_name, use_pa=False,
                                              slidingWindow=slidingWindow, mode=mode, fname=fname)

    print("\n" + "#" * 60)
    print("# Running evaluation WITH Point Adjustment")
    print("#" * 60)
    res_pa, summary_pa = evaluate_dataset(ds_name, use_pa=True,
                                          slidingWindow=slidingWindow, mode=mode, fname=fname)

    if summary_nopa is None or summary_pa is None:
        print("Error: Could not complete comparison")
        return None, None

    # 打印对比
    print("\n" + "=" * 60)
    print("PA Effect Comparison (Label-based metrics only)")
    print("=" * 60)

    if mode == 'all' or (isinstance(summary_nopa, dict) and 'Precision' in summary_nopa):
        # All 模式，直接对比
        print(f"\n{'Metric':<12} {'Without PA':>12} {'With PA':>12} {'Δ':>10}")
        print("-" * 48)
        for metric in ['Precision', 'Recall', 'F-score', 'ACC', 'Aff-Pre', 'Aff-Rec', 'Aff-F1']:
            val_nopa = summary_nopa.get(metric, 0)
            val_pa = summary_pa.get(metric, 0)
            delta = val_pa - val_nopa
            print(f"{metric:<12} {val_nopa:>12.4f} {val_pa:>12.4f} {delta:>+10.4f}")
    else:
        # Single 模式，对比 Macro 和 Micro
        print("\n[Macro-averaged]")
        print(f"{'Metric':<12} {'Without PA':>12} {'With PA':>12} {'Δ':>10}")
        print("-" * 48)
        for metric in ['Precision', 'Recall', 'F-score', 'ACC', 'Aff-Pre', 'Aff-Rec', 'Aff-F1']:
            val_nopa = summary_nopa['macro'].get(f'{metric}_mean', 0)
            val_pa = summary_pa['macro'].get(f'{metric}_mean', 0)
            delta = val_pa - val_nopa
            print(f"{metric:<12} {val_nopa:>12.4f} {val_pa:>12.4f} {delta:>+10.4f}")

        print("\n[Micro-averaged]")
        print(f"{'Metric':<12} {'Without PA':>12} {'With PA':>12} {'Δ':>10}")
        print("-" * 48)
        for metric in ['Precision', 'Recall', 'F-score', 'ACC']:
            val_nopa = summary_nopa['micro'].get(metric, 0)
            val_pa = summary_pa['micro'].get(metric, 0)
            delta = val_pa - val_nopa
            print(f"{metric:<12} {val_nopa:>12.4f} {val_pa:>12.4f} {delta:>+10.4f}")

    return res_nopa, res_pa


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anomaly Detection Evaluation')
    parser.add_argument('--dataset', type=str, default='MSL',
                        help='Dataset name')
    parser.add_argument('--fname', type=str, default='All',
                        help='File name: "All" for aggregated evaluation, or specific channel name')
    parser.add_argument('--use_pa', action='store_true',
                        help='Enable Point Adjustment for label-based metrics')
    parser.add_argument('--sliding_window', type=int, default=100,
                        help='Sliding window for VUS/R_AUC')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'single', 'all'],
                        help='Evaluation mode: auto, single (per-channel), all (aggregated)')
    parser.add_argument('--compare_pa', action='store_true',
                        help='Run both PA and non-PA evaluation and compare')

    args = parser.parse_args()

    if args.compare_pa:
        compare_pa_effect(
            ds_name=args.dataset,
            slidingWindow=args.sliding_window,
            mode=args.mode,
            fname=args.fname
        )
    else:
        evaluate_dataset(
            ds_name=args.dataset,
            use_pa=args.use_pa,
            slidingWindow=args.sliding_window,
            mode=args.mode,
            fname=args.fname
        )