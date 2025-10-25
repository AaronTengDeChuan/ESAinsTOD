# coding: utf-8
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, classification_report
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list


def load_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    y_true, y_pred = defaultdict(list), defaultdict(list)
    intent_names = defaultdict(set)
    errors = defaultdict(lambda : defaultdict(list))
    did2sent = {}
    for entry in results:
        for turn in entry:
            for dom, intents in turn['ispn'].items():
                intent_names[dom].update(intents)

    meaningless_intents = defaultdict(int)
    for entry in results:
        for turn in entry:
            did = turn['dial_id']
            did2sent[did] = turn['user']
            dom_g = turn['ispn'].keys()
            dom_p = turn['ispn_gen'].keys()
            overlap = set(dom_g) & set(dom_p)
            for dom in overlap:
                assert len(turn['ispn'][dom]) == 1, f"Multiple intents in gold for {did} {dom}: {turn['ispn'][dom]}"
                assert len(turn['ispn_gen'][dom]) == 1, f"Multiple intents in pred for {did} {dom}: {turn['ispn_gen'][dom]}"
                gold_int = turn['ispn'][dom][0]
                pred_int = turn['ispn_gen'][dom][0]
                if pred_int not in intent_names[dom]:
                    meaningless_intents[pred_int] += 1
                    continue
                y_true[dom].append(gold_int)
                y_pred[dom].append(pred_int)
                if gold_int != pred_int:
                    errors[dom][gold_int].append((did, pred_int))

    print(f"Ignored meaningless intents: {json.dumps(meaningless_intents, indent=2)}")
    # print counter for each intent
    for dom in errors:
        total = 0
        print("\n" + "=" * 50)
        print(f"Domain: {dom}")
        for intent, err_list in sorted(errors[dom].items(), key=lambda x: len(x[1]), reverse=True):
            counter = dict(Counter([pred for _, pred in err_list]).most_common())
            total += len(err_list)
            if len(err_list) >= 5:
                print(f"Intent: {intent}, Errors: {len(err_list)}, Predicted as: {json.dumps(counter, indent=2)}")
                print("-" * 50)
        print(f"Total errors in domain '{dom}': {total}, Total samples: {len(y_true[dom])}, Error Rate: {total / len(y_true[dom]):.2%}")
    return y_true, y_pred, intent_names, errors, did2sent


# --- 1-1 绘制归一化混淆矩阵 ---

def plot_normalized_confusion_matrix(y_true, y_pred, labels):
    """
    绘制归一化混淆矩阵，专注于行（真实值）标准化，
    以观察每一类有多少被正确分类和错误分类。
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # 行标准化：每一行除以该行的总和 (即：真实类别下的预测百分比)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 调整图形大小以适应77个类别
    plt.figure(figsize=(25, 20))

    # 使用Seaborn的热力图
    sns.heatmap(
        cm_normalized,
        annot=False,  # 不在每个单元格上显示数值，因为数值太多且图太小
        fmt='.2f',
        cmap='Blues',  # 蓝色系通常用于此目的
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,  # 增加线条分隔，提高可读性
        linecolor='lightgrey',
        cbar=True
    )

    plt.title('Normalized Confusion Matrix (Row-Normalized) - 77 Intents', fontsize=20)
    plt.ylabel('True Intent', fontsize=18)
    plt.xlabel('Predicted Intent', fontsize=18)

    # 缩小标签字体，使其适应图表
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6, rotation=0)

    plt.tight_layout()
    plt.show()


# --- 1-2 绘制分类报告热力图 (推荐的可观性更高的图表) ---

def plot_classification_report_heatmap(y_true, y_pred, labels):
    """
    将分类报告 (Precision, Recall, F1-Score) 转换为热力图进行可视化。
    """
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()

    # 提取关键指标 (排除 'accuracy', 'macro avg', 'weighted avg')
    metrics_df = df_report.iloc[:-3, :][['precision', 'recall', 'f1-score']]

    # 调整图形大小
    plt.figure(figsize=(15, 25))

    sns.heatmap(
        metrics_df,
        annot=True,
        cmap='viridis',  # 'viridis' 或 'YlGnBu' 都是不错的选择
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Score Value'}
    )

    plt.title('Classification Report Heatmap (Precision, Recall, F1-Score)', fontsize=20)
    plt.ylabel('Intent Name', fontsize=18)
    plt.xlabel('Metric', fontsize=18)

    # 缩小标签字体，使其适应图表
    plt.yticks(fontsize=8, rotation=0)
    plt.xticks(fontsize=10)

    plt.tight_layout()
    plt.show()


# --- 2-1 核心功能：自动识别低性能意图 ---

def get_low_performance_intents(y_true, y_pred, labels, threshold=0.6):
    """
    根据F1-Score阈值，自动筛选出低性能的意图列表。

    参数:
        y_true (list/array): 真实标签。
        y_pred (list/array): 预测标签。
        labels (list): 所有意图的名称列表。
        threshold (float): F1-Score的阈值，低于此值视为低性能。

    返回:
        list: 低于阈值的意图名称列表 (按F1-Score升序排列)。
        pd.DataFrame: 完整的分类报告DataFrame。
    """
    # 获取分类报告的字典格式
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()

    # 筛选出每个类别的F1-Score
    # 排除 'accuracy', 'macro avg', 'weighted avg'
    class_f1_scores = df_report.iloc[:-3]['f1-score']

    # 筛选出F1-Score低于阈值的类别
    low_performance_classes = class_f1_scores[class_f1_scores < threshold]

    # 按F1-Score升序排序，最差的排在最前面
    low_performance_classes_sorted = low_performance_classes.sort_values(ascending=True)

    print(f"检测到 {len(low_performance_classes_sorted)} 个 F1-Score 低于 {threshold:.2f} 的意图：")
    print(low_performance_classes_sorted)

    return low_performance_classes_sorted.index.tolist(), df_report


# --- 2-2. 绘制子集混淆矩阵的函数 ---

def plot_subset_confusion_matrix(y_true, y_pred, subset_labels, title_suffix=""):
    """
    绘制仅包含低性能意图的混淆矩阵子集。

    由于子集通常较小，可以考虑显示数值。
    """
    cm = confusion_matrix(y_true, y_pred, labels=subset_labels)
    # 行标准化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 调整图形大小，适应子集数量
    # 动态调整 figsize，例如每个意图分配1个单位长度
    size = max(8, len(subset_labels) * 0.8)
    plt.figure(figsize=(size, size))

    # 针对较小的矩阵，建议开启 annot
    sns.heatmap(
        cm_normalized,
        annot=True,  # 开启数值显示
        fmt='.2f',
        cmap='YlOrRd',  # 使用暖色调突出“问题区域”
        xticklabels=subset_labels,
        yticklabels=subset_labels,
        linewidths=0.5,
        linecolor='black',
        cbar=True
    )

    plt.title(f'Normalized Confusion Matrix for Low Performance Intents {title_suffix}', fontsize=14)
    plt.ylabel('True Intent', fontsize=12)
    plt.xlabel('Predicted Intent', fontsize=12)

    # 根据数量调整标签大小
    label_size = max(5, 10 - len(subset_labels) // 5)
    plt.xticks(fontsize=label_size, rotation=90)
    plt.yticks(fontsize=label_size, rotation=0)

    plt.tight_layout()
    plt.show()


# 3-1 定义“混淆距离”矩阵

def calculate_confusion_distance(cm_df):
    """
    计算类别间的混淆距离 (Dissimilarity Matrix)。
    距离定义： (行标准化后的非对角线错误率 + 列标准化后的非对角线错误率) / 2
    该距离矩阵是半对称的。
    """
    # 行标准化：每一行被错误分类的比例 (1 - 召回率)
    cm_row_norm = cm_df.div(cm_df.sum(axis=1), axis=0)

    # 距离矩阵初始化
    distance_matrix = cm_row_norm.copy()

    # 距离计算：非对角线元素的值代表类别之间的混淆程度
    for i in cm_df.index:
        for j in cm_df.columns:
            if i == j:
                # 对角线距离设为0 (自己和自己的距离)
                distance_matrix.loc[i, j] = 0
            else:
                # 距离计算：i 被错分成 j 的比例
                distance_matrix.loc[i, j] = cm_row_norm.loc[i, j]

    # 为了聚类算法，我们倾向于使用一个接近对称的距离度量，
    # 但由于混淆矩阵的非对称性，我们只使用行标准化距离作为聚类依据
    # 也可以尝试使用 (1 - F1-Score) 作为距离度量，这里我们使用原始的行标准化距离

    # 我们需要将DataFrame转换为scipy所需的condensed distance matrix
    # 使用 distance_matrix.values 作为原始数据进行聚类
    return distance_matrix.values


# 3-2 层次聚类并获取排序

def cluster_and_reorder(distance_matrix_values, labels):
    """
    对距离矩阵进行层次聚类，并返回最佳的排序顺序。
    """
    # 将距离矩阵转换为向量形式 (scipy.cluster.hierarchy.linkage 要求)
    # 由于我们的距离矩阵是 N x N 的，且不是标准的距离矩阵，直接使用 linkage
    # 需要将其转换为 condensed distance matrix (只包含上三角或下三角)
    from scipy.spatial.distance import squareform

    # 构造一个近似对称的距离矩阵 (例如：使用 max(A[i,j], A[j,i]) 作为 i, j 之间的距离)
    # 这里我们简化，直接使用 distance_matrix_values 作为输入，
    # 但需要确保对角线为0，且值非负。

    # 简化处理：使用类别之间的欧氏距离作为 dissimilarity，
    # 即将每个意图的错误分类模式视为一个向量，计算其相似性。
    # Z = linkage(distance_matrix_values, method='ward')  # 'ward' 适用于欧氏距离

    # 另一种方法：直接将 row_norm 视为特征向量
    cm_row_norm = cm_subset_df.div(cm_subset_df.sum(axis=1), axis=0)
    Z = linkage(cm_row_norm.fillna(0).values, method='ward', metric='euclidean')

    # 获取聚类后的最佳叶节点顺序
    reordered_indices = leaves_list(Z)
    reordered_labels = [labels[i] for i in reordered_indices]

    return Z, reordered_labels


# 3-3 绘制聚类排序后的混淆矩阵

def plot_clustered_confusion_matrix(cm_df, reordered_labels):
    """
    绘制经过聚类排序后的混淆矩阵热力图。
    """
    # 重新排序 DataFrame
    cm_clustered = cm_df.reindex(index=reordered_labels, columns=reordered_labels)

    # 归一化 (行标准化)
    cm_normalized = cm_clustered.div(cm_clustered.sum(axis=1), axis=0)
    cm_normalized = cm_normalized.fillna(0)  # 避免 NaN

    # 调整图形大小
    size = max(10, len(reordered_labels) * 0.8)
    plt.figure(figsize=(size, size))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',  # 换一个颜色方案，例如 'YlGnBu'
        xticklabels=reordered_labels,
        yticklabels=reordered_labels,
        linewidths=0.5,
        linecolor='black',
        cbar=True
    )

    plt.title('Clustered Normalized Confusion Matrix (Focusing on Misclassification)', fontsize=16)
    plt.ylabel('True Intent (Clustered)', fontsize=14)
    plt.xlabel('Predicted Intent (Clustered)', fontsize=14)

    # 缩小标签字体
    label_size = max(5, 10 - len(reordered_labels) // 5)
    plt.xticks(fontsize=label_size, rotation=90)
    plt.yticks(fontsize=label_size, rotation=0)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    enable_analyze = False

    file_path = "infer_results/v1.0/single_turn/banking/test_samples_2024-03-17_09-03-40.484099.json"
    dom2true, dom2pred, dom2names, errors, did2sent = load_results(file_path)

    if enable_analyze:
        # 设置F1-Score阈值，用于识别低性能意图 (您可以根据模型性能调整)
        F1_THRESHOLD = 0.9
        for dom in dom2true:
            y_true, y_pred = dom2true[dom], dom2pred[dom]
            intent_names = list(dom2names[dom])
            print(f"Domain: {dom}, Number of samples: {len(y_true)}, Unique intents: {len(intent_names)}")

            # continue
            # --- 4. 执行绘图 ---
            # 1. 绘制归一化混淆矩阵
            # print("--- 绘制归一化混淆矩阵 (可能较密) ---")
            # plot_normalized_confusion_matrix(y_true, y_pred, labels=intent_names))

            # 2. 绘制分类报告热力图 (推荐的更可观的图表)
            # print("\n--- 绘制分类报告热力图 (更推荐，可观性高) ---")
            # plot_classification_report_heatmap(y_true, y_pred, labels=intent_names))


            # 自动识别低性能意图
            low_intents, full_report_df = get_low_performance_intents(
                y_true,
                y_pred,
                labels=intent_names,
                threshold=F1_THRESHOLD
            )

            if low_intents:
                pass
                # print("\n--- 绘制低性能意图的混淆矩阵子集 (推荐的可观性) ---")
                # plot_subset_confusion_matrix(y_true, y_pred, low_intents, title_suffix=f"(F1-Score < {F1_THRESHOLD:.2f})")
                #
                # # 绘制低性能意图的分类报告热力图子集
                # print("\n--- 绘制低性能意图的分类报告热力图子集 ---")
                # plt.figure(figsize=(8, len(low_intents) * 0.5))
                #
                # subset_report = full_report_df.loc[low_intents][['precision', 'recall', 'f1-score']]
                # sns.heatmap(
                #     subset_report,
                #     annot=True,
                #     cmap='viridis',
                #     fmt='.2f',
                #     linewidths=0.5
                # )
                #
                # plt.title(f'Classification Report for Low Performance Intents (F1-Score < {F1_THRESHOLD:.2f})', fontsize=14)
                # plt.yticks(rotation=0)
                # plt.tight_layout()
                # plt.show()
            else:
                print("\n所有意图的性能都高于设定的阈值，无需单独展示子集。")


            # 1. 计算混淆矩阵子集
            cm_subset = confusion_matrix(y_true, y_pred, labels=low_intents)
            cm_subset_df = pd.DataFrame(cm_subset, index=low_intents, columns=low_intents)
            distance_matrix_values = calculate_confusion_distance(cm_subset_df)
            # 执行聚类和排序
            Z, reordered_intents = cluster_and_reorder(distance_matrix_values, low_intents)
            # 4. 绘制聚类结构图 (Dendrogram)
            plt.figure(figsize=(10, 5))
            dendrogram(
                Z,
                labels=low_intents,
                orientation='top',
                leaf_rotation=90,
                leaf_font_size=8
            )
            plt.title('Hierarchical Clustering Dendrogram of Low Performance Intents', fontsize=14)
            plt.ylabel('Distance', fontsize=12)
            plt.tight_layout()
            plt.show()
            # 最终绘制聚类后的混淆矩阵
            print("\n--- 绘制经过聚类排序的混淆矩阵 (更容易看出混淆组) ---")
            plot_clustered_confusion_matrix(cm_subset_df, reordered_intents)
    else:
        counter = Counter(dom2true["banking"])
        dom_errors = errors["banking"]
        similar_groups = [
            ["card_delivery_estimate", "card_arrival"],
            ["topping_up_by_card", "transfer_into_account"],
            ["declined_card_payment", "declined_transfer", "failed_transfer"],
            ["compromised_card", "card_payment_not_recognised", "direct_debit_payment_not_recognised"],
            ["transfer_timing", "balance_not_updated_after_cheque_or_cash_deposit", "pending_transfer"],
        ]
        for group in similar_groups:
            print("\n" + "=" * 50)
            print(f"Analyzing similar intent group: {group}")
            group_cnt, tot = 0, 0
            for g in group:
                print("-" * 50)
                pred2cases = defaultdict(list)
                g_errors = dom_errors.get(g, [])
                cnt = 0
                for did, p in g_errors:
                    if p in group:
                        pred2cases[p].append(did)
                        cnt += 1
                if pred2cases:
                    print(f"Intent: {g}, Total: {counter[g]}, Errors: {len(g_errors)}, Errors in Group: {cnt}, Error Rate: {cnt / counter[g]:.2%}")
                    for p, cases in pred2cases.items():
                        print(f"  Predicted as '{p}': {len(cases)} cases, Examples:")
                        for i, case in enumerate(cases, start=1):
                            print(f"    - Case {i}: {did2sent[case]}")
                else:
                    print(f"Intent: {g}, Total: {counter[g]}, Errors: {len(g_errors)}, No errors in this group.")
                group_cnt += cnt
                tot += counter[g]
            print(f"Group Total Samples: {tot}, Group Total Errors in Group: {group_cnt}, Group Error Rate: {group_cnt} / {tot} = {group_cnt / tot:.2%}")
        print("\n" + "=" * 50)
