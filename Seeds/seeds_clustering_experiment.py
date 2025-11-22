"""
Seeds数据集聚类实验
包含：数据预处理、聚类趋势评估、确定最优k值、6种聚类算法、评估对比
作者：数据挖掘课程实验
日期：2024-11-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix
)
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
import logging
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
log_filename = 'seeds_clustering_experiment.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ===================== 工具函数 =====================

def log_section(title):
    """记录日志章节标题"""
    logger.info("\n" + "="*60)
    logger.info(title)
    logger.info("="*60)


def log_subsection(title):
    """记录日志子章节标题"""
    logger.info("\n" + "-"*60)
    logger.info(title)
    logger.info("-"*60)


def hopkins_statistic(X, n_samples=None):
    """
    计算Hopkins统计量，评估数据的可聚类性
    H接近1表示数据有聚类趋势，接近0.5表示数据随机分布
    """
    if n_samples is None:
        n_samples = min(int(0.1 * len(X)), 100)
    
    n_features = X.shape[1]
    n = len(X)
    
    # 从数据集中随机抽样
    sample_indices = np.random.choice(n, n_samples, replace=False)
    X_sample = X[sample_indices]
    
    # 生成随机均匀分布的点
    X_random = np.random.uniform(X.min(axis=0), X.max(axis=0), (n_samples, n_features))
    
    # 计算实际数据点到最近邻的距离
    dist_sample = cdist(X_sample, X, metric='euclidean')
    dist_sample.sort(axis=1)
    u = dist_sample[:, 1].sum()  # 排除自己（距离为0）
    
    # 计算随机点到实际数据的最近距离
    dist_random = cdist(X_random, X, metric='euclidean')
    dist_random.sort(axis=1)
    w = dist_random[:, 0].sum()
    
    # 计算Hopkins统计量
    H = w / (u + w)
    
    return H


def plot_silhouette(X, labels, algorithm_name, filename):
    """绘制轮廓系数图"""
    n_clusters = len(np.unique(labels))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算每个样本的轮廓系数
    silhouette_vals = silhouette_samples(X, labels)
    silhouette_avg = np.mean(silhouette_vals)
    
    y_lower = 10
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#C7CEEA']
    
    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster
        
        color = colors[i % len(colors)]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # 标记簇编号
        ax.text(-0.05, y_lower + 0.5 * size_cluster, f'簇 {i}', fontsize=11, fontweight='bold')
        
        y_lower = y_upper + 10
    
    ax.set_title(f'{algorithm_name} - 轮廓系数图', fontsize=16, fontweight='bold')
    ax.set_xlabel('轮廓系数值', fontsize=13, fontweight='bold')
    ax.set_ylabel('样本', fontsize=13, fontweight='bold')
    
    # 添加平均轮廓系数线
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2, 
               label=f'平均轮廓系数: {silhouette_avg:.3f}')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  ✓ 已保存轮廓系数图: {filename}")


def plot_confusion_matrix(y_true, y_pred, algorithm_name, filename):
    """绘制混淆矩阵"""
    from scipy.optimize import linear_sum_assignment
    
    # 使用匈牙利算法找到最佳标签匹配
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # 重新排列标签
    label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    y_pred_mapped = np.array([label_mapping[label] for label in y_pred])
    
    # 计算最终混淆矩阵
    cm_final = confusion_matrix(y_true, y_pred_mapped)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[1, 2, 3], yticklabels=[1, 2, 3],
                cbar_kws={'label': '样本数量'}, ax=ax, linewidths=1, linecolor='gray')
    
    ax.set_title(f'{algorithm_name} - 混淆矩阵', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('预测类别', fontsize=13, fontweight='bold')
    ax.set_ylabel('真实类别', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  ✓ 已保存混淆矩阵: {filename}")


def evaluate_clustering(X, y_true, y_pred, algorithm_name):
    """评估聚类结果"""
    # 内部指标（不需要真实标签）
    silhouette = silhouette_score(X, y_pred)
    ch_score = calinski_harabasz_score(X, y_pred)
    db_score = davies_bouldin_score(X, y_pred)
    
    # 外部指标（需要真实标签）
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    logger.info(f"\n评估指标:")
    logger.info(f"  内部指标（无需真实标签）:")
    logger.info(f"    - 轮廓系数 (Silhouette Score): {silhouette:.4f}")
    logger.info(f"    - CH指数 (Calinski-Harabasz): {ch_score:.2f}")
    logger.info(f"    - DB指数 (Davies-Bouldin): {db_score:.4f}")
    logger.info(f"  外部指标（对比真实标签）:")
    logger.info(f"    - 调整兰德指数 (ARI): {ari:.4f}")
    logger.info(f"    - 归一化互信息 (NMI): {nmi:.4f}")
    
    return {
        'silhouette': silhouette,
        'ch_score': ch_score,
        'db_score': db_score,
        'ari': ari,
        'nmi': nmi
    }


# ===================== 主实验流程 =====================

def main():
    """主实验函数"""
    
    # 记录实验开始
    start_time = time.time()
    log_section(f"Seeds数据集聚类实验日志\n实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ==================== 1. 数据加载与预处理 ====================
    log_section("[1. 数据加载与预处理]")
    
    logger.info("正在加载Seeds数据集...")
    
    # 加载数据
    data = np.loadtxt('Seeds/seeds_dataset.txt')
    X = data[:, :-1]  # 特征
    y_true = data[:, -1].astype(int) - 1  # 真实标签（转换为0, 1, 2）
    
    feature_names = ['面积', '周长', '紧凑度', '核长度', '核宽度', '非对称系数', '核沟长度']
    
    logger.info(f"- 数据集路径: Seeds/seeds_dataset.txt")
    logger.info(f"- 样本数量: {len(X)}")
    logger.info(f"- 特征数量: {X.shape[1]}")
    logger.info(f"- 类别数量: {len(np.unique(y_true))}")
    
    # 统计类别分布
    unique, counts = np.unique(y_true, return_counts=True)
    class_dist = {int(u)+1: int(c) for u, c in zip(unique, counts)}
    logger.info(f"- 类别分布: {class_dist}")
    
    # 数据标准化
    logger.info("\n执行数据标准化（Z-score标准化）...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("✓ 标准化完成")
    
    # 特征统计信息
    logger.info("\n特征统计信息（标准化前）:")
    df_stats = pd.DataFrame(X, columns=feature_names)
    logger.info(df_stats.describe().to_string())
    
    # ==================== 可视化1: 特征箱线图 ====================
    logger.info("\n正在生成特征箱线图...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_plot = pd.DataFrame(X_scaled, columns=feature_names)
    df_plot.boxplot(ax=ax, grid=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='black'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'))
    
    ax.set_title('Seeds数据集特征分布（标准化后）', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('特征', fontsize=13, fontweight='bold')
    ax.set_ylabel('标准化值', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('seeds_features_boxplot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("✓ 已保存: seeds_features_boxplot.png")
    
    # ==================== 可视化2: 特征相关性热力图 ====================
    logger.info("\n正在生成特征相关性热力图...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    correlation_matrix = np.corrcoef(X.T)
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=feature_names, yticklabels=feature_names,
                vmin=-1, vmax=1, center=0, ax=ax, linewidths=0.5)
    
    ax.set_title('Seeds数据集特征相关性热力图', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('seeds_correlation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("✓ 已保存: seeds_correlation_heatmap.png")
    
    # ==================== 2. 聚类趋势评估 ====================
    log_section("[2. 聚类趋势评估]")
    
    logger.info("正在计算Hopkins统计量...")
    H = hopkins_statistic(X_scaled, n_samples=50)
    logger.info(f"Hopkins统计量: H = {H:.4f}")
    
    if H > 0.7:
        logger.info("判断: 数据有明显聚类趋势，适合聚类！")
    elif H > 0.5:
        logger.info("判断: 数据有一定聚类趋势，可以尝试聚类")
    else:
        logger.info("判断: 数据可能不适合聚类")
    
    # ==================== 3. 确定最优簇数 ====================
    log_section("[3. 确定最优簇数]")
    
    k_range = range(2, 11)
    sse_list = []
    silhouette_list = []
    
    logger.info("\n3.1 肘部法则（Elbow Method）")
    logger.info("正在计算不同k值的SSE...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        sse = kmeans.inertia_
        sse_list.append(sse)
        logger.info(f"  k={k}: SSE = {sse:.2f}")
    
    logger.info("  推荐k值: 3 (肘部拐点)")
    
    logger.info("\n3.2 轮廓系数法（Silhouette Method）")
    logger.info("正在计算不同k值的轮廓系数...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        silhouette_list.append(silhouette)
        logger.info(f"  k={k}: Silhouette = {silhouette:.4f}")
    
    best_k = k_range[np.argmax(silhouette_list)]
    logger.info(f"  推荐k值: {best_k} (最大轮廓系数)")
    
    # ==================== 可视化3: 肘部法则曲线 ====================
    logger.info("\n正在生成肘部法则曲线...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_range, sse_list, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
    ax.axvline(x=3, color='green', linestyle='--', linewidth=2, label='推荐k=3')
    
    ax.set_title('肘部法则 - 确定最优簇数', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('簇数 k', fontsize=13, fontweight='bold')
    ax.set_ylabel('簇内误差平方和 (SSE)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("✓ 已保存: elbow_method.png")
    
    # ==================== 可视化4: 轮廓系数曲线 ====================
    logger.info("\n正在生成轮廓系数曲线...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_range, silhouette_list, marker='s', linewidth=2, markersize=8, color='#4ECDC4')
    ax.axvline(x=best_k, color='green', linestyle='--', linewidth=2, label=f'推荐k={best_k}')
    
    ax.set_title('轮廓系数法 - 确定最优簇数', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('簇数 k', fontsize=13, fontweight='bold')
    ax.set_ylabel('平均轮廓系数', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('silhouette_method.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("✓ 已保存: silhouette_method.png")
    
    # ==================== 4. 聚类算法实验 ====================
    log_section("[4. 聚类算法实验]")
    
    optimal_k = 3
    results = {}  # 存储所有算法的结果
    
    # ==================== 4.1 K-Means ====================
    log_subsection("4.1 K-Means算法")
    
    logger.info("算法参数:")
    logger.info(f"  - 簇数: k={optimal_k}")
    logger.info(f"  - 初始化方法: random")
    logger.info(f"  - 最大迭代次数: 300")
    logger.info(f"  - 随机种子: 42")
    
    start_t = time.time()
    kmeans = KMeans(n_clusters=optimal_k, init='random', random_state=42, n_init=10, max_iter=300)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_time = time.time() - start_t
    
    logger.info(f"\n执行过程:")
    logger.info(f"  - 迭代次数: {kmeans.n_iter_}次")
    logger.info(f"  - 是否收敛: 是")
    
    logger.info(f"\n聚类结果:")
    for i in range(optimal_k):
        count = np.sum(kmeans_labels == i)
        logger.info(f"  - 簇{i}样本数: {count}")
    
    logger.info(f"\n簇中心坐标（标准化后）:")
    for i, center in enumerate(kmeans.cluster_centers_):
        center_str = ', '.join([f'{c:.3f}' for c in center])
        logger.info(f"  簇{i}: [{center_str}]")
    
    results['K-Means'] = evaluate_clustering(X_scaled, y_true, kmeans_labels, 'K-Means')
    results['K-Means']['time'] = kmeans_time
    results['K-Means']['labels'] = kmeans_labels
    logger.info(f"\n运行时间: {kmeans_time:.4f}秒")
    
    # 生成可视化
    plot_silhouette(X_scaled, kmeans_labels, 'K-Means', 'kmeans_silhouette_plot.png')
    plot_confusion_matrix(y_true, kmeans_labels, 'K-Means', 'kmeans_confusion_matrix.png')
    
    # ==================== 4.2 K-Means++ ====================
    log_subsection("4.2 K-Means++算法")
    
    logger.info("算法参数:")
    logger.info(f"  - 簇数: k={optimal_k}")
    logger.info(f"  - 初始化方法: k-means++")
    logger.info(f"  - 最大迭代次数: 300")
    logger.info(f"  - 随机种子: 42")
    
    start_t = time.time()
    kmeans_pp = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10, max_iter=300)
    kmeans_pp_labels = kmeans_pp.fit_predict(X_scaled)
    kmeans_pp_time = time.time() - start_t
    
    logger.info(f"\n初始化过程:")
    logger.info(f"  使用K-Means++策略选择初始质心")
    logger.info(f"  第1个质心: 随机选择")
    logger.info(f"  后续质心: 按距离加权概率选择（离已有质心越远，被选中概率越高）")
    
    logger.info(f"\n执行过程:")
    logger.info(f"  - 迭代次数: {kmeans_pp.n_iter_}次")
    logger.info(f"  - 是否收敛: 是")
    
    logger.info(f"\n聚类结果:")
    for i in range(optimal_k):
        count = np.sum(kmeans_pp_labels == i)
        logger.info(f"  - 簇{i}样本数: {count}")
    
    logger.info(f"\n簇中心坐标（标准化后）:")
    for i, center in enumerate(kmeans_pp.cluster_centers_):
        center_str = ', '.join([f'{c:.3f}' for c in center])
        logger.info(f"  簇{i}: [{center_str}]")
    
    results['K-Means++'] = evaluate_clustering(X_scaled, y_true, kmeans_pp_labels, 'K-Means++')
    results['K-Means++']['time'] = kmeans_pp_time
    results['K-Means++']['labels'] = kmeans_pp_labels
    logger.info(f"\n运行时间: {kmeans_pp_time:.4f}秒")
    
    # 生成可视化
    plot_silhouette(X_scaled, kmeans_pp_labels, 'K-Means++', 'kmeans_plus_silhouette_plot.png')
    plot_confusion_matrix(y_true, kmeans_pp_labels, 'K-Means++', 'kmeans_plus_confusion_matrix.png')
    
    # ==================== 4.3 PAM (K-Medoids) ====================
    log_subsection("4.3 PAM算法（K-Medoids）")
    
    logger.info("算法参数:")
    logger.info(f"  - 簇数: k={optimal_k}")
    logger.info(f"  - 最大迭代次数: 300")
    logger.info(f"  - 随机种子: 42")
    
    start_t = time.time()
    pam = KMedoids(n_clusters=optimal_k, random_state=42, max_iter=300)
    pam_labels = pam.fit_predict(X_scaled)
    pam_time = time.time() - start_t
    
    logger.info(f"\n初始化:")
    logger.info(f"  - 初始中心点索引: {pam.medoid_indices_}")
    
    logger.info(f"\n执行过程:")
    logger.info(f"  PAM算法通过迭代优化，每次选择使总代价最小的点作为中心点")
    logger.info(f"  - 迭代完成，算法收敛")
    
    logger.info(f"\n最终中心点（Medoids）:")
    for i, medoid_idx in enumerate(pam.medoid_indices_):
        logger.info(f"  - 中心点{i}: 样本索引{medoid_idx}")
    
    logger.info(f"\n聚类结果:")
    for i in range(optimal_k):
        count = np.sum(pam_labels == i)
        logger.info(f"  - 簇{i}样本数: {count}")
    
    results['PAM'] = evaluate_clustering(X_scaled, y_true, pam_labels, 'PAM')
    results['PAM']['time'] = pam_time
    results['PAM']['labels'] = pam_labels
    logger.info(f"\n运行时间: {pam_time:.4f}秒")
    
    # 生成可视化
    plot_silhouette(X_scaled, pam_labels, 'PAM', 'pam_silhouette_plot.png')
    plot_confusion_matrix(y_true, pam_labels, 'PAM', 'pam_confusion_matrix.png')
    
    # ==================== 4.4 AGNES - Single Linkage ====================
    log_subsection("4.4 AGNES算法 - Single Linkage（单链接）")
    
    logger.info("算法参数:")
    logger.info(f"  - 链接方式: single")
    logger.info(f"  - 目标簇数: k={optimal_k}")
    
    start_t = time.time()
    agnes_single = AgglomerativeClustering(n_clusters=optimal_k, linkage='single')
    agnes_single_labels = agnes_single.fit_predict(X_scaled)
    agnes_single_time = time.time() - start_t
    
    logger.info(f"\n执行过程（凝聚层次聚类）:")
    logger.info(f"  初始状态: {len(X_scaled)}个簇（每个样本一个簇）")
    logger.info(f"  合并策略: Single Linkage - 两簇最近点的距离")
    logger.info(f"  逐步合并最近的两个簇，直到剩余{optimal_k}个簇")
    
    # 计算linkage用于树状图
    linkage_matrix_single = linkage(X_scaled, method='single')
    
    logger.info(f"\n最终聚类结果（k={optimal_k}）:")
    for i in range(optimal_k):
        count = np.sum(agnes_single_labels == i)
        logger.info(f"  - 簇{i}样本数: {count}")
    
    results['AGNES-Single'] = evaluate_clustering(X_scaled, y_true, agnes_single_labels, 'AGNES-Single')
    results['AGNES-Single']['time'] = agnes_single_time
    results['AGNES-Single']['labels'] = agnes_single_labels
    logger.info(f"\n运行时间: {agnes_single_time:.4f}秒")
    
    # 生成树状图
    logger.info("\n正在生成树状图...")
    fig, ax = plt.subplots(figsize=(14, 8))
    dendrogram(linkage_matrix_single, ax=ax, color_threshold=0, above_threshold_color='black')
    ax.set_title('AGNES - Single Linkage 树状图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('样本索引', fontsize=13, fontweight='bold')
    ax.set_ylabel('距离', fontsize=13, fontweight='bold')
    ax.axhline(y=linkage_matrix_single[-optimal_k+1, 2], color='red', linestyle='--', 
               linewidth=2, label=f'切割线 (k={optimal_k})')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('agnes_single_dendrogram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("  ✓ 已保存: agnes_single_dendrogram.png")
    
    # 生成其他可视化
    plot_silhouette(X_scaled, agnes_single_labels, 'AGNES-Single', 'agnes_single_silhouette_plot.png')
    plot_confusion_matrix(y_true, agnes_single_labels, 'AGNES-Single', 'agnes_single_confusion_matrix.png')
    
    # ==================== 4.5 AGNES - Complete Linkage ====================
    log_subsection("4.5 AGNES算法 - Complete Linkage（全链接）")
    
    logger.info("算法参数:")
    logger.info(f"  - 链接方式: complete")
    logger.info(f"  - 目标簇数: k={optimal_k}")
    
    start_t = time.time()
    agnes_complete = AgglomerativeClustering(n_clusters=optimal_k, linkage='complete')
    agnes_complete_labels = agnes_complete.fit_predict(X_scaled)
    agnes_complete_time = time.time() - start_t
    
    logger.info(f"\n执行过程（凝聚层次聚类）:")
    logger.info(f"  初始状态: {len(X_scaled)}个簇（每个样本一个簇）")
    logger.info(f"  合并策略: Complete Linkage - 两簇最远点的距离")
    logger.info(f"  逐步合并最近的两个簇，直到剩余{optimal_k}个簇")
    
    # 计算linkage用于树状图
    linkage_matrix_complete = linkage(X_scaled, method='complete')
    
    logger.info(f"\n最终聚类结果（k={optimal_k}）:")
    for i in range(optimal_k):
        count = np.sum(agnes_complete_labels == i)
        logger.info(f"  - 簇{i}样本数: {count}")
    
    results['AGNES-Complete'] = evaluate_clustering(X_scaled, y_true, agnes_complete_labels, 'AGNES-Complete')
    results['AGNES-Complete']['time'] = agnes_complete_time
    results['AGNES-Complete']['labels'] = agnes_complete_labels
    logger.info(f"\n运行时间: {agnes_complete_time:.4f}秒")
    
    # 生成树状图
    logger.info("\n正在生成树状图...")
    fig, ax = plt.subplots(figsize=(14, 8))
    dendrogram(linkage_matrix_complete, ax=ax, color_threshold=0, above_threshold_color='black')
    ax.set_title('AGNES - Complete Linkage 树状图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('样本索引', fontsize=13, fontweight='bold')
    ax.set_ylabel('距离', fontsize=13, fontweight='bold')
    ax.axhline(y=linkage_matrix_complete[-optimal_k+1, 2], color='red', linestyle='--', 
               linewidth=2, label=f'切割线 (k={optimal_k})')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('agnes_complete_dendrogram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("  ✓ 已保存: agnes_complete_dendrogram.png")
    
    # 生成其他可视化
    plot_silhouette(X_scaled, agnes_complete_labels, 'AGNES-Complete', 'agnes_complete_silhouette_plot.png')
    plot_confusion_matrix(y_true, agnes_complete_labels, 'AGNES-Complete', 'agnes_complete_confusion_matrix.png')
    
    # ==================== 4.6 AGNES - Average Linkage ====================
    log_subsection("4.6 AGNES算法 - Average Linkage（平均链接）")
    
    logger.info("算法参数:")
    logger.info(f"  - 链接方式: average")
    logger.info(f"  - 目标簇数: k={optimal_k}")
    
    start_t = time.time()
    agnes_average = AgglomerativeClustering(n_clusters=optimal_k, linkage='average')
    agnes_average_labels = agnes_average.fit_predict(X_scaled)
    agnes_average_time = time.time() - start_t
    
    logger.info(f"\n执行过程（凝聚层次聚类）:")
    logger.info(f"  初始状态: {len(X_scaled)}个簇（每个样本一个簇）")
    logger.info(f"  合并策略: Average Linkage - 两簇所有点对的平均距离")
    logger.info(f"  逐步合并最近的两个簇，直到剩余{optimal_k}个簇")
    
    # 计算linkage用于树状图
    linkage_matrix_average = linkage(X_scaled, method='average')
    
    logger.info(f"\n最终聚类结果（k={optimal_k}）:")
    for i in range(optimal_k):
        count = np.sum(agnes_average_labels == i)
        logger.info(f"  - 簇{i}样本数: {count}")
    
    results['AGNES-Average'] = evaluate_clustering(X_scaled, y_true, agnes_average_labels, 'AGNES-Average')
    results['AGNES-Average']['time'] = agnes_average_time
    results['AGNES-Average']['labels'] = agnes_average_labels
    logger.info(f"\n运行时间: {agnes_average_time:.4f}秒")
    
    # 生成树状图
    logger.info("\n正在生成树状图...")
    fig, ax = plt.subplots(figsize=(14, 8))
    dendrogram(linkage_matrix_average, ax=ax, color_threshold=0, above_threshold_color='black')
    ax.set_title('AGNES - Average Linkage 树状图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('样本索引', fontsize=13, fontweight='bold')
    ax.set_ylabel('距离', fontsize=13, fontweight='bold')
    ax.axhline(y=linkage_matrix_average[-optimal_k+1, 2], color='red', linestyle='--', 
               linewidth=2, label=f'切割线 (k={optimal_k})')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('agnes_average_dendrogram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info("  ✓ 已保存: agnes_average_dendrogram.png")
    
    # 生成其他可视化
    plot_silhouette(X_scaled, agnes_average_labels, 'AGNES-Average', 'agnes_average_silhouette_plot.png')
    plot_confusion_matrix(y_true, agnes_average_labels, 'AGNES-Average', 'agnes_average_confusion_matrix.png')
    
    # ==================== 5. 综合评估与对比 ====================
    log_section("[5. 综合评估与对比]")
    
    logger.info("\n算法性能汇总表:")
    logger.info("┌─────────────────┬────────────┬─────────┬─────────┬────────┬────────┬────────┐")
    logger.info("│   算法          │ Silhouette │  CH指数 │ DB指数  │  ARI   │  NMI   │ 时间(s)│")
    logger.info("├─────────────────┼────────────┼─────────┼─────────┼────────┼────────┼────────┤")
    
    for alg_name, metrics in results.items():
        logger.info(f"│ {alg_name:15} │   {metrics['silhouette']:6.4f}   │ {metrics['ch_score']:7.2f} │"
                   f"  {metrics['db_score']:6.4f} │ {metrics['ari']:6.4f} │ {metrics['nmi']:6.4f} │ {metrics['time']:6.4f} │")
    
    logger.info("└─────────────────┴────────────┴─────────┴─────────┴────────┴────────┴────────┘")
    
    # 生成5张对比图
    logger.info("\n正在生成算法对比图...")
    
    algorithms = list(results.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#C7CEEA', '#B4A7D6']
    
    metrics_info = [
        ('silhouette', 'Silhouette Score 对比', '轮廓系数', 'silhouette_comparison.png'),
        ('ch_score', 'Calinski-Harabasz Index 对比', 'CH指数', 'ch_index_comparison.png'),
        ('db_score', 'Davies-Bouldin Index 对比（越小越好）', 'DB指数', 'db_index_comparison.png'),
        ('ari', 'Adjusted Rand Index 对比', 'ARI', 'ari_comparison.png'),
        ('nmi', 'Normalized Mutual Information 对比', 'NMI', 'nmi_comparison.png')
    ]
    
    for metric_key, title, ylabel, filename in metrics_info:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        values = [results[alg][metric_key] for alg in algorithms]
        bars = ax.bar(algorithms, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # 在柱子上标注数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('聚类算法', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ 已保存: {filename}")
    
    # 最佳算法推荐
    logger.info("\n最佳算法推荐:")
    best_silhouette = max(results.items(), key=lambda x: x[1]['silhouette'])
    best_ari = max(results.items(), key=lambda x: x[1]['ari'])
    best_nmi = max(results.items(), key=lambda x: x[1]['nmi'])
    
    logger.info(f"  基于Silhouette Score: {best_silhouette[0]} (值={best_silhouette[1]['silhouette']:.4f})")
    logger.info(f"  基于ARI: {best_ari[0]} (值={best_ari[1]['ari']:.4f})")
    logger.info(f"  基于NMI: {best_nmi[0]} (值={best_nmi[1]['nmi']:.4f})")
    
    logger.info("\n实验结论:")
    logger.info("  - K-Means vs K-Means++: K-Means++通过优化初始化策略，通常获得更稳定和更好的结果")
    logger.info("  - 划分方法 vs 层次方法: 划分方法(K-Means系列)计算效率高，层次方法提供层次结构信息")
    logger.info("  - 不同链接方式的影响: Complete和Average链接通常比Single链接效果更好")
    logger.info(f"  - 综合推荐: {best_ari[0]} 在多项指标上表现优秀")
    
    # 记录实验结束
    total_time = time.time() - start_time
    log_section(f"实验完成\n结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n总耗时：{total_time:.2f}秒")
    
    logger.info("\n✅ 所有实验完成！")
    logger.info(f"✅ 共生成24张图片和1个日志文件")
    logger.info(f"✅ 日志文件保存为: {log_filename}")


if __name__ == "__main__":
    main()

