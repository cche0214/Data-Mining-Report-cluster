"""
DBSCAN算法实验
包含：DBSCAN vs K-Means对比、参数影响探究
作者：数据挖掘课程实验
日期：2024-11-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist
import warnings
import logging
import time
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
log_filename = 'dbscan_experiment.log'
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
    """计算Hopkins统计量"""
    if n_samples is None:
        n_samples = min(int(0.1 * len(X)), 100)
    
    n_features = X.shape[1]
    n = len(X)
    
    sample_indices = np.random.choice(n, n_samples, replace=False)
    X_sample = X[sample_indices]
    X_random = np.random.uniform(X.min(axis=0), X.max(axis=0), (n_samples, n_features))
    
    dist_sample = cdist(X_sample, X, metric='euclidean')
    dist_sample.sort(axis=1)
    u = dist_sample[:, 1].sum()
    
    dist_random = cdist(X_random, X, metric='euclidean')
    dist_random.sort(axis=1)
    w = dist_random[:, 0].sum()
    
    H = w / (u + w)
    return H


def plot_comparison_dbscan_kmeans(X, y_true, dbscan_labels, kmeans_labels, 
                                  dataset_name, filename):
    """绘制DBSCAN vs K-Means对比图（左右对比）"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 获取唯一标签
    dbscan_unique = np.unique(dbscan_labels)
    kmeans_unique = np.unique(kmeans_labels)
    
    # 配色方案
    colors_dbscan = plt.cm.tab10(np.linspace(0, 1, len(dbscan_unique)))
    colors_kmeans = plt.cm.tab10(np.linspace(0, 1, len(kmeans_unique)))
    
    # 左图：DBSCAN
    ax_left = axes[0]
    for idx, label in enumerate(dbscan_unique):
        if label == -1:  # 噪声点
            mask = dbscan_labels == label
            ax_left.scatter(X[mask, 0], X[mask, 1], 
                          c='gray', marker='x', s=80, alpha=0.7,
                          label='噪声点', linewidths=2, zorder=1)
        else:
            mask = dbscan_labels == label
            ax_left.scatter(X[mask, 0], X[mask, 1], 
                          c=[colors_dbscan[idx]], s=60, alpha=0.8,
                          edgecolors='white', linewidth=1.5,
                          label=f'簇 {label}', zorder=2)
    
    ax_left.set_title(f'DBSCAN聚类结果', fontsize=16, fontweight='bold', pad=15)
    ax_left.set_xlabel('特征 1', fontsize=13, fontweight='bold')
    ax_left.set_ylabel('特征 2', fontsize=13, fontweight='bold')
    ax_left.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_left.grid(True, alpha=0.3, linestyle='--')
    ax_left.set_aspect('equal')
    
    # 右图：K-Means
    ax_right = axes[1]
    for idx, label in enumerate(kmeans_unique):
        mask = kmeans_labels == label
        ax_right.scatter(X[mask, 0], X[mask, 1], 
                        c=[colors_kmeans[idx]], s=60, alpha=0.8,
                        edgecolors='white', linewidth=1.5,
                        label=f'簇 {label}')
    
    ax_right.set_title(f'K-Means聚类结果 (k={len(kmeans_unique)})', 
                      fontsize=16, fontweight='bold', pad=15)
    ax_right.set_xlabel('特征 1', fontsize=13, fontweight='bold')
    ax_right.set_ylabel('特征 2', fontsize=13, fontweight='bold')
    ax_right.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_right.grid(True, alpha=0.3, linestyle='--')
    ax_right.set_aspect('equal')
    
    plt.suptitle(f'{dataset_name}数据集 - DBSCAN vs K-Means对比', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  ✓ 已保存对比图: {filename}")


def plot_parameter_effect(X, y_true, param_values, param_name, fixed_param_name, fixed_param_value,
                         dataset_name, filename):
    """绘制参数影响图（多子图）"""
    n_params = len(param_values)
    fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 6))
    
    if n_params == 1:
        axes = [axes]
    
    for idx, param_value in enumerate(param_values):
        ax = axes[idx]
        
        # 设置DBSCAN参数
        if param_name == 'eps':
            dbscan = DBSCAN(eps=param_value, min_samples=fixed_param_value)
        else:  # min_samples
            dbscan = DBSCAN(eps=fixed_param_value, min_samples=param_value)
        
        labels = dbscan.fit_predict(X)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)
        
        # 配色
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # 绘制聚类结果
        for label_idx, label in enumerate(unique_labels):
            if label == -1:
                mask = labels == label
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c='gray', marker='x', s=80, alpha=0.7,
                          label='噪声点', linewidths=2, zorder=1)
            else:
                mask = labels == label
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c=[colors[label_idx]], s=60, alpha=0.8,
                          edgecolors='white', linewidth=1.5,
                          label=f'簇 {label}', zorder=2)
        
        # 设置标题和标签
        if param_name == 'eps':
            title = f'{param_name}={param_value}\n{fixed_param_name}={fixed_param_value}\n簇数={n_clusters}, 噪声={n_noise}'
        else:
            title = f'{fixed_param_name}={fixed_param_value}\n{param_name}={param_value}\n簇数={n_clusters}, 噪声={n_noise}'
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('特征 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('特征 2', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
    
    plt.suptitle(f'{dataset_name}数据集 - {param_name}参数影响', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  ✓ 已保存参数影响图: {filename}")


# ===================== 主实验流程 =====================

def main():
    """主实验函数"""
    
    start_time = time.time()
    log_section(f"DBSCAN算法实验日志\n实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ==================== 1. 数据加载 ====================
    log_section("[1. 数据加载与预处理]")
    
    datasets_info = {
        '月牙形': {
            'file': '../DBSCAN/dbscan_moons.csv',
            'recommended_eps': 0.3,
            'recommended_min_samples': 5,
            'kmeans_k': 2
        },
        '同心圆': {
            'file': '../DBSCAN/dbscan_circles.csv',
            'recommended_eps': 0.2,
            'recommended_min_samples': 5,
            'kmeans_k': 2
        },
        'S形曲线': {
            'file': '../DBSCAN/dbscan_s_curve.csv',
            'recommended_eps': 0.5,
            'recommended_min_samples': 5,
            'kmeans_k': 2
        },
        '复杂混合': {
            'file': '../DBSCAN/dbscan_complex.csv',
            'recommended_eps': 0.5,
            'recommended_min_samples': 10,
            'kmeans_k': 3
        }
    }
    
    datasets = {}
    logger.info("正在加载数据集...")
    
    for name, info in datasets_info.items():
        df = pd.read_csv(info['file'])
        X = df[['feature1', 'feature2']].values
        y_true = df['true_label'].values
        
        # 计算Hopkins统计量
        H = hopkins_statistic(X)
        
        datasets[name] = {
            'X': X,
            'y_true': y_true,
            'hopkins': H,
            'recommended_eps': info['recommended_eps'],
            'recommended_min_samples': info['recommended_min_samples'],
            'kmeans_k': info['kmeans_k']
        }
        
        logger.info(f"\n{name}数据集:")
        logger.info(f"  - 样本数量: {len(X)}")
        logger.info(f"  - Hopkins统计量: H = {H:.4f}")
        logger.info(f"  - 推荐参数: eps={info['recommended_eps']}, min_samples={info['recommended_min_samples']}")
    
    # ==================== 2. DBSCAN vs K-Means对比 ====================
    log_section("[2. DBSCAN vs K-Means对比实验]")
    log_subsection("目标：验证DBSCAN在非凸形状数据上的优势")
    
    comparison_datasets = ['月牙形', '同心圆', 'S形曲线']
    
    for dataset_name in comparison_datasets:
        log_subsection(f"2.{comparison_datasets.index(dataset_name)+1} {dataset_name}数据集")
        
        X = datasets[dataset_name]['X']
        y_true = datasets[dataset_name]['y_true']
        eps = datasets[dataset_name]['recommended_eps']
        min_samples = datasets[dataset_name]['recommended_min_samples']
        k = datasets[dataset_name]['kmeans_k']
        
        logger.info(f"\n数据集: {dataset_name}")
        logger.info(f"  - 样本数: {len(X)}")
        logger.info(f"  - 真实簇数: {len(np.unique(y_true))}")
        
        # 运行DBSCAN
        logger.info(f"\n运行DBSCAN算法:")
        logger.info(f"  - 参数: eps={eps}, min_samples={min_samples}")
        
        start_t = time.time()
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X)
        dbscan_time = time.time() - start_t
        
        n_clusters_dbscan = len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise_dbscan = np.sum(dbscan_labels == -1)
        
        logger.info(f"  - 识别簇数: {n_clusters_dbscan}")
        logger.info(f"  - 噪声点数: {n_noise_dbscan}")
        logger.info(f"  - 运行时间: {dbscan_time:.4f}秒")
        
        # 运行K-Means
        logger.info(f"\n运行K-Means算法:")
        logger.info(f"  - 参数: k={k}")
        
        start_t = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        kmeans_time = time.time() - start_t
        
        logger.info(f"  - 运行时间: {kmeans_time:.4f}秒")
        
        # 评估指标
        # DBSCAN评估（排除噪声点）
        if n_clusters_dbscan > 1:
            mask_dbscan = dbscan_labels != -1
            if np.sum(mask_dbscan) > 1:
                silhouette_dbscan = silhouette_score(X[mask_dbscan], dbscan_labels[mask_dbscan])
            else:
                silhouette_dbscan = -1
        else:
            silhouette_dbscan = -1
        
        nmi_dbscan = normalized_mutual_info_score(y_true, dbscan_labels)
        
        # K-Means评估
        silhouette_kmeans = silhouette_score(X, kmeans_labels)
        nmi_kmeans = normalized_mutual_info_score(y_true, kmeans_labels)
        
        logger.info(f"\n评估指标对比:")
        logger.info(f"  DBSCAN:")
        logger.info(f"    - 轮廓系数: {silhouette_dbscan:.4f}")
        logger.info(f"    - NMI: {nmi_dbscan:.4f}")
        logger.info(f"  K-Means:")
        logger.info(f"    - 轮廓系数: {silhouette_kmeans:.4f}")
        logger.info(f"    - NMI: {nmi_kmeans:.4f}")
        logger.info(f"  优势: DBSCAN的NMI比K-Means高 {nmi_dbscan - nmi_kmeans:.4f}")
        
        # 生成对比图
        filename = f'dbscan_vs_kmeans_{dataset_name}.png'
        plot_comparison_dbscan_kmeans(X, y_true, dbscan_labels, kmeans_labels, 
                                     dataset_name, filename)
    
    # ==================== 3. 参数影响探究 ====================
    log_section("[3. DBSCAN参数影响探究]")
    log_subsection("目标：分析eps和min_samples对聚类结果的影响")
    
    dataset_name = '复杂混合'
    X = datasets[dataset_name]['X']
    y_true = datasets[dataset_name]['y_true']
    
    logger.info(f"\n使用数据集: {dataset_name}")
    logger.info(f"  - 样本数: {len(X)}")
    logger.info(f"  - 真实簇数: {len(np.unique(y_true[y_true >= 0]))}")
    logger.info(f"  - 真实噪声点数: {np.sum(y_true == -1)}")
    
    # 3.1 eps参数影响（固定min_samples=10）
    log_subsection("3.1 eps参数影响（固定min_samples=10）")
    
    fixed_min_samples = 10
    eps_values = [0.3, 0.5, 0.7]
    
    logger.info(f"\n固定参数: min_samples={fixed_min_samples}")
    logger.info(f"变化参数: eps={eps_values}")
    
    results_eps = []
    for eps in eps_values:
        logger.info(f"\n  eps={eps}:")
        dbscan = DBSCAN(eps=eps, min_samples=fixed_min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        # 评估（排除噪声点）
        if n_clusters > 1:
            mask = labels != -1
            if np.sum(mask) > 1:
                silhouette = silhouette_score(X[mask], labels[mask])
            else:
                silhouette = -1
        else:
            silhouette = -1
        
        nmi = normalized_mutual_info_score(y_true, labels)
        
        logger.info(f"    - 识别簇数: {n_clusters}")
        logger.info(f"    - 噪声点数: {n_noise}")
        logger.info(f"    - 轮廓系数: {silhouette:.4f}")
        logger.info(f"    - NMI: {nmi:.4f}")
        
        results_eps.append({
            'eps': eps,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': silhouette,
            'nmi': nmi,
            'labels': labels
        })
    
    # 生成eps影响图
    filename = 'dbscan_eps_effect.png'
    plot_parameter_effect(X, y_true, eps_values, 'eps', 'min_samples', fixed_min_samples,
                        dataset_name, filename)
    
    # 3.2 min_samples参数影响（固定eps=0.5）
    log_subsection("3.2 min_samples参数影响（固定eps=0.5）")
    
    fixed_eps = 0.5
    min_samples_values = [5, 10, 15]
    
    logger.info(f"\n固定参数: eps={fixed_eps}")
    logger.info(f"变化参数: min_samples={min_samples_values}")
    
    results_min_samples = []
    for min_samples in min_samples_values:
        logger.info(f"\n  min_samples={min_samples}:")
        dbscan = DBSCAN(eps=fixed_eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        # 评估（排除噪声点）
        if n_clusters > 1:
            mask = labels != -1
            if np.sum(mask) > 1:
                silhouette = silhouette_score(X[mask], labels[mask])
            else:
                silhouette = -1
        else:
            silhouette = -1
        
        nmi = normalized_mutual_info_score(y_true, labels)
        
        logger.info(f"    - 识别簇数: {n_clusters}")
        logger.info(f"    - 噪声点数: {n_noise}")
        logger.info(f"    - 轮廓系数: {silhouette:.4f}")
        logger.info(f"    - NMI: {nmi:.4f}")
        
        results_min_samples.append({
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': silhouette,
            'nmi': nmi,
            'labels': labels
        })
    
    # 生成min_samples影响图
    filename = 'dbscan_min_samples_effect.png'
    plot_parameter_effect(X, y_true, min_samples_values, 'min_samples', 'eps', fixed_eps,
                        dataset_name, filename)
    
    # ==================== 4. 实验总结 ====================
    log_section("[4. 实验总结]")
    
    logger.info("\n实验结论:")
    logger.info("1. DBSCAN vs K-Means对比:")
    logger.info("   - DBSCAN能够正确识别非凸形状（月牙形、同心圆、S形曲线）")
    logger.info("   - K-Means强制将数据划分为球形簇，在非凸数据上表现较差")
    logger.info("   - DBSCAN自动确定簇数，无需预先指定k值")
    
    logger.info("\n2. 参数影响分析:")
    logger.info("   - eps参数：")
    logger.info("     * 较小eps（0.3）：识别更多小簇，噪声点较多")
    logger.info("     * 适中eps（0.5）：平衡簇数和噪声识别")
    logger.info("     * 较大eps（0.7）：可能合并簇，噪声点减少")
    logger.info("   - min_samples参数：")
    logger.info("     * 较小min_samples（5）：识别更多簇，噪声点较少")
    logger.info("     * 适中min_samples（10）：平衡簇数和噪声识别")
    logger.info("     * 较大min_samples（15）：更严格，可能将小簇识别为噪声")
    
    logger.info("\n3. 最佳参数推荐:")
    best_eps = max(results_eps, key=lambda x: x['nmi'])
    best_min_samples = max(results_min_samples, key=lambda x: x['nmi'])
    logger.info(f"   - 基于NMI，最佳eps={best_eps['eps']}（NMI={best_eps['nmi']:.4f}）")
    logger.info(f"   - 基于NMI，最佳min_samples={best_min_samples['min_samples']}（NMI={best_min_samples['nmi']:.4f}）")
    
    total_time = time.time() - start_time
    log_section(f"实验完成\n结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n总耗时：{total_time:.2f}秒")
    
    logger.info("\n✅ 所有实验完成！")
    logger.info(f"✅ 共生成5张图片和1个日志文件")
    logger.info(f"✅ 日志文件保存为: {log_filename}")


if __name__ == "__main__":
    main()

