"""
生成适合DBSCAN算法的形状刁钻的数据集
包含：月牙形、同心圆、S形曲线和噪声点
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
import pandas as pd
from matplotlib import rcParams

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']  # 优先使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
np.random.seed(42)

# ============= 生成多种形状的数据 =============

print("正在生成DBSCAN测试数据集...")

# 1. 月牙形数据（两个交叉的月牙）
moons_X, moons_y = make_moons(n_samples=300, noise=0.05, random_state=42)

# 2. 同心圆数据（两个圆环）
circles_X, circles_y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# 3. S形曲线数据（自定义生成）
n_samples = 300
t = np.linspace(0, 4*np.pi, n_samples)
s_curve_X = np.column_stack([
    t,
    np.sin(t) + np.random.normal(0, 0.1, n_samples)
])
s_curve_y = (t > 2*np.pi).astype(int)

# 4. 复杂混合数据：包含不规则形状 + 噪声点
# 生成三个不同密度的团块
blob1 = np.random.randn(150, 2) * 0.3 + np.array([0, 0])
blob2 = np.random.randn(100, 2) * 0.2 + np.array([3, 3])
blob3 = np.random.randn(120, 2) * 0.25 + np.array([-2, 3])

# 生成噪声点（均匀分布）
noise_points = np.random.uniform(-5, 5, (50, 2))

# 合并数据
complex_X = np.vstack([blob1, blob2, blob3, noise_points])
complex_y = np.hstack([
    np.zeros(150),
    np.ones(100),
    np.full(120, 2),
    np.full(50, -1)  # 噪声点标记为-1
])

# ============= 保存数据集 =============

# 保存为CSV文件
datasets = {
    'dbscan_moons': (moons_X, moons_y),
    'dbscan_circles': (circles_X, circles_y),
    'dbscan_s_curve': (s_curve_X, s_curve_y),
    'dbscan_complex': (complex_X, complex_y)
}

for name, (X, y) in datasets.items():
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    df['true_label'] = y.astype(int)
    df.to_csv(f'{name}.csv', index=False)
    print(f"✓ 已保存: {name}.csv (样本数: {len(X)})")

# ============= 为每个数据集生成独立的可视化图片 =============

print("\n正在生成可视化图片...")

# 定义配色方案（使用更美观的颜色）
colors_2class = ['#FF6B6B', '#4ECDC4']  # 红色和青色
colors_multi = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D']  # 多类别配色

datasets_list = [
    {
        'title': '月牙形数据集（Moons Dataset）',
        'filename': 'dbscan_moons_visualization.png',
        'X': moons_X,
        'y': moons_y,
        'description': '两个交叉的月牙状簇，K-Means无法识别此非凸形状',
        'params': 'eps=0.3, min_samples=5'
    },
    {
        'title': '同心圆数据集（Circles Dataset）',
        'filename': 'dbscan_circles_visualization.png',
        'X': circles_X,
        'y': circles_y,
        'description': '内外两个同心圆环，K-Means会错误切割圆环',
        'params': 'eps=0.2, min_samples=5'
    },
    {
        'title': 'S形曲线数据集（S-Curve Dataset）',
        'filename': 'dbscan_s_curve_visualization.png',
        'X': s_curve_X,
        'y': s_curve_y,
        'description': '两条S形曲线簇，展示DBSCAN处理曲线形状的能力',
        'params': 'eps=0.5, min_samples=5'
    },
    {
        'title': '复杂混合数据集（Complex Dataset with Noise）',
        'filename': 'dbscan_complex_visualization.png',
        'X': complex_X,
        'y': complex_y,
        'description': '包含3个不同密度的簇和50个噪声点，适合参数敏感性分析',
        'params': 'eps=0.5, min_samples=10'
    }
]

for dataset_info in datasets_list:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    X = dataset_info['X']
    y = dataset_info['y']
    
    # 为不同类别使用不同颜色
    unique_labels = np.unique(y)
    
    if len(unique_labels) <= 2:
        colors = colors_2class
    else:
        colors = colors_multi
    
    # 绘制每个类别
    for idx, label in enumerate(unique_labels):
        mask = y == label
        if label == -1:  # 噪声点
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c='gray', marker='x', s=80, alpha=0.6,
                      label='噪声点', linewidths=2, zorder=1)
        else:
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=colors[int(label) % len(colors)], 
                      s=60, alpha=0.8, edgecolors='white', linewidth=1.5,
                      label=f'簇 {int(label)}', zorder=2)
    
    # 设置标题和标签
    ax.set_title(dataset_info['title'], fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('特征 1', fontsize=13, fontweight='bold')
    ax.set_ylabel('特征 2', fontsize=13, fontweight='bold')
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor='black')
    
    # 添加说明文字
    textstr = f'数据特点：{dataset_info["description"]}\n推荐DBSCAN参数：{dataset_info["params"]}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.5, -0.15, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center', bbox=props,
            wrap=True)
    
    # 设置坐标轴样式
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(dataset_info['filename'], dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 已保存: {dataset_info['filename']}")
    plt.close()

print("\n✓ 所有可视化图片生成完成！")

# ============= 数据集统计信息 =============

print("\n" + "="*60)
print("数据集统计信息：")
print("="*60)

for name, (X, y) in datasets.items():
    print(f"\n【{name}】")
    print(f"  样本数量: {len(X)}")
    print(f"  特征数量: {X.shape[1]}")
    print(f"  类别数量: {len(np.unique(y))}")
    print(f"  类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  数据范围: X1[{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], "
          f"X2[{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")

print("\n" + "="*60)
print("📊 数据集特点说明：")
print("="*60)
print("""
1. 【月牙形数据】
   - 两个交叉的月牙状簇
   - K-Means会失败（无法识别非凸形状）
   - DBSCAN可以完美识别
   - 推荐参数：eps=0.3, min_samples=5

2. 【同心圆数据】
   - 两个同心圆环
   - K-Means会将圆环切割
   - DBSCAN可以识别内外圆环
   - 推荐参数：eps=0.2, min_samples=5

3. 【S形曲线数据】
   - 两条S形曲线簇
   - 适合展示DBSCAN处理曲线簇的能力
   - 推荐参数：eps=0.5, min_samples=5

4. 【复杂混合数据】
   - 包含3个不同密度的簇 + 噪声点
   - 适合测试参数敏感性
   - 可以展示DBSCAN的噪声识别能力
   - 推荐参数：eps=0.5, min_samples=10
""")

print("\n" + "="*60)
print("✅ 所有数据集和可视化图片生成完成！")
print("="*60)
print("\n生成的文件列表：")
print("📁 数据文件（CSV）：")
print("   - dbscan_moons.csv")
print("   - dbscan_circles.csv")
print("   - dbscan_s_curve.csv")
print("   - dbscan_complex.csv")
print("\n📊 可视化图片（PNG）：")
print("   - dbscan_moons_visualization.png")
print("   - dbscan_circles_visualization.png")
print("   - dbscan_s_curve_visualization.png")
print("   - dbscan_complex_visualization.png")
print("\n💡 推荐实验流程：")
print("   1. 先在月牙形数据上对比K-Means vs DBSCAN")
print("   2. 在复杂混合数据上探究eps和min_samples的影响")
print("   3. 可视化不同参数下的聚类结果")

