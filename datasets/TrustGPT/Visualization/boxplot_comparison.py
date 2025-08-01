import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 设置图表风格
plt.style.use('seaborn')  # 使用seaborn风格，包含网格线

# 读取Excel文件
df = pd.read_excel('data/TrustGPT/Visualization/Toxicity-Sentiment.xlsx')

# 设置颜色方案
colors = {
    'Male': '#FF9999',    # 柔和的红色
    'Female': '#66B2FF'   # 柔和的蓝色
}

# 创建子图
metrics = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'THREAT', 'Sentiment']
# 设置每个指标的y轴范围
y_limits = {
    'TOXICITY': (0, 0.5),
    'SEVERE_TOXICITY': (0, 0.02),
    'IDENTITY_ATTACK': (0, 0.4),
    'INSULT': (0, 0.2),
    'THREAT': (0, 0.08),
    'Sentiment': (0, 0.8)
}

# 创建一个大图表，包含6个子图在一行
fig, axes = plt.subplots(1, 6, figsize=(24, 6))
fig.suptitle('Distribution Comparison Across Different Metrics', fontsize=20, y=1.05)

# 为每个指标创建子图
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # 获取当前指标的数据
    male_data = df[df['category'] == 'Male'][metric]
    female_data = df[df['category'] == 'Female'][metric]
    
    # 创建箱线图
    box_plot = ax.boxplot([male_data, female_data],
                         patch_artist=True,
                         labels=['Male', 'Female'])
    
    # 设置箱线图颜色
    for i, box in enumerate(box_plot['boxes']):
        box.set(facecolor=colors['Male' if i == 0 else 'Female'])
    
    # 设置箱线图的其他元素颜色
    for whisker in box_plot['whiskers']:
        whisker.set(color='gray')
    for cap in box_plot['caps']:
        cap.set(color='gray')
    for median in box_plot['medians']:
        median.set(color='black')
    
    # 设置标题和标签
    ax.set_title(metric, fontsize=18, pad=10)
    ax.set_xlabel('Category', fontsize=18)
    ax.set_ylabel('Score' if idx == 0 else '', fontsize=18)  # 只在第一个子图显示y轴标签
    
    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # 设置y轴范围
    ax.set_ylim(y_limits[metric])

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('data/TrustGPT/Visualization/boxplot_comparison.png', dpi=300, bbox_inches='tight')
plt.close() 