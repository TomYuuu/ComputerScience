import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据和元数据
df = pd.read_csv("E:\PythonCode\BADM550_NEW_PROJECT\data\indexes_dataset.csv")
feature_metadata = pd.read_csv("E:\PythonCode\BADM550_NEW_PROJECT\data\metadata_with_index.csv")
index_metadata = pd.read_csv("E:\PythonCode\BADM550_NEW_PROJECT\data\indexes_metadata.csv")

# 创建映射字典
feature_desc = dict(zip(feature_metadata['column_name'], feature_metadata['description']))
index_desc = dict(zip(index_metadata['index_name'], index_metadata['description']))

# 定义各指数对应的参数
INDEX_MAPPING = {
    "wellness_index": [col for col in df.columns if col not in ["CommunityAreaName", "EVI", "SSI", "EOI"]],
    "EVI": ["TRAFVOL", "AIRQUAL", "WATQUAL", "HEATVUL", "NOISEP"],
    "SSI": ["SCHLRAT", "HSDROP", "ASPROG", "SOCASSO", "VCRIME", "PCRIME", "ERTIME", "VSSERV"],
    "EOI": ["HOUSBURD", "MEDINC", "UNEMPLOY", "CHILCBD", "COLLACC", "BROADND"]
}

# 初始化页面设置
st.set_page_config(layout="wide")
st.title("Multi-Dimensional Community Index Dashboard")

# 侧边栏控件
with st.sidebar:
    st.header("Configuration")

    # 指数选择
    selected_index = st.selectbox(
        "Select Index",
        options=["wellness_index", "EVI", "SSI", "EOI"],
        format_func=lambda x: f"{x} - {index_desc[x]}" if x != "wellness_index" else x
    )

    # 获取可用参数
    available_features = INDEX_MAPPING[selected_index]
    feature_options = [(f, feature_desc[f]) for f in available_features]

    # 第一个特征选择
    col1_desc = st.selectbox(
        "Select Primary Feature",
        options=feature_options,
        format_func=lambda x: x[1]
    )
    col1 = col1_desc[0]

    # 第二个特征选择（排除已选）
    remaining_features = [f for f in feature_options if f[0] != col1]
    col2_desc = st.selectbox(
        "Select Secondary Feature",
        options=remaining_features,
        format_func=lambda x: x[1]
    )
    col2 = col2_desc[0]


# ================== 图表绘制函数 ==================
def get_stats(data):
    """获取完整的五数概括"""
    return {
        'Min': data.min(),
        'Q1': data.quantile(0.25),
        'Mean': data.mean(),
        'Q3': data.quantile(0.75),
        'Max': data.max()
    }


def plot_full_stats(ax, data, title, chart_type='bar'):
    """通用统计图表绘制"""
    stats = get_stats(data)
    metrics = list(stats.keys())
    values = list(stats.values())

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    if chart_type == 'bar':
        bars = ax.bar(metrics, values, color=colors)
        ax.set_ylabel('Value')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{height:.2f}", ha='center', va='bottom')

    elif chart_type == 'box':
        box = ax.boxplot(data, vert=False, patch_artist=True,
                         boxprops=dict(facecolor='#2ca02c'),
                         whiskerprops=dict(color='#2ca02c'),
                         capprops=dict(color='#2ca02c'),
                         medianprops=dict(color='red'))

        # 添加统计标注
        stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))

        ax.set_yticks([])

    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)


def plot_dual_comparison(ax, feature_data, index_data, feature_name, index_name):
    """双轴对比图（特征柱状图 + 指数折线图）"""
    # 获取完整统计
    feature_stats = get_stats(feature_data)
    index_stats = get_stats(index_data)

    # 柱状图参数
    metrics = list(feature_stats.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 绘制特征柱状图
    bars = ax.bar(metrics, feature_stats.values(), color=colors, alpha=0.7)

    # 创建双轴
    ax2 = ax.twinx()

    # 绘制指数折线图
    line, = ax2.plot(metrics, index_stats.values(),
                     color='black', marker='o',
                     linewidth=2, markersize=8)

    # 设置标签
    ax.set_title(f"{feature_name} vs {index_name}", fontsize=12)
    ax.set_ylabel(feature_name)
    ax2.set_ylabel(index_name, rotation=-90, labelpad=15)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f"{height:.2f}", ha='center', va='bottom')

    for x, y in zip(metrics, index_stats.values()):
        ax2.text(x, y, f"{y:.2f}", ha='center', va='bottom')

    # 添加图例
    ax.legend([bars[0], line],
              [feature_name, index_name],
              loc='upper left')


# ================== 主界面布局 ==================
# 第一行：指数统计（五数柱状图）
st.header(f"{selected_index} Analysis")
fig1, ax1 = plt.subplots(figsize=(10, 4))
plot_full_stats(ax1, df[selected_index],
                f"{selected_index} Statistics\n{index_desc.get(selected_index, '')}")
plt.tight_layout()
st.pyplot(fig1)

# 第二行：特征分布（箱线图）
st.header("Feature Distribution Analysis")
col_box1, col_box2 = st.columns(2)

with col_box1:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    plot_full_stats(ax2, df[col1],
                    f"{col1}\n{feature_desc[col1][:50]}...",
                    chart_type='box')
    plt.tight_layout()
    st.pyplot(fig2)

with col_box2:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    plot_full_stats(ax3, df[col2],
                    f"{col2}\n{feature_desc[col2][:50]}...",
                    chart_type='box')
    plt.tight_layout()
    st.pyplot(fig3)

# 第三行：双参数对比
st.header("Feature-Index Comparison")
col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    plot_dual_comparison(ax4, df[col1], df[selected_index],
                         feature_name=f"{col1}\n{feature_desc[col1][:30]}...",
                         index_name=selected_index)
    plt.tight_layout()
    st.pyplot(fig4)

with col_comp2:
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    plot_dual_comparison(ax5, df[col2], df[selected_index],
                         feature_name=f"{col2}\n{feature_desc[col2][:30]}...",
                         index_name=selected_index)
    plt.tight_layout()
    st.pyplot(fig5)

# 数据说明面板
with st.expander("Data Descriptions"):
    st.subheader("Feature Descriptions")
    st.dataframe(feature_metadata[["column_name", "description"]])

    st.subheader("Index Descriptions")
    st.dataframe(index_metadata)
