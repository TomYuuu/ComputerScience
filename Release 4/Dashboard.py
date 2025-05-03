import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium import Choropleth
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import json
import numpy as np

# ========== Load Data ==========
data = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/indexes_dataset.csv")
feature_metadata = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/metadata_with_index.csv")
index_metadata = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/indexes_metadata.csv")
manual_mapping = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/Completed_FineNeighborhood_to_CommunityAreaName.csv")

with open("E:/PythonCode/BADM550_NEW_PROJECT/data/chicago_neighborhoods.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ========== Metadata Mappings ==========
feature_desc = dict(zip(feature_metadata['column_name'], feature_metadata['description']))
index_desc = dict(zip(index_metadata['index_name'], index_metadata['index_description']))

INDEX_MAPPING = {
    "wellness_index": [col for col in data.columns if col not in ["CommunityAreaName", "EVI", "SSI", "EOI"]],
    "EVI": ["TRAFVOL", "AIRQUAL", "WATQUAL", "HEATVUL", "NOISEP"],
    "SSI": ["SCHLRAT", "HSDROP", "ASPROG", "SOCASSO", "VCRIME", "PCRIME", "ERTIME", "VSSERV"],
    "EOI": ["HOUSBURD", "MEDINC", "UNEMPLOY", "CHILCBD", "COLLACC", "BROADND"]
}

# ========== Streamlit UI ==========
st.set_page_config(layout="wide")
st.title("📊 Chicago Community Index Dashboard")

# ========== Sidebar ==========
with st.sidebar:
    st.header("Configuration")
    selected_index = st.selectbox(
        "Select Index",
        options=["wellness_index", "EVI", "SSI", "EOI"],
        format_func=lambda x: f"{x} - {index_desc.get(x, '')}" if x != "wellness_index" else x
    )
    available_features = INDEX_MAPPING[selected_index]
    feature_options = [(f, feature_desc.get(f, f)) for f in available_features]

    col1_desc = st.selectbox("Select Primary Feature", options=feature_options, format_func=lambda x: x[1])
    col1 = col1_desc[0]
    remaining_features = [f for f in feature_options if f[0] != col1]
    col2_desc = st.selectbox("Select Secondary Feature", options=remaining_features, format_func=lambda x: x[1])
    col2 = col2_desc[0]

    # --- Chatbot in Sidebar ---
    st.markdown("---")
    st.header("💬 Community Data Chatbot")

    def auto_detect_column(df, keyword):
        return [col for col in df.columns if keyword.lower() in col.lower()][0]

    def map_full_name_to_shortcut(user_input):
        user_input_lower = user_input.lower()
        desc_col = auto_detect_column(feature_metadata, 'description')
        name_col = auto_detect_column(feature_metadata, 'name')
        for _, row in feature_metadata.iterrows():
            if str(row[desc_col]).lower() in user_input_lower:
                return row[name_col]
        for _, row in feature_metadata.iterrows():
            if str(row[name_col]).lower() in user_input_lower:
                return row[name_col]
        return None

    def search_answer(user_input):
        user_input_lower = user_input.lower()
        index_name_col = auto_detect_column(index_metadata, 'name')
        index_desc_col = auto_detect_column(index_metadata, 'description')
        feature_name_col = auto_detect_column(feature_metadata, 'name')
        feature_desc_col = auto_detect_column(feature_metadata, 'description')

        shortcut_col = map_full_name_to_shortcut(user_input)

        # --- Check for maximum ---
        if any(keyword in user_input_lower for keyword in ['highest', 'maximum', '最大', '最高']):
            if shortcut_col and shortcut_col in data.columns:
                neighborhood_col = auto_detect_column(data, 'community') if any('community' in c.lower() for c in data.columns) else auto_detect_column(data, 'neigh')
                max_row = data.loc[data[shortcut_col].idxmax()]
                neighborhood_name = max_row[neighborhood_col]
                max_value = max_row[shortcut_col]
                return f"🏆 Among all communities, **{neighborhood_name}** has the highest **{shortcut_col}**, with a value of {max_value:.2f}."

        # --- Check for average ---
        if any(keyword in user_input_lower for keyword in ['average', 'mean', '平均']):
            if shortcut_col and shortcut_col in data.columns:
                avg_value = data[shortcut_col].mean()
                return f"📊 The average value of **{shortcut_col}** is {avg_value:.2f}."

        # --- Check for specific community + feature value ---
        if shortcut_col and shortcut_col in data.columns:
            for neighborhood_col in ['CommunityAreaName', 'FineNeighborhood']:
                if neighborhood_col in data.columns:
                    for name in data[neighborhood_col].dropna().unique():
                        if name.lower() in user_input_lower:
                            value = data.loc[data[neighborhood_col].str.lower() == name.lower(), shortcut_col].values
                            if len(value) > 0:
                                return f"🏠 The value of **{shortcut_col}** in **{name}** is {value[0]:.2f}."

        # --- Check for definition ---
        for _, row in index_metadata.iterrows():
            if str(row[index_name_col]).lower() in user_input_lower:
                return f"📈 {row[index_name_col]}: {row[index_desc_col]}"

        for _, row in feature_metadata.iterrows():
            if str(row[feature_name_col]).lower() in user_input_lower:
                return f"🔎 {row[feature_name_col]}: {row[feature_desc_col]}"

        return "❓ Sorry, I couldn't understand your question. Please try asking differently."

    st.markdown("""
    <style>
    .chat-bubble {
      background-color: #f0f2f6;
      padding: 12px;
      border-radius: 10px;
      margin-bottom: 10px;
      display: inline-block;
      max-width: 90%;
    }
    .user-bubble {
      background-color: #d1e7dd;
      text-align: right;
      margin-left: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    user_input = st.text_input("Enter your question (e.g., Which community has the highest HOUSPRB?):", key="chatbot_input")

    if user_input:
        st.markdown(f'<div class="chat-bubble user-bubble">{user_input}</div>', unsafe_allow_html=True)
        response = search_answer(user_input)
        st.markdown(f'<div class="chat-bubble">{response}</div>', unsafe_allow_html=True)

# ========== Main Dashboard ==========

# Geospatial Preparation
gdf_raw = gpd.GeoDataFrame.from_features(geojson_data["features"])
gdf_raw = gdf_raw.rename(columns={"pri_neigh": "FineNeighborhood"})
gdf_raw.set_crs(epsg=4326, inplace=True)
gdf = gdf_raw.merge(manual_mapping, on="FineNeighborhood", how="left")
gdf = gdf.merge(data, on="CommunityAreaName", how="left")
gdf["CommunityAreaName"] = gdf["CommunityAreaName"].fillna("Unknown")

def get_stats(data):
    return {'Min': data.min(), 'Q1': data.quantile(0.25), 'Mean': data.mean(), 'Q3': data.quantile(0.75), 'Max': data.max()}

def plot_full_stats(ax, data, title, chart_type='bar'):
    stats = get_stats(data)
    metrics, values = list(stats.keys()), list(stats.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    if chart_type == 'bar':
        bars = ax.bar(metrics, values, color=colors)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", ha='center', va='bottom')
    elif chart_type == 'box':
        ax.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='#2ca02c'))
        stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_dual_comparison(ax, feature_data, index_data, feature_name, index_name):
    feature_stats, index_stats = get_stats(feature_data), get_stats(index_data)
    metrics, colors = list(feature_stats.keys()), ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(metrics, feature_stats.values(), color=colors, alpha=0.7)
    ax2 = ax.twinx()
    line, = ax2.plot(metrics, index_stats.values(), color='black', marker='o', linewidth=2)
    ax.set_title(f"{feature_name} vs {index_name}", fontsize=12)
    ax.legend([bars[0], line], [feature_name, index_name], loc='upper left')

def render_map(gdf, column, title):
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=10)
    gdf_json = json.loads(gdf.to_json())
    for idx, feature in enumerate(gdf_json["features"]):
        feature["properties"]["CommunityAreaName"] = gdf.iloc[idx]["CommunityAreaName"]

    Choropleth(
        geo_data=gdf_json,
        name="choropleth",
        data=gdf,
        columns=["CommunityAreaName", column],
        key_on="feature.properties.CommunityAreaName",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
        highlight=True,
        nan_fill_color='gray'
    ).add_to(m)
    return m

# Main Charts
st.header("Choropleth Map")
st.markdown(f"### {selected_index} Geographic Distribution")
st_folium(render_map(gdf, selected_index, selected_index), use_container_width=True)

st.header("Factor-Level Maps")
col_map1, col_map2 = st.columns(2)
with col_map1:
    st.subheader(f"{col1} Map")
    st_folium(render_map(gdf, col1, col1), use_container_width=True)
with col_map2:
    st.subheader(f"{col2} Map")
    st_folium(render_map(gdf, col2, col2), use_container_width=True)

st.header(f"{selected_index} Index Summary")
fig1, ax1 = plt.subplots(figsize=(10, 4))
plot_full_stats(ax1, data[selected_index], f"{selected_index} Stats\n{index_desc.get(selected_index, '')}")
st.pyplot(fig1)

st.header("Feature Distributions")
col_box1, col_box2 = st.columns(2)
with col_box1:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    plot_full_stats(ax2, data[col1], f"{col1}\n{feature_desc.get(col1, col1)}", chart_type='box')
    st.pyplot(fig2)

with col_box2:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    plot_full_stats(ax3, data[col2], f"{col2}\n{feature_desc.get(col2, col2)}", chart_type='box')
    st.pyplot(fig3)

st.header("Feature vs Index Comparison")
col_comp1, col_comp2 = st.columns(2)
with col_comp1:
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    plot_dual_comparison(ax4, data[col1], data[selected_index], col1, selected_index)
    st.pyplot(fig4)

with col_comp2:
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    plot_dual_comparison(ax5, data[col2], data[selected_index], col2, selected_index)
    st.pyplot(fig5)

# Metadata
with st.expander("🗂️ Full Feature Metadata"):
    st.dataframe(feature_metadata[["column_name", "description"]])

with st.expander("🗂️ Index Metadata"):
    st.dataframe(index_metadata)