import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
import pandas as pd
import plotly.express as px
import json
from pathlib import Path
from functools import wraps

# ======================
# 全局配置
# ======================
st.set_page_config(
    page_title="Chicago Community Data Dashboard",
    layout="wide",
    page_icon="📊"
)
st.title("📊 Chicago Community Data Dashboard")

# ======================
# 常量定义
# ======================
DATA_PATH = Path(r'E:\PythonCode\AI\ChicagoNeighborhoodData.json')
CHART_CONFIG = {
    "hole": 0.3,
    "height": 350,
    "width": 350,
    "margin": dict(l=10, r=10, t=40, b=10),
    "uniformtext": dict(minsize=12, mode='hide')
}
COLUMN_RATIO = [2, 1]


# ======================
# 工具函数
# ======================
def safe_numeric(value, default=0):
    """安全转换各类数值格式"""
    if isinstance(value, (int, float)):
        return float(value)

    # 处理带符号的字符串
    cleaned_value = str(value).strip().replace('$', '').replace(',', '').replace('%', '')

    try:
        return float(cleaned_value) if cleaned_value else default
    except:
        return default


# ======================
# 数据加载
# ======================
@st.cache_data
def load_data():
    try:
        with open(DATA_PATH, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # 数据预处理
        numeric_cols = ['Population_Children_Under_19', 'Population_Adults_20To64',
                        'Population_Elderly_Over65', 'MedianHouseholdIncome',
                        'PovertyRate', 'UnemploymentRate']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(safe_numeric)

        return df
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return pd.DataFrame()


df = load_data()


# ======================
# 组件模块
# ======================
def community_selector():
    """社区选择侧边栏组件"""
    st.sidebar.header("Select Community")

    # 社区搜索功能
    search_term = st.sidebar.text_input(
        "🔍 Search Community:",
        key="community_search",
        placeholder="Start typing to filter..."
    )

    all_communities = df["CommunityAreaName"].tolist()
    filtered = [
        c for c in all_communities
        if search_term.lower() in c.lower()
    ] if search_term else all_communities

    # 处理无结果情况
    if not filtered:
        st.sidebar.warning("⚠️ No matching communities found")
        return None

    # 保持选择状态
    return st.sidebar.selectbox(
        label="Select Community:",
        options=filtered,
        index=0,
        key="community_select",
        help="Select a community area to view detailed statistics"
    )


# ======================
# 装饰器
# ======================
def validate_community(func):
    """数据验证装饰器"""

    @wraps(func)
    def wrapper(community_name, *args, **kwargs):
        if not community_name:
            st.info("👈 Please select a community from the sidebar to begin")
            return

        community_data = df[df['CommunityAreaName'] == community_name]
        if community_data.empty:
            st.warning(f"⚠️ No data found for {community_name}!")
            return

        try:
            # 添加数据预览
            with st.expander("⚠️ Debug: Raw Data Preview"):
                st.write(community_data.iloc[0].to_dict())

            return func(community_data.iloc[0], *args, **kwargs)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.exception(e)

    return wrapper


# ======================
# 图表组件
# ======================
def create_chart_section(
        data,
        checkbox_label,
        chart_title,
        labels,
        value_keys,
        insights_text,
        default_visible=True
):
    """通用图表创建组件"""
    if st.checkbox(checkbox_label, value=default_visible):
        col1, col2 = st.columns(COLUMN_RATIO)

        with col1:
            # 安全获取数据
            values = [safe_numeric(data.get(k, 0)) for k in value_keys]
            total = sum(values)

            if total == 0:
                st.warning("No available data for this category")
                return

            # 创建饼图
            fig = px.pie(
                names=labels,
                values=values,
                title=chart_title,
                hole=CHART_CONFIG["hole"]
            )
            fig.update_layout(
                height=CHART_CONFIG["height"],
                width=CHART_CONFIG["width"],
                margin=CHART_CONFIG["margin"],
                uniformtext=CHART_CONFIG["uniformtext"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 生成统计摘要
            st.markdown(f"### {chart_title.split(' ')[0]} Insights")
            st.markdown(insights_text)

            # 显示详细数据
            with st.expander("📊 View raw data"):
                data_dict = {label: f"{value:,.0f}" for label, value in zip(labels, values)}
                st.json(data_dict)


# ======================
# 数据展示模块
# ======================
@validate_community
def show_population(data):
    """人口分布模块"""
    # 定义需要使用的字段键
    pop_keys = [
        "Population_Children_Under_19",
        "Population_Adults_20To64",
        "Population_Elderly_Over65"
    ]

    # 安全获取并转换数值
    def safe_get_int(key):
        value = data.get(key)
        try:
            return int(float(value)) if value not in [None, "", "N/A"] else 0
        except:
            return 0

    # 计算总数
    total_population = sum(safe_get_int(k) for k in pop_keys)

    white_population = safe_numeric(data.get('Population_White', 0))
    white_percentage = (white_population / total_population * 100) if total_population > 0 else 0

    HispanicOrLatino_population = safe_numeric(data.get('Population_HispanicOrLatino', 0))
    HispanicOrLatino_percentage = (HispanicOrLatino_population / total_population * 100) if total_population > 0 else 0

    Black_population = safe_numeric(data.get('Population_Black', 0))
    Black_percentage = (Black_population / total_population * 100) if total_population > 0 else 0

    Asian_population = safe_numeric(data.get('Population_Asian', 0))
    Asian_percentage = (Asian_population / total_population * 100) if total_population > 0 else 0

    Other_population = safe_numeric(data.get('Population_Black', 0))
    Other_percentage = (Other_population / total_population * 100) if total_population > 0 else 0

    create_chart_section(
        data=data,
        checkbox_label="Show Population Distribution",
        chart_title=f"{data['CommunityAreaName']} Age Distribution",
        labels=["Children (<19)", "Adults (20-64)", "Elderly (>65)"],
        value_keys=pop_keys,
        insights_text=f"""
        **Population Overview**  
        - Total Population: {total_population:,}
        - White Population: {white_population:,} ({white_percentage:.1f}%)
        - HispanicOrLatino Population: {HispanicOrLatino_population:,} ({HispanicOrLatino_percentage:.1f}%)
        - Black Population: {Black_population:,} ({Black_percentage:.1f}%)
        - Asian Population: {Asian_population:,} ({Asian_percentage:.1f}%)
        - Other MultipleRaces Population: {Other_population:,} ({Other_percentage:.1f}%)
        """
    )

    col1, col2 = st.columns([2, 3])
    with col1:
        if st.button(f"📈 Compare with City Average - Population",
                     key="pop_compare_btn"):
            st.session_state['show_pop_compare'] = not st.session_state.get('show_pop_compare', False)

    if st.session_state.get('show_pop_compare', False):
        with st.expander("City Comparison: Population Distribution", expanded=True):
            # 准备数据
            pop_keys = ['Population_Children_Under_19',
                        'Population_Adults_20To64',
                        'Population_Elderly_Over65']
            current_values = [data[k] for k in pop_keys]
            city_avg = df[pop_keys].mean().tolist()

            comparison_df = pd.DataFrame({
                'Category': ['Children (<19)', 'Adults (20-64)', 'Elderly (>65)'],
                'Current': current_values,
                'City Average': city_avg
            }).melt(id_vars='Category', var_name='Type', value_name='Population')

            # 创建对比图表
            fig = px.bar(
                comparison_df,
                x='Category',
                y='Population',
                color='Type',
                barmode='group',
                title=f"Population Distribution Comparison: {data['CommunityAreaName']} vs City Average",
                labels={'Population': 'Population Count'},
                height=400
            )
            fig.update_layout(uniformtext_minsize=10)
            st.plotly_chart(fig, use_container_width=True)


@validate_community
def show_education(data):
    """教育水平模块"""
    high_edu = safe_numeric(data.get('EducationLV_GraduateOrProfessional'))
    Bachelor = safe_numeric(data.get('EducationLV_Bachelor'))
    Associate = safe_numeric(data.get('EducationLV_Associate'))
    SomeCollege_NoDegree = safe_numeric(data.get('EducationLV_SomeCollege_NoDegree'))
    HighSchoolOrEquivalent = safe_numeric(data.get('EducationLV_HighSchoolOrEquivalent'))
    LessThanHighSchool = safe_numeric(data.get('EducationLV_LessThanHighSchool'))

    Total_edu = high_edu + Bachelor + Associate + SomeCollege_NoDegree + HighSchoolOrEquivalent + LessThanHighSchool
    total_population = safe_numeric(data.get('Population_Total'))

    high_edu_rate = (high_edu / Total_edu * 100) if Total_edu > 0 else 0
    Percentage_educated = (Total_edu / total_population * 100) if Total_edu > 0 else 0

    create_chart_section(
        data=data,
        checkbox_label="Show Education Distribution",
        chart_title=f"{data['CommunityAreaName']} Education Levels",
        labels=[
            "Below High School",
            "High School Graduate",
            "College Student",
            "Bachelor's Degree",
            "Master's Degree or Higher"
        ],
        value_keys=[
            "EducationLV_LessThanHighSchool",
            "EducationLV_HighSchoolOrEquivalent",
            "EducationLV_SomeCollege_NoDegree",
            "EducationLV_Bachelor",
            "EducationLV_GraduateOrProfessional"
        ],
        insights_text=f"""
        **Education Highlights**  
        - Percentage educated: {Percentage_educated:.1f}/10
        - Proportion with tertiary education: {high_edu_rate:.1f}% 
        """
    )


@validate_community
def show_income(data):
    """收入分布模块"""
    # 安全获取关键指标
    MedianIncome = safe_numeric(data.get('Income_MedianIncome', 0))
    PerCapitaIncome = safe_numeric(data.get('Income_PerCapitaIncome', 0))

    create_chart_section(
        data=data,
        checkbox_label="Show Income Distribution",
        chart_title=f"{data['CommunityAreaName']} Income Distribution",
        labels=[
            "≤$25K",
            "$25K-49K",
            "$50K-74K",
            "$75K-99K",
            "$100K-149K",
            "≥$150K"
        ],
        value_keys=[
            "Income_Less25000",
            "Income_25000To49999",
            "Income_50000To74999",
            "Income_75000To99999",
            "Income_100000To149999",
            "Income_150000AndOver"
        ],
        insights_text=f"""
        **Economic Indicators**  
        - Median Income: {MedianIncome}
        - Per Capita Income: {PerCapitaIncome}
        """
    )


@validate_community
def show_employment(data):
    """就业分布模块"""
    # 安全获取数据
    employed = safe_numeric(data.get('Employment_Employed', 0))
    unemployed = safe_numeric(data.get('Employment_Unemployed', 0))
    not_in_labor = safe_numeric(data.get('Employment_NotInLaborForce', 0))

    # 计算关键指标
    total_workforce = employed + unemployed
    unemployment_rate = (unemployed / total_workforce * 100) if total_workforce > 0 else 0
    labor_participation_rate = (total_workforce / (total_workforce + not_in_labor) * 100) if (
                                                                                                         total_workforce + not_in_labor) > 0 else 0

    # 生成展示文本
    insights = f"""
    **Employment Overview**  
    - Labor Force Participation Rate: {labor_participation_rate:.1f}%
    - Unemployment Rate: {unemployment_rate:.1f}%
    - Working Age Population: {total_workforce + not_in_labor:,}
    """

    create_chart_section(
        data=data,
        checkbox_label="Show Employment Distribution",
        chart_title=f"{data['CommunityAreaName']} Employment Status",
        labels=["Employed", "Unemployed", "Not in Labor"],
        value_keys=[
            "Employment_Employed",
            "Employment_Unemployed",
            "Employment_NotInLaborForce"
        ],
        insights_text=insights
    )


@validate_community
def show_housing(data):
    """住房分布模块"""

    # 安全获取数据
    single_units = safe_numeric(data.get('HousingTypes_Single', 0))
    multi_units = safe_numeric(data.get('HousingTypes_Multi', 0))
    mobile_units = safe_numeric(data.get('HousingTypes_Mobile', 0))

    # 计算关键指标
    total_housing = single_units + multi_units + mobile_units
    homeowner  = safe_numeric(data.get('Homeownership', 0))
    renters = safe_numeric(data.get('Renters', 0))


    homeownership_rate = (homeowner / total_housing * 100) if total_housing > 0 else 0
    renter_rate = (renters / total_housing * 100) if total_housing > 0 else 0

    insights = f"""
    **Housing Overview**  
    - Total Housing Units: {total_housing:,}
    - Homeownership Rate: {homeownership_rate:.1f}%
    - Rental Units: {renter_rate:.1f}%
    """

    create_chart_section(
        data=data,
        checkbox_label="Show Housing Distribution",
        chart_title=f"{data['CommunityAreaName']} Housing Types",
        labels=["Single Family", "Multi-Unit", "Mobile"],
        value_keys=[
            "HousingTypes_Single",
            "HousingTypes_Multi",
            "HousingTypes_Mobile"
        ],
        insights_text=insights
    )

# ======================
# 主程序
# ======================
selected_community = community_selector()

if selected_community:
    # 原有模块
    show_population(selected_community)
    show_education(selected_community)
    show_income(selected_community)

    # 新增模块
    show_employment(selected_community)
    show_housing(selected_community)

    # 添加分割线
    st.markdown("---")

# ======================
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str):
    """Queries ChromaDB and retrieves relevant context for the AI model."""
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    results = db.similarity_search_with_score(query_text, k=20)
    if not results:
        return "No relevant data found in the database."

    # Debugging: Print retrieved documents
    for doc, score in results:
        print(f"📄 Retrieved doc: {doc.metadata.get('id', 'Unknown')} | Score: {score}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="deepseek-r1:7b", base_url="http://localhost:11434")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    return f"Response: {response_text}\nSources: {sources}"

# Streamlit user interface
def main():
    st.title("RAG Query Application")
    query_text = st.text_input("Enter your question:")

    if query_text:
        result = query_rag(query_text)
        st.write(result)

if __name__ == "__main__":
    main()

