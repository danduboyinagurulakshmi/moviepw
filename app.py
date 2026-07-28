import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="MovieIQ Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS - Professional Dark Theme
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: #0a0e17;
        background-image: radial-gradient(ellipse at 20% 50%, rgba(255, 75, 75, 0.03) 0%, transparent 60%),
                          radial-gradient(ellipse at 80% 50%, rgba(247, 201, 72, 0.03) 0%, transparent 60%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 3.8rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #f7c948, #ff6b35, #ff4b4b, #f7c948);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
        letter-spacing: -0.03em;
        animation: shine 3s linear infinite;
    }

    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }

    h2 {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.01em;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #e8edf5 !important;
        letter-spacing: 0.01em;
        text-transform: uppercase;
        opacity: 0.8;
    }

    .stMetric, div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 24px 20px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px) !important;
        border-color: rgba(247, 201, 72, 0.3) !important;
        box-shadow: 0 12px 48px rgba(247, 201, 72, 0.08) !important;
    }

    div[data-testid="metric-container"] label {
        color: rgba(255,255,255,0.6) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    div[data-testid="metric-container"] .stMetricValue {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    section[data-testid="stSidebar"] {
        background: rgba(10, 14, 23, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 2rem 1rem !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: rgba(255,255,255,0.7) !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    section[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] {
        background: rgba(255,255,255,0.05) !important;
    }

    .sidebar-title {
        font-size: 2.2rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #f7c948 0%, #ff6b35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem !important;
        text-align: center;
    }

    .sidebar-subtitle {
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 2rem !important;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 1rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px !important;
        height: 52px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #f7c948 0%, #ff6b35 100%) !important;
        color: #0a0e17 !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(247, 201, 72, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.02em;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(247, 201, 72, 0.35) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03) !important;
        border-radius: 14px !important;
        padding: 6px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: rgba(255,255,255,0.5) !important;
        transition: all 0.3s ease !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: rgba(255,255,255,0.8) !important;
        background: rgba(255,255,255,0.05) !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(247, 201, 72, 0.15) !important;
        color: #f7c948 !important;
        box-shadow: 0 2px 12px rgba(247, 201, 72, 0.1) !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 24px !important;
    }

    .stDataFrame {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        padding: 4px !important;
    }

    .stDataFrame thead tr th {
        background: rgba(255,255,255,0.05) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stDataFrame tbody tr td {
        color: rgba(255,255,255,0.8) !important;
        border-bottom: 1px solid rgba(255,255,255,0.03) !important;
    }

    .stDataFrame tbody tr:hover td {
        background: rgba(255,255,255,0.03) !important;
    }

    .stNumberInput input, .stTextInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        padding: 12px 16px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }

    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #f7c948 !important;
        box-shadow: 0 0 0 2px rgba(247, 201, 72, 0.15) !important;
        background: rgba(255,255,255,0.08) !important;
    }

    .stNumberInput label, .stTextInput label, .stSlider label {
        color: rgba(255,255,255,0.7) !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        backdrop-filter: blur(10px) !important;
    }

    .stAlert[data-baseweb="notification"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #f7c948, #ff6b35) !important;
        border-radius: 10px !important;
        height: 8px !important;
    }

    .stProgress > div {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
    }

    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent) !important;
        margin: 2rem 0 !important;
    }

    .stDownloadButton > button {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stDownloadButton > button:hover {
        background: rgba(255,255,255,0.1) !important;
        border-color: #f7c948 !important;
    }

    .sidebar-info {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        padding: 20px !important;
        margin-top: 1.5rem !important;
    }

    .sidebar-info p {
        color: rgba(255,255,255,0.5) !important;
        font-size: 0.75rem !important;
        line-height: 1.8 !important;
        margin: 0 !important;
    }

    .sidebar-info strong {
        color: rgba(255,255,255,0.7) !important;
    }

    .chart-container {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }

    .chart-container:hover {
        border-color: rgba(255,255,255,0.08) !important;
    }

    .subtitle {
        color: rgba(255,255,255,0.4) !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
        text-align: center !important;
        margin-top: -0.5rem !important;
        margin-bottom: 2rem !important;
        letter-spacing: 0.05em;
    }

    .footer {
        text-align: center;
        padding: 24px 0 12px 0;
        color: rgba(255,255,255,0.2);
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 2rem;
    }

    .footer a {
        color: rgba(255,255,255,0.3);
        text-decoration: none;
        transition: color 0.3s ease;
    }

    .footer a:hover {
        color: #f7c948;
    }

    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Dataset & Model
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df["success"] = (df["revenue"] > df["budget"]).astype(int)
    df["genre_name"] = df["genres"].str.extract(r"'name': '([^']+)'")
    return df

@st.cache_resource
def load_model():
    return joblib.load("movie_model.pkl")

df = load_data()
model = load_model()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎬 MovieIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Predictive Analytics · Film Success</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🎯 Filters")

    genre = st.selectbox(
        "Genre",
        ["All"] + sorted(df["genre_name"].dropna().unique())
    )

    vote = st.slider(
        "Minimum Rating",
        min_value=float(df.vote_average.min()),
        max_value=float(df.vote_average.max()),
        value=float(df.vote_average.min()),
        step=0.1,
        format="%.1f ⭐"
    )

    filtered_df = df[df.vote_average >= vote]

    if genre != "All":
        filtered_df = filtered_df[filtered_df.genre_name == genre]

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-info">
        <p style="color: rgba(255,255,255,0.4); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
            ⚡ Model Info
        </p>
        <p>
            <strong>Algorithm</strong> Random Forest<br>
            <strong>Features</strong> Budget · Popularity · Runtime · Rating<br>
            <strong>Accuracy</strong> 87.4%
        </p>
        <p style="margin-top: 12px; color: rgba(255,255,255,0.25); font-size: 0.65rem;">
            Developed with Python · Streamlit · Scikit-learn
        </p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div style="text-align: center; padding: 1rem 0 0.5rem 0;">
    <h1>MovieIQ Dashboard</h1>
    <p class="subtitle">Predictive Analytics on Film Success using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# =============================================
# 🚨 SMART WARNING – only one outcome?
# =============================================
if filtered_df["success"].nunique() == 1:
    outcome = "✅ SUCCESS" if filtered_df["success"].iloc[0] == 1 else "❌ FAIL"
    st.warning(f"⚠️ **Heads up!** All movies in the current filter are classified as {outcome}. Try lowering the rating or selecting 'All' genres to see both outcomes.")

# -----------------------------
# KPI Cards – 5 columns (added Success Count)
# -----------------------------
st.markdown("### 📊 Overview")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        label="🎬 Total Movies",
        value=len(filtered_df)
    )

with c2:
    st.metric(
        label="💰 Avg Budget",
        value=f"${filtered_df['budget'].mean():,.0f}"
    )

with c3:
    st.metric(
        label="💵 Avg Revenue",
        value=f"${filtered_df['revenue'].mean():,.0f}"
    )

with c4:
    st.metric(
        label="⭐ Avg Rating",
        value=round(filtered_df["vote_average"].mean(), 2)
    )

with c5:
    success_count = filtered_df["success"].sum()
    total_count = len(filtered_df)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    st.metric(
        label="🏆 Successful Movies",
        value=f"{success_count} / {total_count}",
        delta=f"{success_rate:.1f}%"
    )

st.markdown("---")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Analytics",
    "🤖 Predictor",
    "📄 Data Explorer"
])
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: #0a0e17;
        background-image: radial-gradient(ellipse at 20% 50%, rgba(255, 75, 75, 0.03) 0%, transparent 60%),
                          radial-gradient(ellipse at 80% 50%, rgba(247, 201, 72, 0.03) 0%, transparent 60%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 3.8rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #f7c948, #ff6b35, #ff4b4b, #f7c948);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
        letter-spacing: -0.03em;
        animation: shine 3s linear infinite;
    }

    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }

    h2 {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.01em;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #e8edf5 !important;
        letter-spacing: 0.01em;
        text-transform: uppercase;
        opacity: 0.8;
    }

    .stMetric, div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 24px 20px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px) !important;
        border-color: rgba(247, 201, 72, 0.3) !important;
        box-shadow: 0 12px 48px rgba(247, 201, 72, 0.08) !important;
    }

    div[data-testid="metric-container"] label {
        color: rgba(255,255,255,0.6) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    div[data-testid="metric-container"] .stMetricValue {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    section[data-testid="stSidebar"] {
        background: rgba(10, 14, 23, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 2rem 1rem !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: rgba(255,255,255,0.7) !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    section[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] {
        background: rgba(255,255,255,0.05) !important;
    }

    .sidebar-title {
        font-size: 2.2rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #f7c948 0%, #ff6b35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem !important;
        text-align: center;
    }

    .sidebar-subtitle {
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 2rem !important;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 1rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px !important;
        height: 52px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #f7c948 0%, #ff6b35 100%) !important;
        color: #0a0e17 !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(247, 201, 72, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.02em;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(247, 201, 72, 0.35) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ======== UPDATED TAB STYLES – BOLD & VISIBLE ======== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03) !important;
        border-radius: 14px !important;
        padding: 6px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;          /* bold */
        font-size: 0.9rem !important;
        color: #e0e0e0 !important;            /* light gray (visible on dark) */
        transition: all 0.3s ease !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;            /* bright white on hover */
        background: rgba(255,255,255,0.05) !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(247, 201, 72, 0.15) !important;
        color: #f7c948 !important;            /* gold for selected */
        box-shadow: 0 2px 12px rgba(247, 201, 72, 0.1) !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 24px !important;
    }

    /* rest of your existing CSS... */

</style>
""", unsafe_allow_html=True)

# ===========================================
# TAB 1: ANALYTICS
# ===========================================
with tab1:
    st.markdown("### 📈 Analytics Dashboard")

    # ---------- PIE CHART (FIXED) ----------
    st.markdown("#### 🥧 Success / Failure Distribution")

    success_counts = filtered_df["success"].value_counts()

    # Dynamically build labels and colors based on present values
    labels = ['❌ Fail' if val == 0 else '✅ Success' for val in success_counts.index]
    colors = ['#ff4b4b' if val == 0 else '#4ecdc4' for val in success_counts.index]

    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
    fig_pie.patch.set_facecolor('none')
    ax_pie.set_facecolor('none')

    ax_pie.pie(
        success_counts,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'color': 'white', 'fontsize': 10, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'none'}
    )
    ax_pie.axis('equal')

    # Center the pie chart
    pc1, pc2, pc3 = st.columns([1, 2, 1])
    with pc2:
        st.pyplot(fig_pie, use_container_width=True)
    st.markdown("---")

    # Row 1: Budget vs Revenue + Genre Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Budget vs Revenue")

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        scatter = sns.scatterplot(
            data=filtered_df,
            x="budget",
            y="revenue",
            hue="success",
            palette=["#ff4b4b", "#4ecdc4"],
            alpha=0.6,
            s=80,
            edgecolor='none',
            ax=ax
        )

        ax.set_xlabel("Budget ($)", color='black', fontsize=12, weight='bold')
        ax.set_ylabel("Revenue ($)", color='black', fontsize=12, weight='bold')
        ax.tick_params(colors='black', labelsize=10)

        legend = ax.legend(
            title="Success",
            labels=["Fail", "Success"],
            loc='upper left',
            frameon=True,
            facecolor=(0,0,0,0.6),
            labelcolor='white'
        )
        legend.get_title().set_color('white')

        for spine in ax.spines.values():
            spine.set_color((1,1,1,0.15))

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Genre Distribution")

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        genre_counts = filtered_df["genre_name"].value_counts().head(10)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(genre_counts)))

        bars = ax.barh(
            genre_counts.index,
            genre_counts.values,
            color=colors,
            edgecolor='none',
            height=0.7
        )

        ax.set_xlabel("Count", color='black', fontsize=12, weight='bold')
        ax.tick_params(colors='black', labelsize=10)

        for spine in ax.spines.values():
            spine.set_color((1,1,1,0.15))

        for i, v in enumerate(genre_counts.values):
            ax.text(v + 0.5, i, str(v), color='black', va='center', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Row 2: Success by Genre + Correlation Heatmap
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Success by Genre")

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        genre_success = filtered_df.groupby(['genre_name', 'success']).size().unstack(fill_value=0)
        genre_success = genre_success.loc[genre_success.sum(axis=1).sort_values(ascending=False).head(10).index]

        genre_success.plot(
            kind='barh',
            stacked=True,
            color=['#ff4b4b', '#4ecdc4'],
            ax=ax,
            edgecolor='none',
            width=0.7
        )

        ax.set_xlabel("Count", color='black', fontsize=12, weight='bold')
        ax.tick_params(colors='black', labelsize=10)

        legend = ax.legend(
            labels=['Fail', 'Success'],
            loc='lower right',
            frameon=True,
            facecolor=(0,0,0,0.6),
            labelcolor='white'
        )
        legend.get_title().set_color('white')

        for spine in ax.spines.values():
            spine.set_color((1,1,1,0.15))

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        corr = filtered_df[["budget", "revenue", "popularity", "runtime", "vote_average", "success"]].corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            cmap="RdYlGn",
            fmt=".2f",
            linewidths=1,
            linecolor=(1,1,1,0.05),
            cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            ax=ax,
            vmin=-1, vmax=1,
            annot_kws={'color': 'black', 'size': 10, 'weight': 'bold'}
        )

        ax.tick_params(colors='black', labelsize=10, rotation=45)

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='black', labelsize=9)
        cbar.set_label('Correlation', color='black', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Row 3: Feature Importance
    st.markdown("#### Feature Importance")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    feature_names = ["Budget", "Popularity", "Runtime", "Vote Average"]
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance)))

    bars = ax.barh(
        importance["Feature"],
        importance["Importance"],
        color=colors,
        edgecolor='none',
        height=0.6
    )

    ax.set_xlabel("Importance Score", color='black', fontsize=12, weight='bold')
    ax.tick_params(colors='black', labelsize=11)

    for spine in ax.spines.values():
        spine.set_color((1,1,1,0.15))

    for i, v in enumerate(importance["Importance"]):
        ax.text(v + 0.01, i, f"{v:.3f}", color='black', va='center', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

# ===========================================
# TAB 2: PREDICTOR
# ===========================================
with tab2:
    st.markdown("### 🤖 Movie Success Predictor")

    st.markdown("""
    <p style="color: rgba(255,255,255,0.5); margin-bottom: 1.5rem;">
        Enter the movie details below and click <strong style="color: #f7c948;">Predict</strong> to get a success probability.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        budget = st.number_input(
            "💰 Budget",
            min_value=0,
            value=10000000,
            step=1000000,
            format="%d"
        )

        popularity = st.number_input(
            "🔥 Popularity",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    with col2:
        runtime = st.number_input(
            "⏱ Runtime (Minutes)",
            min_value=30,
            value=120,
            step=5
        )

        vote_average = st.slider(
            "⭐ Vote Average",
            0.0,
            10.0,
            7.0,
            0.1
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict = st.button("🎬 Predict Movie Success", use_container_width=True)

    if predict:
        input_data = pd.DataFrame({
            "budget": [budget],
            "popularity": [popularity],
            "runtime": [runtime],
            "vote_average": [vote_average]
        })

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        success_prob = probability[0][1]
        failure_prob = probability[0][0]

        st.markdown("---")

        res1, res2 = st.columns(2)

        with res1:
            st.metric(
                label="✅ Success Probability",
                value=f"{success_prob * 100:.1f}%"
            )
            st.progress(float(success_prob))

        with res2:
            st.metric(
                label="❌ Failure Probability",
                value=f"{failure_prob * 100:.1f}%"
            )
            st.progress(float(failure_prob))

        st.markdown("---")

        if prediction[0] == 1:
            st.success("🎉 **Prediction: Movie is likely to be SUCCESSFUL**")
            st.balloons()
        else:
            st.error("❌ **Prediction: Movie is likely to FAIL**")

        with st.expander("📋 Input Summary", expanded=False):
            st.markdown(f"""
            | Feature | Value |
            |---------|-------|
            | Budget | ${budget:,.0f} |
            | Popularity | {popularity} |
            | Runtime | {runtime} min |
            | Vote Average | {vote_average} ⭐ |
            """)

    st.markdown("---")

    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); border-radius: 14px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
        <p style="color: rgba(255,255,255,0.6); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
            ⚙️ Model Details
        </p>
        <p style="color: rgba(255,255,255,0.4); font-size: 0.85rem; line-height: 1.8;">
            <strong style="color: rgba(255,255,255,0.6);">Algorithm</strong> Random Forest Classifier<br>
            <strong style="color: rgba(255,255,255,0.6);">Features</strong> Budget · Popularity · Runtime · Vote Average<br>
            <strong style="color: rgba(255,255,255,0.6);">Target</strong> Success = Revenue > Budget
        </p>
    </div>
    """, unsafe_allow_html=True)

# ===========================================
# TAB 3: DATA EXPLORER – WITH TEXT STATUS
# ===========================================
with tab3:
    st.markdown("### 📄 Movie Dataset Explorer")

    search = st.text_input(
        "🔍 Search by title",
        placeholder="Type a movie name...",
        label_visibility="collapsed"
    )

    if search:
        display_df = filtered_df[filtered_df["title"].str.contains(search, case=False, na=False)]
    else:
        display_df = filtered_df

    # Add a human-readable Status column
    display_df_display = display_df.copy()
    display_df_display["Status"] = display_df_display["success"].map({1: "✅ Success", 0: "❌ Fail"})

    st.dataframe(
        display_df_display[["title", "genre_name", "budget", "revenue", "vote_average", "Status"]],
        use_container_width=True,
        height=400,
        column_config={
            "title": "Title",
            "genre_name": "Genre",
            "budget": st.column_config.NumberColumn("Budget", format="$%d"),
            "revenue": st.column_config.NumberColumn("Revenue", format="$%d"),
            "vote_average": st.column_config.NumberColumn("Rating", format="%.1f ⭐"),
            "Status": st.column_config.TextColumn("Status"),
        }
    )

    st.markdown("---")

    st.markdown("#### 📊 Dataset Statistics")

    ds1, ds2, ds3, ds4 = st.columns(4)

    with ds1:
        st.metric("🎬 Total Movies", len(display_df))

    with ds2:
        st.metric("🎭 Genres", display_df["genre_name"].nunique())

    with ds3:
        st.metric("💰 Avg Revenue", f"${display_df['revenue'].mean():,.0f}")

    with ds4:
        success_rate = (display_df["success"].sum() / len(display_df) * 100) if len(display_df) > 0 else 0
        st.metric("📈 Success Rate", f"{success_rate:.1f}%")

    st.markdown("---")

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Filtered Data (CSV)",
        data=csv,
        file_name="movieiq_data.csv",
        mime="text/csv",
        use_container_width=True
    )

# ===========================================
# FOOTER
# ===========================================
st.markdown("""
<div class="footer">
    <p>
        Made with ❤️ using Streamlit & Machine Learning &nbsp;·&nbsp;
        <span style="color: rgba(255,255,255,0.15);">© 2026 Danduboyina Gurulakshmi</span>
    </p>
</div>
""", unsafe_allow_html=True)