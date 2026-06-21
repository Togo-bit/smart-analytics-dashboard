from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import posthog
import os
import uuid
from plotly.colors import qualitative
from ml.preprocessing import (
    clean_data,
    prepare_features,
    detect_problem_type,
    scale_data
)
from ml.training import train_data
from ml.evaluation import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, classification_report
from pandas.api.types import is_numeric_dtype
import base64
from groq import Groq

import base64

st.set_page_config(
    page_title="SalesPulse",
    page_icon="📈",
)

st.markdown("""
<style>

.hero-container{
    background: linear-gradient(
        135deg,
        #0B1220 0%,
        #111827 50%,
        #172554 100%
    );

    padding:40px 50px;

    border-radius:24px;

    border:1px solid rgba(255,255,255,0.08);

    margin-bottom:30px;
}

.hero-card{
    border-radius:18px;
    border:1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.hero-header{
    background: linear-gradient(
        135deg,
        #0B1220 0%,
        #111827 50%,
        #1E3A8A 100%
    );

    padding:50px;

    border-radius:24px;

    border:1px solid rgba(255,255,255,0.08);

    margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

logo_base64 = get_base64_image(
    "logo.logo"
)

st.markdown(
f"""
<div class="hero-header">

<div style="
display:flex;
align-items:center;
gap:20px;
">

<img
src="data:image/png;base64,{logo_base64}"
style="
height:90px;
border-radius:12px;
">

<span style="
font-size:72px;
font-weight:800;
color:white;
">
SalesPulse
</span>

</div>

<div style="
margin-top:30px;
font-size:48px;
font-weight:700;
color:white;
line-height:1.2;
">
Turn Raw Data Into Business Decisions
</div>

<div style="
margin-top:20px;
font-size:20px;
color:#CBD5E1;
">
Upload data, uncover trends, identify risks,
forecast outcomes, and make smarter decisions.
</div>

</div>
""",
unsafe_allow_html=True
)

groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error("GROQ_API_KEY not found")
    st.stop()

client = Groq(api_key=groq_key)
import numpy as np

def find_revenue_column(df):

    revenue_keywords = [
        "revenue",
        "sales",
        "amount",
        "income",
        "profit",
        "price",
        "total"
    ]

    for col in df.columns:

        if col.lower() in revenue_keywords:
            return col

    return None

def generate_findings(df):

    findings = []

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    # =========================
    # CATEGORY CONCENTRATION
    # =========================

    for col in categorical_cols:

        try:

            share = (
                df[col]
                .value_counts(normalize=True)
                * 100
            )

            if share.iloc[0] > 40:

                findings.append(
                    f"{share.index[0]} dominates '{col}' with {share.iloc[0]:.1f}% share."
                )

        except:
            pass

    # =========================
    # NUMERIC CONCENTRATION
    # =========================

    for col in numeric_cols:

        try:

            top10_share = (
                df.nlargest(
                    min(10, len(df)),
                    col
                )[col].sum()
                /
                df[col].sum()
            ) * 100

            if top10_share > 30:

                findings.append(
                    f"Top records contribute {top10_share:.1f}% of total {col}."
                )

        except:
            pass

        # =========================
        # REVENUE CONCENTRATION
        # =========================

        revenue_col = find_revenue_column(df)

        if revenue_col:

            categorical_cols = df.select_dtypes(
                exclude=np.number
            ).columns

            for cat_col in categorical_cols:

                try:

                    revenue_by_group = (
                        df.groupby(cat_col)[revenue_col]
                        .sum()
                        .sort_values(ascending=False)
                    )

                    if len(revenue_by_group) >= 5:
                        top5_share = (
                                             revenue_by_group.head(5).sum()
                                             /
                                             revenue_by_group.sum()
                                     ) * 100

                        findings.append(
                            f"Top 5 {cat_col} contribute {top5_share:.1f}% of total {revenue_col}."
                        )

                except:
                    pass

                # =========================
                # PARETO ANALYSIS
                # =========================

                if revenue_col:

                    categorical_cols = df.select_dtypes(
                        exclude=np.number
                    ).columns

                    for cat_col in categorical_cols:

                        try:

                            revenue_dist = (
                                df.groupby(cat_col)[revenue_col]
                                .sum()
                                .sort_values(ascending=False)
                            )

                            cumulative = (
                                    revenue_dist.cumsum()
                                    /
                                    revenue_dist.sum()
                            )

                            count_needed = (
                                    cumulative <= 0.80
                            ).sum()

                            percent_needed = (
                                                     count_needed
                                                     /
                                                     len(revenue_dist)
                                             ) * 100

                            findings.append(
                                f"{percent_needed:.1f}% of {cat_col} generate 80% of {revenue_col}."
                            )

                        except:
                            pass

                # =========================
                # DEPENDENCY RISK
                # =========================

                if revenue_col:

                    categorical_cols = df.select_dtypes(
                        exclude=np.number
                    ).columns

                    for cat_col in categorical_cols:

                        try:

                            revenue_dist = (
                                df.groupby(cat_col)[revenue_col]
                                .sum()
                                .sort_values(ascending=False)
                            )

                            largest_share = (
                                                    revenue_dist.iloc[0]
                                                    /
                                                    revenue_dist.sum()
                                            ) * 100

                            if largest_share > 20:
                                findings.append(
                                    f"Largest {cat_col} contributes {largest_share:.1f}% of total {revenue_col}, creating dependency risk."
                                )

                        except:
                            pass

    # =========================
    # CORRELATIONS
    # =========================

    if len(numeric_cols) >= 2:

        corr = df[numeric_cols].corr()

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):

                value = corr.iloc[i, j]

                if abs(value) > 0.7:

                    findings.append(
                        f"{numeric_cols[i]} and {numeric_cols[j]} are strongly related ({value:.2f})."
                    )

    # =========================
    # ANOMALY DETECTION
    # =========================

    for col in numeric_cols:

        try:

            q99 = df[col].quantile(0.99)

            extreme = df[df[col] > q99]

            if len(extreme) > 0:

                findings.append(
                    f"A small number of records have exceptionally high {col} values."
                )

        except:
            pass

    return findings[:10]

def generate_ai_insights(findings):

    if not findings:
        return "No significant insights found."

    prompt = f"""
You are an experienced business analyst.

Dataset Findings:

{chr(10).join(findings)}

Generate:

📈 Trend Summary
🚨 Risks / Anomalies
🔮 Forecast
💡 Recommended Actions

Rules:

- Focus on business impact.
- Do not repeat findings word-for-word.
- If evidence is insufficient for forecasting,
  say 'Additional historical data required.'
- Give practical recommendations.
- Keep under 150 words.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

st.set_page_config(layout='wide')

st.markdown("""
<style>

.card {

    background:#111827;

    padding:20px;

    border-radius:18px;

    border:1px solid rgba(
        255,
        255,
        255,
        0.08
    );

    box-shadow:
        0 8px 24px rgba(
            0,
            0,
            0,
            0.35
        );

    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ================= CHART BUTTONS ONLY ================= */

/* Target ONLY buttons inside chart container */
.chart-buttons button {
    height: 80px !important;
    aspect-ratio: 1 / 1 !important;
    width: 100% !important;

    font-size: 26px !important;
    border-radius: 12px !important;
    padding: 0 !important;

    border: 1px solid rgba(255,255,255,0.1) !important;
    background-color: #1E293B !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Hover effect */
.chart-buttons button:hover {
    background-color: #334155 !important;
    transform: scale(1.05);
    transition: 0.2s ease;
}

/* Selected button */
.chart-buttons button[aria-pressed="true"] {
    background-color: #2563EB !important;
    color: white !important;
    border: 1px solid #2563EB !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.insight-card{
    background:#111827;
    border-left:4px solid #7C3AED;
    padding:14px;
    margin-bottom:10px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Tab container */
.stTabs [data-baseweb="tab-list"]{
    gap:12px;
    margin-top:10px;
}

/* Individual tabs */
.stTabs [data-baseweb="tab"]{
    background-color: rgba(255,255,255,0.04);
    border-radius:12px;
    padding:10px 20px;
    color:white;
    font-weight:600;
    border:1px solid rgba(255,255,255,0.08);
}

/* Active tab */
.stTabs [aria-selected="true"]{
    background: linear-gradient(
        135deg,
        #1E3A5F,
        #164E63
    );
    color:white !important;
}

/* Remove red underline */
.stTabs [data-baseweb="tab-highlight"]{
    display:none;
}

</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "https://flask-backend-ygwu.onrender.com/")
posthog.api_key = st.secrets["POSTHOG_API_KEY"]
posthog.host = "https://app.posthog.com"
def get_user_id():

    # Logged in user
    if "user_email" in st.session_state:
        return st.session_state["user_email"]

    # Anonymous visitor
    if "guest_id" not in st.session_state:
        st.session_state["guest_id"] = f"guest_{uuid.uuid4()}"

    return st.session_state["guest_id"]

# Track app opened
posthog.capture(
    distinct_id=get_user_id(),
    event="app_opened"
)

# AUTH Section
if "token" not in st.session_state:
    page = st.sidebar.selectbox('Choose page',['Login','Register'])

    if page == 'Login':
        st.subheader('Login')

        email = st.text_input('Email',key = 'login_email')
        password = st.text_input('Password', type = 'password', key = 'login_password')

        if st.button('Login'):
            try:
                res = requests.post(
                    f"{API_URL}/login",
                    data={'email': email, 'password': password}
                )

                if res.status_code == 200:
                    response_data = res.json()

                    if "token" in response_data:

                        st.session_state["user_email"] = email
                        st.session_state['token'] = response_data['token']

                        posthog.capture(
                            distinct_id=email,
                            event="user_identified",
                            properties={
                                "email": email
                            }
                        )

                        posthog.capture(
                            distinct_id=email,
                            event="user_logged_in"
                        )

                        st.success("Login Successful")
                        st.rerun()
                    else:
                        st.error("Token not found in response")

                else:
                    st.error(res.json().get("message", "Login failed"))

            except requests.exceptions.RequestException:
                st.error("Server not reachable")
                st.stop()

        st.markdown("---")

        st.markdown("### Or Login with Google")

        google_login_url = f"{API_URL}google/login"

    elif page == 'Register':
        st.subheader('Register')

        username = st.text_input('username', key = 'reg_username')
        email = st.text_input('email', key = 'reg_email')
        password = st.text_input('password', type = 'password', key = 'reg_password')

        if st.button('Register'):
            try:
                res = requests.post(
                    f"{API_URL}/register",
                    data={
                        "username": username,
                        "email": email,
                        "password": password
                    }
                )

                if res.status_code == 200:

                    posthog.capture(
                        distinct_id=email,
                        event="user_registered",
                        properties={
                            "email": email,
                            "username": username
                        }
                    )

                    st.success('Registration Successful')

                else:
                    st.error(res.json().get("message", "Registration failed"))

            except requests.exceptions.RequestException:
                st.error("Server not reachable")
                st.stop()

else:
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}

    with st.sidebar:

        st.markdown("#  SalesPulse")

        st.caption(
            "Decision Intelligence\nfor Growing Businesses"
        )

        st.divider()

        if st.button("🚪 Logout"):

            posthog.capture(
                distinct_id=get_user_id(),
                event="user_logged_out"
            )

            del st.session_state['token']
            del st.session_state['user_email']

            if "df" in st.session_state:
                del st.session_state["df"]

            st.rerun()

    if "charts" not in st.session_state:
        st.session_state["charts"] = []

    if "selected_chart" not in st.session_state:
        st.session_state["selected_chart"] = None

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("""
        ### 📈 Growth

        Identify trends, high-performing
        segments and expansion potential.
        """)

    with c2:
        st.warning("""
        ### ⚠ Revenue Risks

        Detect concentration risks,
        declining performance and anomalies.
        """)

    with c3:
        st.info("""
        ### 🔮 Forecasting

        Predict future outcomes using
        historical patterns.
        """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Upload",
        "Dashboard",
        "Analysis",
        "Predictions"
    ])

    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True
    )

    with tab1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📂 Start Your Analysis")

        st.caption(
            "Supports CSV and Excel files"
        )

        # =========================
        # DATA SOURCE
        # =========================

        st.markdown("### Upload CSV or Excel File")

        # =========================
        # CSV / EXCEL
        # =========================

        with st.container(border=True):

            upload_file = st.file_uploader(
                "Drag & Drop your CSV or Excel file",
                type=['csv', 'xlsx', 'xls']
            )

        if upload_file:

            try:

                file_extension = upload_file.name.split(".")[-1]

                if file_extension == "csv":

                    df = pd.read_csv(upload_file)

                elif file_extension in ["xlsx", "xls"]:

                    excel_file = pd.ExcelFile(upload_file)

                    selected_sheet = st.selectbox(
                        "Select Excel Sheet",
                        excel_file.sheet_names
                    )

                    df = pd.read_excel(
                        upload_file,
                        sheet_name=selected_sheet
                    )

                # =========================
                # POSTHOG
                # =========================

                posthog.capture(
                    distinct_id=get_user_id(),
                    event="dataset_uploaded",
                    properties={
                        "rows": df.shape[0],
                        "columns": df.shape[1]
                    }
                )

                # =========================
                # RESET CHARTS FOR NEW FILE
                # =========================

                new_file_name = upload_file.name

                if "last_uploaded_file" not in st.session_state:
                    st.session_state["last_uploaded_file"] = ""

                if st.session_state["last_uploaded_file"] != new_file_name:
                    st.session_state["charts"] = []
                    st.session_state["selected_chart"] = None

                    # Clear previous AI results
                    st.session_state.pop("findings", None)
                    st.session_state.pop("ai_summary", None)
                    st.session_state.pop("insight_file", None)

                    st.session_state["last_uploaded_file"] = new_file_name

                st.session_state["df"] = df

                current_file = upload_file.name

                if (
                        "insight_file" not in st.session_state
                        or st.session_state["insight_file"] != current_file
                ):
                    findings = generate_findings(df)

                    ai_summary = generate_ai_insights(findings)

                    st.session_state["findings"] = findings
                    st.session_state["ai_summary"] = ai_summary

                    st.session_state["insight_file"] = current_file

            except Exception as e:

                posthog.capture(
                    distinct_id=get_user_id(),
                    event="upload_error",
                    properties={
                        "error": str(e)
                    }
                )

                st.error(f"⚠️ Error reading file: {e}")

        # =========================
        # PERSISTENT PREVIEW
        # =========================
        if "df" not in st.session_state:
            st.markdown("""
        ### What can SalesPulse do?
            """)

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.info("Revenue")

            with c2:
                st.info("Forecasting")

            with c3:
                st.info("Trends")

            with c4:
                st.info("Anomalies")

        if "df" in st.session_state:
            st.markdown("### 🔍 Dataset Preview")

            st.dataframe(
                st.session_state["df"].head()
            )

            rows = st.session_state["df"].shape[0]
            cols = st.session_state["df"].shape[1]

            missing = (
                st.session_state["df"]
                .isna()
                .sum()
                .sum()
            )

            duplicates = (
                st.session_state["df"]
                .duplicated()
                .sum()
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Rows", rows)

            c2.metric("Columns", cols)

            c3.metric("Missing Values", missing)

            c4.metric("Duplicates", duplicates)

            st.markdown("---")

            st.subheader("📈 Executive Summary")

            st.info(
                st.session_state["ai_summary"]
            )

            st.subheader("📊 Key Findings")

            for finding in st.session_state["findings"]:
                st.markdown(
                    f"""
                    <div class="insight-card">
                        📌 {finding}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        if "df" not in st.session_state:
            st.info("📁 Upload a file first")
        else:
            df = st.session_state['df']

            # ================= ADD NEW CHART =================

            if st.button("➕ Create New Chart"):
                numeric_columns = df.select_dtypes(include='number').columns

                new_chart = {

                    "id": str(uuid.uuid4()),

                    # BASIC
                    "title": "New Chart",
                    "subtitle": "",

                    # CHART
                    "chart_type": "bar",
                    "group_col": df.columns[0],
                    "numeric_cols": [numeric_columns[0]] if len(numeric_columns) > 0 else [],

                    # AGGREGATION
                    "aggregations": {},

                    # THEMES
                    "chart_theme": "Default",
                    "pie_theme": "Vibrant",

                    # AXIS
                    "x_label": df.columns[0],
                    "y_label": "Value",

                    "axis_font_size": 16,
                    "axis_color": "#FFFFFF",
                    "axis_bold": True,

                    # HOVER
                    "hover_bg": "#111827",
                    "hover_font_color": "#FFFFFF",
                    "hover_font_size": 14,

                    # FILTERS
                    "filter_type": "None",
                    "filter_column": None,
                    "top_n": 5,
                    "selected_values": []
                }

                st.session_state["charts"].append(new_chart)

                st.session_state["selected_chart"] = new_chart["id"]

                st.rerun()

            # ================= SELECTED CHART =================

            selected_chart = None

            for chart_obj in st.session_state["charts"]:

                if chart_obj["id"] == st.session_state["selected_chart"]:
                    selected_chart = chart_obj
                    break

            # ================= SIDEBAR CONTROLS =================

            if selected_chart:

                # ================= DEFAULT KEYS =================

                defaults = {

                    "aggregations": {},
                    "subtitle": "",

                    "chart_theme": "Default",
                    "pie_theme": "Vibrant",

                    "x_label": selected_chart["group_col"],
                    "y_label": "Value",

                    "axis_font_size": 16,
                    "axis_color": "#FFFFFF",
                    "axis_bold": True,

                    "hover_bg": "#111827",
                    "hover_font_color": "#FFFFFF",
                    "hover_font_size": 14,

                    "filter_type": "None",
                    "filter_column": None,
                    "top_n": 5,
                    "selected_values": []
                }

                for key, value in defaults.items():

                    if key not in selected_chart:
                        selected_chart[key] = value

                st.sidebar.markdown("## 🎛️ Chart Controls")

                # ================= TITLE =================

                selected_chart["title"] = st.sidebar.text_input(
                    "Chart Title",
                    value=selected_chart["title"],
                    key=f"title_{selected_chart['id']}"
                )

                selected_chart["subtitle"] = st.sidebar.text_input(
                    "Chart Subtitle",
                    value=selected_chart["subtitle"],
                    key=f"subtitle_{selected_chart['id']}"
                )

                # ================= CHART ICONS =================

                st.sidebar.markdown("### 📊 Chart Type")

                cols = st.sidebar.columns(5)

                chart_icons = {
                    "📊": "bar",
                    "📈": "line",
                    "📉": "area",
                    "🥧": "pie",
                    "📦": "histogram"
                }

                for idx, (icon, chart_value) in enumerate(chart_icons.items()):

                    with cols[idx]:

                        if st.button(
                                icon,
                                key=f"{selected_chart['id']}_{chart_value}",
                                help=chart_value.capitalize()
                        ):
                            selected_chart["chart_type"] = chart_value

                st.sidebar.caption(
                    f"Selected: {selected_chart['chart_type'].upper()}"
                )

                # ================= GROUPING =================

                if (
                        "group_col" not in selected_chart
                        or selected_chart["group_col"] not in df.columns
                ):
                    selected_chart["group_col"] = df.columns[0]

                selected_chart["group_col"] = st.sidebar.selectbox(
                    "Group By",
                    df.columns,
                    index=list(df.columns).index(
                        selected_chart["group_col"]
                    ),
                    key=f"group_{selected_chart['id']}"
                )

                numeric_cols_available = [

                    col for col in
                    df.select_dtypes(include='number').columns

                    if col != selected_chart["group_col"]
                ]

                # REMOVE INVALID COLUMNS
                selected_chart["numeric_cols"] = [

                    col for col in selected_chart["numeric_cols"]

                    if col in numeric_cols_available
                ]

                max_selections = (
                    1 if selected_chart["chart_type"] == "pie"
                    else None
                )

                if (
                        selected_chart["chart_type"] == "pie"
                        and len(selected_chart["numeric_cols"]) > 1
                ):
                    selected_chart["numeric_cols"] = [
                        selected_chart["numeric_cols"][0]
                    ]

                selected_chart["numeric_cols"] = st.sidebar.multiselect(
                    "Numeric Columns",
                    numeric_cols_available,
                    default=selected_chart["numeric_cols"],
                    max_selections=max_selections
                )

                if selected_chart["chart_type"] == "pie":
                    st.sidebar.caption(
                        "🥧 Pie charts support only one numeric column."
                    )

                # REMOVE GROUP COLUMN FROM NUMERIC COLUMNS
                if selected_chart["group_col"] in selected_chart["numeric_cols"]:
                    selected_chart["numeric_cols"].remove(
                        selected_chart["group_col"]
                    )

                    st.sidebar.warning(
                        "Group column cannot also be a numeric column"
                    )

                if len(selected_chart["numeric_cols"]) == 0:
                    st.sidebar.warning(
                        "Please select atleast one numeric column"
                    )

                # ================= PER COLUMN AGGREGATION =================

                st.sidebar.markdown("### 📊 Column Aggregations")

                aggregation_options = ["sum", "mean", "max", "min", "count"]

                # Initialize missing aggregations
                for col in selected_chart["numeric_cols"]:

                    if col not in selected_chart["aggregations"]:
                        selected_chart["aggregations"][col] = "sum"

                # Remove deleted columns from aggregation dict
                selected_chart["aggregations"] = {

                    col: agg
                    for col, agg in selected_chart["aggregations"].items()
                    if col in selected_chart["numeric_cols"]
                }

                # UI for each column
                for col in selected_chart["numeric_cols"]:
                    selected_chart["aggregations"][col] = st.sidebar.selectbox(
                        f"{col} Aggregation",
                        aggregation_options,
                        index=aggregation_options.index(
                            selected_chart["aggregations"][col]
                        ),
                        key=f"agg_{selected_chart['id']}_{col}"
                    )

                # ================= THEMES =================

                st.sidebar.markdown("### 🎨 Themes")

                selected_chart["chart_theme"] = st.sidebar.selectbox(
                    "Chart Theme",
                    ["Default", "Vibrant", "Pastel", "Neon"],
                    index=["Default", "Vibrant", "Pastel", "Neon"].index(
                        selected_chart["chart_theme"]
                    )
                )

                pie_theme_options = [
                    "Plotly",
                    "Bold",
                    "Pastel",
                    "Dark24",
                    "Light24",
                    "Safe",
                    "Vivid",
                    "Prism"
                ]

                if selected_chart["pie_theme"] not in pie_theme_options:
                    selected_chart["pie_theme"] = "Plotly"

                selected_chart["pie_theme"] = st.sidebar.selectbox(
                    "Pie Theme",
                    pie_theme_options,
                    index=pie_theme_options.index(
                        selected_chart["pie_theme"]
                    )
                )

                # ================= AXIS =================

                st.sidebar.markdown("### 🧭 Axis Styling")

                selected_chart["x_label"] = st.sidebar.text_input(
                    "X-axis Label",
                    value=selected_chart["x_label"]
                )

                selected_chart["y_label"] = st.sidebar.text_input(
                    "Y-axis Label",
                    value=selected_chart["y_label"]
                )

                selected_chart["axis_font_size"] = st.sidebar.slider(
                    "Axis Font Size",
                    10,
                    30,
                    selected_chart["axis_font_size"]
                )

                selected_chart["axis_color"] = st.sidebar.color_picker(
                    "Axis Color",
                    selected_chart["axis_color"]
                )

                selected_chart["axis_bold"] = st.sidebar.checkbox(
                    "Bold Axis Labels",
                    value=selected_chart["axis_bold"]
                )

                # ================= HOVER =================

                st.sidebar.markdown("### ✨ Hover Styling")

                selected_chart["hover_bg"] = st.sidebar.color_picker(
                    "Hover Background",
                    selected_chart["hover_bg"]
                )

                selected_chart["hover_font_color"] = st.sidebar.color_picker(
                    "Hover Text Color",
                    selected_chart["hover_font_color"]
                )

                selected_chart["hover_font_size"] = st.sidebar.slider(
                    "Hover Font Size",
                    10,
                    24,
                    selected_chart["hover_font_size"]
                )

                # ================= FILTERS =================

                st.sidebar.markdown("### 🔍 Advanced Filters")

                filter_options = [
                    "None",
                    "Top N",
                    "Bottom N",
                    "Include Only"
                ]

                selected_chart["filter_type"] = st.sidebar.selectbox(
                    "Filter Type",
                    filter_options,
                    index=filter_options.index(
                        selected_chart["filter_type"]
                    )
                )

                if selected_chart["filter_type"] in ["Top N", "Bottom N"]:

                    selected_chart["top_n"] = st.sidebar.slider(
                        "Select N",
                        1,
                        20,
                        selected_chart["top_n"]
                    )

                elif selected_chart["filter_type"] == "Include Only":

                    filter_column = st.sidebar.selectbox(
                        "Filter Column",
                        df.columns,
                        index=0 if selected_chart["filter_column"] is None
                        else list(df.columns).index(
                            selected_chart["filter_column"]
                        ),
                        key=f"filter_col_{selected_chart['id']}"
                    )

                    old_filter_col = selected_chart["filter_column"]

                    if old_filter_col != filter_column:
                        selected_chart["selected_values"] = []

                    selected_chart["filter_column"] = filter_column

                    unique_values = sorted(
                        df[filter_column]
                        .dropna()
                        .astype(str)
                        .unique()
                    )

                    valid_defaults = [

                        value
                        for value in selected_chart.get(
                            "selected_values",
                            []
                        )

                        if value in unique_values
                    ]

                    selected_chart["selected_values"] = st.sidebar.multiselect(
                        "Select Values",
                        unique_values,
                        default=valid_defaults,
                        key=f"filter_values_{selected_chart['id']}"
                    )

            # ================= MULTI CHART DASHBOARD =================

            color_palettes = {

                "Default": ["#636EFA", "#EF553B", "#00CC96"],

                "Vibrant": ["#FF6B6B", "#4ECDC4", "#FFE66D"],

                "Pastel": ["#A8DADC", "#FFCAD4", "#CDB4DB"],

                "Neon": ["#00F5D4", "#F15BB5", "#9B5DE5"]
            }

            pie_palettes = {

                "Plotly": qualitative.Plotly,
                "Bold": qualitative.Bold,
                "Pastel": qualitative.Pastel,
                "Dark24": qualitative.Dark24,
                "Light24": qualitative.Light24,
                "Safe": qualitative.Safe,
                "Vivid": qualitative.Vivid,
                "Prism": qualitative.Prism
            }

            # ================= FIX OLD CHART OBJECTS =================

            for chart_obj in st.session_state["charts"]:

                if "aggregations" not in chart_obj:
                    chart_obj["aggregations"] = {}

                if "subtitle" not in chart_obj:
                    chart_obj["subtitle"] = ""

                if "chart_theme" not in chart_obj:
                    chart_obj["chart_theme"] = "Default"

                if "pie_theme" not in chart_obj:
                    chart_obj["pie_theme"] = "Vibrant"

                if "x_label" not in chart_obj:
                    chart_obj["x_label"] = chart_obj["group_col"]

                if "y_label" not in chart_obj:
                    chart_obj["y_label"] = "Value"

                if "axis_font_size" not in chart_obj:
                    chart_obj["axis_font_size"] = 16

                if "axis_color" not in chart_obj:
                    chart_obj["axis_color"] = "#FFFFFF"

                if "axis_bold" not in chart_obj:
                    chart_obj["axis_bold"] = True

                if "hover_bg" not in chart_obj:
                    chart_obj["hover_bg"] = "#111827"

                if "hover_font_color" not in chart_obj:
                    chart_obj["hover_font_color"] = "#FFFFFF"

                if "hover_font_size" not in chart_obj:
                    chart_obj["hover_font_size"] = 14

                if "filter_type" not in chart_obj:
                    chart_obj["filter_type"] = "None"

                if "top_n" not in chart_obj:
                    chart_obj["top_n"] = 5

                if "selected_values" not in chart_obj:
                    chart_obj["selected_values"] = []

                if "filter_column" not in chart_obj:
                    chart_obj["filter_column"] = None

            charts = st.session_state["charts"]

            col1, col2 = st.columns(2)

            for i, chart_obj in enumerate(charts):

                if (
                        "group_col" not in chart_obj
                        or chart_obj["group_col"] not in df.columns
                ):
                    chart_obj["group_col"] = df.columns[0]

                group_col = chart_obj["group_col"]

                # FIX INVALID NUMERIC COLUMNS
                chart_obj["numeric_cols"] = [

                    col for col in chart_obj["numeric_cols"]

                    if col in df.columns
                ]

                numeric_cols = chart_obj["numeric_cols"]

                chart_type = chart_obj["chart_type"]

                has_numeric = len(numeric_cols) > 0

                # ================= EMPTY STATE =================

                if not has_numeric:

                    container = col1 if i % 2 == 0 else col2

                    with container:

                        st.markdown(
                            '<div class="card">',
                            unsafe_allow_html=True
                        )

                        # ACTION BUTTONS
                        col_edit, col_delete = st.columns(2)

                        with col_edit:

                            if st.button(
                                    "✏️ Edit",
                                    key=f"edit_empty_{chart_obj['id']}"
                            ):
                                st.session_state["selected_chart"] = chart_obj["id"]
                                st.rerun()

                        with col_delete:

                            if st.button(
                                    "🗑️ Delete",
                                    key=f"delete_empty_{chart_obj['id']}"
                            ):
                                st.session_state["charts"] = [

                                    c for c in st.session_state["charts"]

                                    if c["id"] != chart_obj["id"]
                                ]

                                st.rerun()

                        st.subheader(chart_obj["title"])

                        st.warning(
                            "⚠️ Please select at least one numeric column"
                        )

                        st.markdown(
                            '</div>',
                            unsafe_allow_html=True
                        )

                    continue

                # ================= AGGREGATION =================

                agg_dict = {

                    col: chart_obj["aggregations"].get(col, "sum")
                    for col in numeric_cols
                }

                filtered_df = df.copy()

                if (
                        chart_obj["filter_type"] == "Include Only"
                        and chart_obj["selected_values"]
                ):
                    filter_col = chart_obj["filter_column"]

                    filtered_df = filtered_df[
                        filtered_df[filter_col]
                        .astype(str)
                        .isin(chart_obj["selected_values"])
                    ]

                plot_data = (
                    filtered_df.groupby(group_col, as_index=False)
                    .agg(agg_dict)
                )

                # ================= FILTERS =================

                if chart_obj["filter_type"] == "Top N":

                    plot_data = plot_data.sort_values(
                        by=numeric_cols[0],
                        ascending=False
                    ).head(chart_obj["top_n"])

                elif chart_obj["filter_type"] == "Bottom N":

                    plot_data = plot_data.sort_values(
                        by=numeric_cols[0],
                        ascending=True
                    ).head(chart_obj["top_n"])


                # ================= CHART TYPES =================

                if chart_type == "bar":

                    fig = px.bar(
                        plot_data,
                        x=group_col,
                        y=numeric_cols
                    )

                elif chart_type == "line":

                    fig = px.line(
                        plot_data,
                        x=group_col,
                        y=numeric_cols
                    )

                elif chart_type == "area":

                    fig = px.area(
                        plot_data,
                        x=group_col,
                        y=numeric_cols
                    )

                elif chart_type == "pie":

                    fig = px.pie(
                        plot_data,
                        names=group_col,
                        values=numeric_cols[0],
                        color_discrete_sequence=
                        pie_palettes[chart_obj["pie_theme"]]
                    )

                elif chart_type == "histogram":

                    fig = px.histogram(
                        df,
                        x=numeric_cols[0]
                    )

                # ================= COLORS =================

                if chart_type != "pie":

                    colors = color_palettes[
                        chart_obj["chart_theme"]
                    ]

                    for idx, trace in enumerate(fig.data):

                        color = colors[idx % len(colors)]

                        if hasattr(trace, "marker"):
                            trace.marker.color = color

                        if hasattr(trace, "line"):
                            trace.line.color = color

                # ================= FONT =================

                font_family = (
                    "Arial Black"
                    if chart_obj["axis_bold"]
                    else "Arial"
                )

                # ================= HOVER =================

                fig.update_layout(

                    template="plotly_dark",

                    xaxis_title=chart_obj["x_label"],
                    yaxis_title=chart_obj["y_label"],

                    hoverlabel=dict(
                        bgcolor=chart_obj["hover_bg"],
                        font_size=chart_obj["hover_font_size"],
                        font_color=chart_obj["hover_font_color"]
                    ),

                    xaxis=dict(

                        title_font=dict(
                            size=chart_obj["axis_font_size"],
                            color=chart_obj["axis_color"],
                            family=font_family
                        ),

                        tickfont=dict(
                            size=chart_obj["axis_font_size"],
                            color=chart_obj["axis_color"]
                        )
                    ),

                    yaxis=dict(

                        title_font=dict(
                            size=chart_obj["axis_font_size"],
                            color=chart_obj["axis_color"],
                            family=font_family
                        ),

                        tickfont=dict(
                            size=chart_obj["axis_font_size"],
                            color=chart_obj["axis_color"]
                        )
                    )
                )

                # ================= HOVER TEMPLATES =================

                if chart_type in ["line", "bar", "area"]:

                    fig.update_traces(

                        hovertemplate=
                        "<b>%{x}</b><br>" +
                        "Value: %{y:,.2f}<br>" +
                        "<extra></extra>"
                    )

                elif chart_type == "pie":

                    fig.update_traces(

                        hovertemplate=
                        "<b>%{label}</b><br>" +
                        "Value: %{value:,.2f}<br>" +
                        "Percent: %{percent}<br>" +
                        "<extra></extra>"
                    )

                # ================= CARD =================

                container = col1 if i % 2 == 0 else col2

                with container:

                    st.markdown(
                        '<div class="card">',
                        unsafe_allow_html=True
                    )

                    # ================= ACTION BUTTONS =================

                    col_edit, col_delete = st.columns(2)

                    with col_edit:

                        if st.button(
                                "✏️ Edit",
                                key=f"edit_btn_{i}_{chart_obj['id']}"
                        ):
                            st.session_state["selected_chart"] = chart_obj["id"]
                            st.rerun()

                    with col_delete:

                        if st.button(
                                "🗑️ Delete",
                                key=f"delete_btn_{i}_{chart_obj['id']}"
                        ):

                            st.session_state["charts"] = [

                                c for c in st.session_state["charts"]

                                if c["id"] != chart_obj["id"]
                            ]

                            # RESET SELECTED CHART
                            if st.session_state["selected_chart"] == chart_obj["id"]:
                                st.session_state["selected_chart"] = None

                            st.rerun()

                    # ================= TITLE =================

                    st.subheader(chart_obj["title"])

                    if chart_obj["subtitle"]:
                        st.caption(chart_obj["subtitle"])

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"chart_{chart_obj['id']}"
                    )


    with tab3:
        if "df" not in st.session_state:
            st.info("📁 Upload a file first")
        else:
            df = st.session_state['df']

            val_cols = df.select_dtypes(include='number').columns

            # ===================== ANALYSIS CARD =====================
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("📈 Single Column Analysis")

            column = st.selectbox('Select Column', val_cols)

            with st.expander("⚙️ Advanced Options"):
                agg = st.selectbox(
                    'Aggregation',
                    ['sum', 'mean', 'max', 'min', 'count']
                )

            value = getattr(df[column], agg)()
            st.metric(label=f"{agg.upper()} of {column}", value=f"{value:,.2f}")

            st.markdown('</div>', unsafe_allow_html=True)

            # ===================== INSIGHTS CARD =====================
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("🤖 Auto Insights")

            mean_val = df[column].mean()
            tolerance = 0.10 * mean_val

            upper = mean_val + tolerance
            lower = mean_val - tolerance

            if value > upper:
                st.success(f"{column} is SIGNIFICANTLY above average 🚀")
            elif value < lower:
                st.warning(f"{column} is SIGNIFICANTLY below average ⚠️")
            else:
                st.info(f"{column} is close to average 👍")

            st.markdown('</div>', unsafe_allow_html=True)

            # ===================== RANDOM INSIGHT CARD =====================
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("🎲 Random Insight")

            num_cols = df.select_dtypes(include='number').columns

            if len(num_cols) > 0:
                import random

                random_col = random.choice(num_cols)

                max_val = df[random_col].max()
                min_val = df[random_col].min()

                st.info(
                    f"Did you know? 🤔\n\n"
                    f"In **{random_col}**, the highest value is **{max_val}** "
                    f"and the lowest is **{min_val}**."
                )
            else:
                st.warning("No numeric data available")

            st.markdown('</div>', unsafe_allow_html=True)

            # ===================== LOGOUT CARD =====================
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("🔐 Session")

            if st.button('Logout'):
                posthog.capture(
                    distinct_id=get_user_id(),
                    event="user_logged_out"
                )

                del st.session_state['token']
                del st.session_state['user_email']

                if "df" in st.session_state:
                    del st.session_state['df']

                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    with tab4:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🤖 Machine Learning Predictions")

        if "df" not in st.session_state:

            st.info("📁 Upload a dataset first")

        else:

            df = st.session_state["df"].copy()

            # =========================
            # TARGET COLUMN
            # =========================

            target_column = st.selectbox(
                "🎯 Select Target Column",
                df.columns
            )

            # =========================
            # DETECT PROBLEM TYPE
            # =========================

            if is_numeric_dtype(df[target_column]):

                unique_values = df[target_column].nunique()

                if unique_values <= 15:
                    problem_type = "classification"
                else:
                    problem_type = "regression"

            else:

                problem_type = "classification"

            st.info(f"Detected Problem Type: {problem_type}")

            # =========================
            # MODEL SELECTION
            # =========================

            if problem_type == "regression":

                model_name = st.selectbox(
                    "🧠 Select Model",
                    [
                        "linear_regression",
                        "decision_tree"
                    ]
                )

            else:

                model_name = st.selectbox(
                    "🧠 Select Model",
                    [
                        "logistic_regression",
                        "decision_tree"
                    ]
                )

            # =========================
            # TRAIN BUTTON
            # =========================

            if st.button("🚀 Train Model"):

                # =========================
                # CLEANING
                # =========================

                numeric_cols = df.select_dtypes(
                    include=['int64', 'float64']
                ).columns

                categorical_cols = df.select_dtypes(
                    include=['object']
                ).columns

                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mean())

                for col in categorical_cols:
                    df[col] = df[col].fillna(df[col].mode()[0])

                # =========================
                # REMOVE HIGH CARDINALITY
                # =========================

                high_cardinality_cols = []

                for col in categorical_cols:

                    if (
                            df[col].nunique() > 50
                            and col != target_column
                    ):
                        high_cardinality_cols.append(col)

                if len(high_cardinality_cols) > 0:
                    df = df.drop(
                        columns=high_cardinality_cols
                    )

                    st.warning(
                        f"Dropped High Cardinality Columns: "
                        f"{high_cardinality_cols}"
                    )

                # =========================
                # FEATURES
                # =========================

                X = df.drop(columns=[target_column])

                y = df[target_column]

                X = pd.get_dummies(
                    X,
                    drop_first=True
                )

                # =========================
                # SPLIT
                # =========================

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )

                # =========================
                # SCALE
                # =========================

                scaler = StandardScaler()

                X_train_scaled = scaler.fit_transform(
                    X_train
                )

                X_test_scaled = scaler.transform(
                    X_test
                )

                # =========================
                # MODEL
                # =========================

                if problem_type == "regression":

                    if model_name == "linear_regression":

                        model = LinearRegression()

                    else:

                        model = DecisionTreeRegressor(
                            random_state=42
                        )

                else:

                    if model_name == "logistic_regression":

                        model = LogisticRegression(
                            max_iter=1000
                        )

                    else:

                        model = DecisionTreeClassifier(
                            random_state=42
                        )

                # =========================
                # TRAIN
                # =========================

                model.fit(
                    X_train_scaled,
                    y_train
                )

                # =========================
                # PREDICTIONS
                # =========================

                y_pred = model.predict(
                    X_test_scaled
                )

                # =========================
                # STORE MODEL
                # =========================

                st.session_state["trained_model"] = model

                st.session_state["feature_columns"] = X.columns.tolist()

                st.session_state["scaler"] = scaler

                st.session_state["problem_type"] = problem_type

                st.session_state["target_column"] = target_column

                st.session_state["training_df"] = df

                # =========================
                # RESULTS
                # =========================

                st.success("✅ Model Trained Successfully")

                if problem_type == "regression":

                    r2 = r2_score(
                        y_test,
                        y_pred
                    )

                    mae = mean_absolute_error(
                        y_test,
                        y_pred
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "R2 Score",
                        f"{r2:.4f}"
                    )

                    col2.metric(
                        "MAE",
                        f"{mae:.2f}"
                    )

                else:

                    accuracy = accuracy_score(
                        y_test,
                        y_pred
                    )

                    st.metric(
                        "Accuracy",
                        f"{accuracy:.4f}"
                    )

            # =========================
            # USER PREDICTION SECTION
            # =========================

            if "trained_model" in st.session_state:

                st.markdown("---")

                st.subheader("🔮 Predict New Data")

                model = st.session_state["trained_model"]

                scaler = st.session_state["scaler"]

                feature_columns = st.session_state["feature_columns"]

                training_df = st.session_state["training_df"]

                target_column = st.session_state["target_column"]

                user_inputs = {}

                input_columns = [

                    col for col in training_df.columns

                    if col != target_column
                ]

                # =========================
                # INPUTS
                # =========================

                for col in input_columns:

                    if is_numeric_dtype(training_df[col]):

                        user_inputs[col] = st.number_input(
                            col,
                            value=float(
                                training_df[col].mean()
                            ),
                            key=f"input_{col}"
                        )

                    else:

                        user_inputs[col] = st.selectbox(
                            col,
                            training_df[col].astype(str).unique(),
                            key=f"input_{col}"
                        )

                # =========================
                # PREDICT BUTTON
                # =========================

                if st.button("🎯 Predict"):
                    input_df = pd.DataFrame(
                        [user_inputs]
                    )

                    input_df = pd.get_dummies(
                        input_df,
                        drop_first=True
                    )

                    # ALIGN FEATURES

                    input_df = input_df.reindex(
                        columns=feature_columns,
                        fill_value=0
                    )

                    # SCALE

                    input_scaled = scaler.transform(
                        input_df
                    )

                    # PREDICT

                    prediction = model.predict(
                        input_scaled
                    )

                    st.success(
                        f"Prediction Result: {prediction[0]}"
                    )

        st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.caption(
    "SalesPulse v1.0 • AI-Powered Decision Intelligence"
)
