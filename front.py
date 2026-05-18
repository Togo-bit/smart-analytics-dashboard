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

st.set_page_config(layout='wide')

st.markdown("""
<style>
.card {
    background: linear-gradient(145deg, #1E293B, #0F172A);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
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

st.markdown("""
# 📊 SalesPulse
### Upload • Analyze • Get Insights
""")

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

    if "charts" not in st.session_state:
        st.session_state["charts"] = []

    if "selected_chart" not in st.session_state:
        st.session_state["selected_chart"] = None

    st.subheader('Upload & Analyze Your Data')

    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "📊 Dashboard", "📈 Analysis", "🤖 Predictions"])

    with tab1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📁 Upload & Preview Data")

        upload_file = st.file_uploader('Upload a CSV file', type=['csv'])

        if upload_file:
            try:
                df = pd.read_csv(upload_file)

                posthog.capture(
                    distinct_id=get_user_id(),
                    event="dataset_uploaded",
                    properties={
                        "rows": df.shape[0],
                        "columns": df.shape[1]
                    }
                )

                # CHECK IF NEW FILE IS DIFFERENT
                new_file_name = upload_file.name

                if "last_uploaded_file" not in st.session_state:
                    st.session_state["last_uploaded_file"] = ""

                # ONLY RESET WHEN NEW FILE IS UPLOADED
                if st.session_state["last_uploaded_file"] != new_file_name:
                    st.session_state["charts"] = []
                    st.session_state["selected_chart"] = None

                    st.session_state["last_uploaded_file"] = new_file_name

                st.session_state['df'] = df

                st.markdown("### 🔍 Preview")
                st.dataframe(df.head())

                col1, col2 = st.columns(2)
                col1.metric('Rows', df.shape[0])
                col2.metric('Columns', df.shape[1])
            except UnicodeDecodeError as e:

                posthog.capture(
                    distinct_id=get_user_id(),
                    event="upload_error",
                    properties={
                        "error": str(e)
                    }
                )

                st.error("⚠️ File encoding issue. Try re-saving your CSV as UTF-8 in Excel.")
        else:
            st.info("Upload a CSV file to see preview")

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

                selected_chart["group_col"] = st.sidebar.selectbox(
                    "Group By",
                    df.columns,
                    index=list(df.columns).index(selected_chart["group_col"])
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

                selected_chart["filter_type"] = st.sidebar.selectbox(
                    "Filter Type",
                    ["None", "Top N", "Bottom N"],
                    index=["None", "Top N", "Bottom N"].index(
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

            charts = st.session_state["charts"]

            col1, col2 = st.columns(2)

            for i, chart_obj in enumerate(charts):

                group_col = chart_obj["group_col"]
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

                plot_data = (
                    df.groupby(group_col, as_index=False)
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
