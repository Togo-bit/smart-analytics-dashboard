from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os

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

st.set_page_config(layout='wide')
st.markdown("""
# 📊 Smart Analytics Dashboard
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
                        st.session_state['token'] = response_data['token']
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
                    st.success('Registration Successful')
                else:
                    st.error(res.json().get("message", "Registration failed"))

            except requests.exceptions.RequestException:
                st.error("Server not reachable")
                st.stop()

else:
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}

    st.subheader('Upload & Analyze Your Data')

    tab1, tab2, tab3 = st.tabs(["📁 Upload", "📊 Dashboard", "📈 Analysis"])

    with tab1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📁 Upload & Preview Data")

        upload_file = st.file_uploader('Upload a CSV file', type=['csv'])

        if upload_file:
            try:
                df = pd.read_csv(upload_file)

                st.session_state['df'] = df

                st.markdown("### 🔍 Preview")
                st.dataframe(df.head())

                col1, col2 = st.columns(2)
                col1.metric('Rows', df.shape[0])
                col2.metric('Columns', df.shape[1])
            except UnicodeDecodeError:
                st.error("⚠️ File encoding issue. Try re-saving your CSV as UTF-8 in Excel.")
        else:
            st.info("Upload a CSV file to see preview")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        if "df" not in st.session_state:
            st.info("📁 Upload a file first")
        else:
            df = st.session_state['df']

            # Sidebar Controls
            st.sidebar.markdown("## 🎛️ Controls")

            with st.sidebar.expander("Filters"):
                group_col = st.selectbox('Group by', df.columns)

                val_cols = df.select_dtypes(include='number').columns
                if len(val_cols) == 0:
                    st.warning("No numeric columns available.")
                    st.stop()

                numeric_cols = st.multiselect('Select Numeric Columns(Y-axis)',val_cols)

                if len(numeric_cols) == 0:
                    st.warning('Please select atleast one numeric column')
                    st.stop()

                if group_col in numeric_cols:
                    st.warning("Group column cannot be one of the selected numeric columns.")
                    st.stop()

            with st.sidebar.expander("Chart Settings"):

                st.markdown("### 📊 Column-wise Aggregation")

                agg_dict = {}
                for col in numeric_cols:
                    agg_dict[col] = st.selectbox(
                        f"{col}",
                        ['sum', 'mean', 'max', 'min', 'count'],
                        key=f"agg_{col}"
                    )

                st.markdown("### 📊 Chart Type")

                cols = st.columns(5)


                def chart_btn(icon, value, help_text):
                    st.markdown('<div class="chart-buttons">', unsafe_allow_html=True)

                    if st.button(icon, key=value, help=help_text, use_container_width=True):
                        st.session_state["chart_type"] = value

                    st.markdown('</div>', unsafe_allow_html=True)


                with cols[0]:
                    chart_btn("📈", "line", "Line Chart")
                with cols[1]:
                    chart_btn("📊", "bar", "Bar Chart")
                with cols[2]:
                    chart_btn("📉", "area", "Area Chart")
                with cols[3]:
                    chart_btn("🥧", "pie", "Pie Chart")
                with cols[4]:
                    chart_btn("📦", "histogram", "Histogram")

                chart = st.session_state.get("chart_type", "bar")

                st.caption(f"Selected: {chart.upper()}")

            with st.sidebar.expander("🔍 Advanced Filters"):

                filter_type = st.selectbox(
                    "Filter Type",
                    ["None", "Top N", "Bottom N", "Include Values"],
                    key="filter_type"
                )

                sort_col = None
                top_n = None
                selected_values = None

                if filter_type in ["Top N", "Bottom N"]:
                    sort_col = st.selectbox(
                        "Sort By Column",
                        numeric_cols,
                        key="sort_col_select"
                    )

                    top_n = st.slider(
                        "Select N",
                        1, 20, 5,
                        key="top_n_slider"
                    )

                elif filter_type == "Include Values":
                    selected_values = st.multiselect(
                        f"Select {group_col} values",
                        sorted(df[group_col].dropna().unique()),
                        key="include_values_multiselect"
                    )

            # Aggregation
            result = df.groupby(group_col).agg(agg_dict)

            # Reset index for filtering
            result = result.reset_index()

            # ================= APPLY FILTER =================
            if filter_type == "Top N" and sort_col:
                result = result.sort_values(by=sort_col, ascending=False).head(top_n)

            elif filter_type == "Bottom N" and sort_col:
                result = result.sort_values(by=sort_col, ascending=True).head(top_n)

            elif filter_type == "Include Values":
                if selected_values:
                    result = result[result[group_col].isin(selected_values)]

            result = result.set_index(group_col)

            # ===================== KPI CARD =====================
            kpi_col = st.selectbox("Select KPI Column", val_cols)

            total = df[kpi_col].sum()
            avg = df[kpi_col].mean()
            max_val = df[kpi_col].max()

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("📌 Key Metrics")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total", f"{total:,.2f}")
            col2.metric("Average", f"{avg:.2f}")
            col3.metric("Max", f"{max_val:.2f}")

            st.markdown('</div>', unsafe_allow_html=True)

            # ===================== GROWTH CARD =====================
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("📈 Growth Analysis")

            date_col = st.selectbox(
                "Select Date Column (optional)",
                ["None"] + list(df.columns)
            )

            if date_col == "None":
                st.info("Select a date column to calculate growth")
            else:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df[kpi_col] = pd.to_numeric(df[kpi_col], errors='coerce')

                    df_clean = df.dropna(subset=[date_col, kpi_col])

                    if df_clean.empty:
                        st.warning("Invalid date column or insufficient data")
                    else:
                        df_clean = df_clean.sort_values(date_col)

                        growth_series = df_clean[kpi_col].pct_change().dropna()

                        if len(growth_series) == 0:
                            st.warning("Not enough data for growth")
                        else:
                            avg_growth = growth_series.mean() * 100
                            st.metric("Avg Growth %", f"{avg_growth:.2f}%")

                except:
                    st.info("Growth not available")

            st.markdown('</div>', unsafe_allow_html=True)

            # ===================== CHART CARD =====================
            st.markdown('<div class="card">', unsafe_allow_html=True)

            agg_text = ", ".join([f"{col} ({agg_dict[col]})" for col in numeric_cols])
            st.subheader(f"📊 {agg_text} by {group_col}")

            result = result.reset_index().set_index(group_col)

            if chart == 'line':
                st.line_chart(result)
            elif chart == 'bar':
                st.bar_chart(result)
            elif chart == 'area':
                st.area_chart(result)
            elif chart == 'pie':
                if len(numeric_cols) > 1:
                    st.warning('Pie Chart only supports one column, Using first selected')

                pie_col = numeric_cols[0]

                pie_data = df.groupby(group_col)[pie_col].agg(agg_dict[pie_col]).reset_index()
                fig = px.pie(
                    pie_data,
                    names=group_col,
                    values=pie_col,
                    title=f"{agg_dict[pie_col].title()} of {pie_col}"
                )
                st.plotly_chart(fig)
            elif chart == 'histogram':
                fig = px.histogram(
                    df,
                    x=numeric_cols,
                    barmode='overlay',
                    title=f"Distrinution of {numeric_cols}"
                )
                st.plotly_chart(fig)

            st.markdown('</div>', unsafe_allow_html=True)

            # ===================== DOWNLOAD CARD =====================
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("Download Analysis")

            results_df = result.reset_index()
            csv = results_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label='Download Aggregated Results',
                data=csv,
                file_name='analysis_output.csv',
                mime='text/csv'
            )

            st.markdown('</div>', unsafe_allow_html=True)

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
                del st.session_state['token']
                if "df" in st.session_state:
                    del st.session_state['df']
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
