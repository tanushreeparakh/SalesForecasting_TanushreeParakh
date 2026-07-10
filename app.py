# -----------------------------------------------------------
# Sales Forecasting & Demand Intelligence System
# Internship Project
# Developed by: Tanushree Parakh
# -----------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Project Paths
# ----------------------------------------------------------

BASE_DIR = Path(__file__).parent
TRAIN_DATA = BASE_DIR / "train.csv"
PROCESSED_DATA = BASE_DIR / "processed_sales.csv"
MODEL_COMPARISON = BASE_DIR / "model_comparison.csv"
CATEGORY_FORECAST = BASE_DIR / "category_region_forecast.csv"
ANOMALY_FILE = BASE_DIR / "weekly_anomalies.csv"
SEGMENT_FILE = BASE_DIR / "product_segments.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

@st.cache_data
def load_sales_data():
    df = pd.read_csv(TRAIN_DATA)

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%d/%m/%Y"
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    df["Month"] = pd.Categorical(
        df["Month"],
        categories=month_order,
        ordered=True
    )

    df["Quarter"] = df["Order Date"].dt.quarter

    df["Shipping Days"] = (
        df["Ship Date"] -
        df["Order Date"]
    ).dt.days
    return df

@st.cache_data
def load_model_results():

    comparison = pd.read_csv(MODEL_COMPARISON)
    forecast = pd.read_csv(CATEGORY_FORECAST)
    anomalies = pd.read_csv(ANOMALY_FILE)
    anomalies["Date"] = pd.to_datetime(
        anomalies["Date"]
    )

    segments = pd.read_csv(SEGMENT_FILE)
    return comparison, forecast, anomalies, segments

@st.cache_resource
def load_model():

    try:
        model = joblib.load(MODEL_PATH)
        return model
    except:
        return None

# ----------------------------------------------------------
# Load Everything
# ----------------------------------------------------------

df = load_sales_data()
comparison_df, forecast_df, anomaly_df, segment_df = load_model_results()
model = load_model()

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(

    "Select Page",

    [
        "Sales Dashboard",
        "Forecast Explorer",
        "Anomaly Detection",
        "Product Segmentation",
        "About Project"
    ]
)

# ----------------------------------------------------------
# Dashboard Title
# ----------------------------------------------------------

st.title("Sales Forecasting & Demand Intelligence System")
st.caption(
    "Interactive dashboard for retail sales analysis, forecasting, anomaly detection and product demand segmentation."
)
st.divider()

# ----------------------------------------------------------
# SALES DASHBOARD
# ----------------------------------------------------------

if page == "Sales Dashboard":
    st.header("Sales Overview Dashboard")

    # -------------------------------
    # KPI Cards
    # -------------------------------

    total_sales = df["Sales"].sum()
    total_orders = len(df)
    avg_sales = df["Sales"].mean()
    avg_shipping = df["Shipping Days"].mean()
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )

    c2.metric(
        "Orders",
        f"{total_orders:,}"
    )

    c3.metric(
        "Average Order Value",
        f"${avg_sales:,.2f}"
    )

    c4.metric(
        "Average Shipping Days",
        f"{avg_shipping:.2f}"
    )

    st.divider()

    # -------------------------------
    # Filters
    # -------------------------------

    left, right = st.columns(2)
    with left:
        year = st.selectbox(
            "Select Year",
            sorted(df["Year"].unique())
        )

    with right:
        region = st.selectbox(
            "Select Region",
            ["All"] + sorted(df["Region"].unique())
        )

    category = st.selectbox(
        "Select Category",
        ["All"] + sorted(df["Category"].unique())
    )

    filtered = df[df["Year"] == year]
    if region != "All":
        filtered = filtered[
            filtered["Region"] == region
        ]
    if category != "All":
        filtered = filtered[
            filtered["Category"] == category
        ]

    # -------------------------------
    # Monthly Sales Trend
    # -------------------------------

    monthly = (
        filtered
        .groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Sales",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------------
    # Category and Region
    # -------------------------------

    left, right = st.columns(2)
    category_sales = (
        filtered
        .groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig_category = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        text_auto=".2s",
        title="Sales by Category"
    )

    fig_category.update_layout(
        template="plotly_white"
    )

    left.plotly_chart(
        fig_category,
        use_container_width=True
    )

    region_sales = (
        filtered
        .groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig_region = px.pie(
        region_sales,
        names="Region",
        values="Sales",
        hole=0.45,
        title="Regional Sales Distribution"
    )

    right.plotly_chart(
        fig_region,
        use_container_width=True
    )

    # -------------------------------
    # Sub-category Analysis
    # -------------------------------

    subcategory = (
        filtered
        .groupby("Sub-Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_sub = px.bar(
        subcategory,
        x="Sales",
        y="Sub-Category",
        orientation="h",
        title="Sales by Sub-Category"
    )

    fig_sub.update_layout(
        template="plotly_white",
        height=600
    )

    st.plotly_chart(
        fig_sub,
        use_container_width=True
    )

# ----------------------------------------------------------
# FORECAST EXPLORER
# ----------------------------------------------------------

elif page == "Forecast Explorer":
    st.header("Sales Forecast Explorer")
    st.write(
        """
        This page displays the forecasting results generated using the
        best-performing forecasting model selected during model comparison.
        """
    )

    st.divider()

    # ------------------------------------------------------
    # Model Performance
    # ------------------------------------------------------

    st.subheader("Model Comparison")
    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # Best Model
    # ------------------------------------------------------

    best_model = comparison_df.sort_values(
        by="RMSE"
    ).iloc[0]
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Selected Model",
        best_model["Model"]
    )

    col2.metric(
        "MAE",
        round(best_model["MAE"], 2)
    )

    col3.metric(
        "RMSE",
        round(best_model["RMSE"], 2)
    )

    col4.metric(
        "MAPE",
        f"{best_model['MAPE']:.2f}%"
    )

    st.success(
        f"Production Recommendation: {best_model['Model']}"
    )

    st.divider()

    # ------------------------------------------------------
    # Forecast Selection
    # ------------------------------------------------------

    st.subheader("Forecast Explorer")
    segment = st.selectbox(
        "Select Category / Region",
        forecast_df["Segment"].unique()
    )

    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    selected = forecast_df[
        forecast_df["Segment"] == segment
    ]

    if not selected.empty:
        values = [
            float(selected["Month 1"].iloc[0]),
            float(selected["Month 2"].iloc[0]),
            float(selected["Month 3"].iloc[0])
        ]

        months = [
            "Month 1",
            "Month 2",
            "Month 3"
        ]

        plot_df = pd.DataFrame({
            "Month": months[:horizon],
            "Forecast": values[:horizon]
        })

        fig = px.line(
            plot_df,
            x="Month",
            y="Forecast",
            markers=True,
            title=f"Forecast for {segment}"
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Forecast Month",
            yaxis_title="Predicted Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            plot_df,
            use_container_width=True
        )

        avg_forecast = float(
            selected["Average Forecast"].iloc[0]
        )

        st.metric(
            "Average Forecast",
            f"${avg_forecast:,.2f}"
        )

    st.divider()

    # ------------------------------------------------------
    # Forecast Comparison Chart
    # ------------------------------------------------------

    st.subheader("Comparison Across All Segments")
    comparison_long = forecast_df.melt(
        id_vars="Segment",
        value_vars=[
            "Month 1",
            "Month 2",
            "Month 3"
        ],
        var_name="Forecast Month",
        value_name="Forecast"
    )

    fig_compare = px.line(
        comparison_long,
        x="Forecast Month",
        y="Forecast",
        color="Segment",
        markers=True,
        title="Category and Region Forecast Comparison"
    )

    fig_compare.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # Forecast Table
    # ------------------------------------------------------

    st.subheader("Forecast Summary")
    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Forecast CSV",
        csv,
        file_name="forecast_results.csv",
        mime="text/csv"
    )

    st.info(
        """
        The forecasts shown above are generated using the best forecasting
        model selected based on RMSE, MAE and MAPE evaluation metrics.
        """
    )

# ----------------------------------------------------------
# ANOMALY DETECTION
# ----------------------------------------------------------

elif page == "Anomaly Detection":
    st.header("Sales Anomaly Detection")
    st.write("""
    This page highlights unusual sales behaviour detected using
    Isolation Forest and Z-Score based anomaly detection.
    """)
    st.divider()

    # -------------------------------
    # KPI Cards
    # -------------------------------

    total_weeks = len(anomaly_df)
    isolation_count = len(
        anomaly_df[anomaly_df["Anomaly"] == -1]
    )

    zscore_count = len(
        anomaly_df[anomaly_df["Z_Anomaly"] == 1]
    )

    average_sales = anomaly_df["Sales"].mean()
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Weeks",
        total_weeks
    )

    c2.metric(
        "Isolation Forest",
        isolation_count
    )

    c3.metric(
        "Z-Score",
        zscore_count
    )

    c4.metric(
        "Average Weekly Sales",
        f"${average_sales:,.0f}"
    )

    st.divider()

    # -------------------------------
    # Isolation Forest Chart
    # -------------------------------

    st.subheader("Isolation Forest Detection")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=anomaly_df["Date"],
            y=anomaly_df["Sales"],
            mode="lines",
            name="Weekly Sales"
        )
    )

    anomaly_points = anomaly_df[
        anomaly_df["Anomaly"] == -1
    ]

    fig.add_trace(
        go.Scatter(
            x=anomaly_points["Date"],
            y=anomaly_points["Sales"],
            mode="markers",
            marker=dict(
                size=10,
                color="red"
            ),
            name="Isolation Forest Anomaly"
        )
    )

    fig.update_layout(
        title="Weekly Sales with Isolation Forest Anomalies",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # -------------------------------
    # Z Score Chart
    # -------------------------------

    st.subheader("Z-Score Detection")
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=anomaly_df["Date"],
            y=anomaly_df["Sales"],
            mode="lines",
            name="Weekly Sales"
        )
    )

    z_points = anomaly_df[
        anomaly_df["Z_Anomaly"] == 1
    ]

    fig2.add_trace(
        go.Scatter(
            x=z_points["Date"],
            y=z_points["Sales"],
            mode="markers",
            marker=dict(
                size=10,
                color="orange"
            ),
            name="Z Score Anomaly"
        )
    )

    fig2.update_layout(
        title="Weekly Sales with Z-Score Anomalies",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    # -------------------------------
    # Comparison
    # -------------------------------

    st.subheader("Method Comparison")
    comparison = pd.DataFrame({
        "Method": [
            "Isolation Forest",
            "Z-Score"
        ],
        "Detected Anomalies": [
            isolation_count,
            zscore_count
        ]
    })

    fig_compare = px.bar(
        comparison,
        x="Method",
        y="Detected Anomalies",
        text="Detected Anomalies",
        title="Comparison of Anomaly Detection Methods"
    )

    fig_compare.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

    st.divider()

    # -------------------------------
    # Detailed Table
    # -------------------------------

    st.subheader("Detected Weekly Anomalies")
    st.dataframe(
        anomaly_df,
        use_container_width=True
    )

    csv = anomaly_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Anomaly Report",
        csv,
        file_name="weekly_anomalies.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("Business Interpretation")
    st.markdown("""
**Isolation Forest**
- Detects unusual sales behaviour using machine learning.
- Captures both sudden sales spikes and unexpected drops.

**Z-Score Method**
- Detects statistically significant deviations from the rolling average.
- Easy to interpret and suitable for monitoring operations.

**Business Insight**
- Many anomalies occur during festive seasons and promotional campaigns.
- Sudden drops may indicate stock shortages, logistics delays, or supply chain disruptions.
- Monitoring anomalies enables proactive inventory planning and demand management.
""")
# ----------------------------------------------------------
# PRODUCT SEGMENTATION
# ----------------------------------------------------------

elif page == "Product Segmentation":
    st.header("Product Demand Segmentation")
    st.write("""
    Products have been grouped into demand clusters using
    K-Means clustering based on sales, growth rate,
    volatility and average order value.
    """)
    st.divider()

    # -------------------------------
    # Cluster Distribution
    # -------------------------------

    cluster_count = (
        segment_df.groupby("Cluster Name")
        .size()
        .reset_index(name="Products")
    )

    fig_cluster = px.bar(
        cluster_count,
        x="Cluster Name",
        y="Products",
        text="Products",
        title="Products in Each Demand Cluster"
    )

    fig_cluster.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig_cluster,
        use_container_width=True
    )

    # -------------------------------
    # PCA Visualization
    # -------------------------------

    st.subheader("Cluster Visualization")
    fig = px.scatter(
        segment_df,
        x="PC1",
        y="PC2",
        color="Cluster Name",
        hover_name="Sub-Category",
        size="Total Sales",
        title="Product Demand Clusters"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------------
    # Cluster Summary
    # -------------------------------

    st.subheader("Cluster Summary")
    summary = (
        segment_df.groupby("Cluster Name")
        .agg(
            Products=("Sub-Category", "count"),
            Average_Sales=("Total Sales", "mean"),
            Average_Growth=("Growth Rate", "mean"),
            Average_Volatility=("Volatility", "mean")
        )
        .round(2)
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.divider()
    st.subheader("Sub-Category Details")
    st.dataframe(
        segment_df,
        use_container_width=True
    )
    csv = segment_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Product Segments",
        csv,
        file_name="product_segments.csv",
        mime="text/csv"
    )
    st.divider()
    st.subheader("Business Recommendations")
    st.markdown("""
### High Volume, Stable Demand
- Maintain sufficient inventory.
- Ensure continuous stock availability.
- Monitor seasonal fluctuations.

### Growing Demand
- Increase procurement gradually.
- Allocate higher marketing budget.
- Expand warehouse allocation.

### Low Volume, High Volatility
- Maintain minimum inventory.
- Use demand-based ordering.
- Monitor promotions carefully.

### Declining Demand
- Reduce stock levels.
- Avoid over-ordering.
- Consider discount campaigns.
""")

# ----------------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------------

elif page == "About Project":
    st.header("About This Project")
    st.write("""
This application was developed as part of the Data Science Internship
Project on Sales Forecasting and Demand Intelligence.

The dashboard combines:

• Exploratory Data Analysis

• Time Series Forecasting

• Demand Forecast Comparison

• Sales Anomaly Detection

• Product Demand Segmentation

• Business Intelligence Dashboard

The project demonstrates how historical retail sales data can be
transformed into actionable insights for inventory planning,
demand forecasting and business decision making.
""")

    st.divider()
    st.subheader("Technologies Used")
    tech = pd.DataFrame({
        "Tool": [
            "Python",
            "Pandas",
            "NumPy",
            "Statsmodels",
            "Prophet",
            "XGBoost",
            "Scikit-learn",
            "Plotly",
            "Streamlit"
        ],
        "Purpose": [
            "Programming",
            "Data Analysis",
            "Numerical Computing",
            "SARIMA Forecasting",
            "Time Series Forecasting",
            "Machine Learning",
            "Clustering & Anomaly Detection",
            "Interactive Visualization",
            "Dashboard Deployment"
        ]
    })

    st.dataframe(
        tech,
        use_container_width=True
    )
    st.divider()
    st.success("Sales Forecasting & Demand Intelligence System completed successfully.")

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.divider()
st.markdown(
    """
<div style='text-align:center;'>
Sales Forecasting & Demand Intelligence System
Developed by <b>Tanushree Parakh</b>
Data Science Internship Project
</div>
""",
unsafe_allow_html=True
)