import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import shap
import io
from imblearn.over_sampling import SMOTE  # For class imbalance handling

# Inject Font Awesome and custom CSS
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        .fa-icon { margin-right: 8px; font-size: 1.2em; vertical-align: middle; }
        .success { color: #28a745; font-weight: bold; }
        .warning { color: #ffc107; font-weight: bold; }
        .title-header { font-size: 2em; font-weight: bold; margin-bottom: 15px; }
        .section-header { font-size: 1.5em; margin-bottom: 10px; }
        .subsection-header { font-size: 1.3em; margin-bottom: 8px; }
        .sidebar-metric { font-size: 1.1em; margin: 5px 0; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "action_history" not in st.session_state:
    st.session_state["action_history"] = []
if "df" not in st.session_state:
    st.session_state["df"] = None
if "refresh" not in st.session_state:
    st.session_state["refresh"] = 0

# Function to display dynamic sidebar summary
def display_sidebar_summary():
    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.markdown('<h2 class="section-header"><i class="fas fa-info-circle fa-icon"></i> Data Summary</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="sidebar-metric"><i class="fas fa-table fa-icon"></i> Rows: {df.shape[0]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sidebar-metric"><i class="fas fa-columns fa-icon"></i> Columns: {df.shape[1]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sidebar-metric"><i class="fas fa-exclamation-circle fa-icon"></i> Missing Values: {df.isnull().sum().sum()}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sidebar-metric"><i class="fas fa-list fa-icon"></i> Unique Values: {df.nunique().sum()}</p>', unsafe_allow_html=True)

# Page configuration
st.set_page_config(page_title="AI Data Analysis Assistant", layout="wide", initial_sidebar_state="expanded")
st.markdown('<h1 class="title-header"><i class="fas fa-chart-line fa-icon"></i> AI-Powered Data Analysis Assistant</h1>', unsafe_allow_html=True)
st.markdown("Upload a dataset to explore, preprocess, visualize, and prepare it for machine learning with advanced tools and actionable insights.")

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="section-header"><i class="fas fa-upload fa-icon"></i> Data Upload</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], help="Upload a dataset to start analysis.")

    st.markdown('<h2 class="section-header"><i class="fas fa-bullseye fa-icon"></i> Target Selection</h2>', unsafe_allow_html=True)
    target = None
    if st.session_state["df"] is not None:
        target = st.selectbox("Select Target Column", options=[""] + st.session_state["df"].columns.tolist(), help="Choose the target variable for analysis and modeling.")

    # Dynamic Data Summary
    display_sidebar_summary()

    # Button to View Data
    if st.session_state["df"] is not None:
        if st.button("View Current Data"):
            st.session_state["show_data"] = True
        else:
            st.session_state["show_data"] = False

# Load and store data
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.session_state["df"] = df
    st.session_state["action_history"].append("Loaded dataset")
    st.session_state["refresh"] += 1  # Trigger re-render
    st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Data loaded successfully!</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> Please upload a CSV or Excel file to begin.</p>', unsafe_allow_html=True)
    st.stop()

# Main content
df = st.session_state["df"].copy()

# Dataset Preview
with st.container():
    st.markdown('<h2 class="section-header"><i class="fas fa-table fa-icon"></i> Dataset Preview</h2>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

# Basic Data Analysis Section
with st.container():
    with st.expander("Basic Data Analysis"):
        st.markdown('<h2 class="section-header"><i class="fas fa-search fa-icon"></i> Dataset Overview</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Unique Values", df.nunique().sum())

        if st.checkbox("Show Statistical Summary", help="View descriptive statistics for all columns."):
            st.markdown('<h3 class="subsection-header"><i class="fas fa-calculator fa-icon"></i> Statistical Summary</h3>', unsafe_allow_html=True)
            st.dataframe(df.describe(include="all"), use_container_width=True)

        if st.checkbox("Show Column Types and Unique Values", help="View data types and unique value counts per column."):
            st.markdown('<h3 class="subsection-header"><i class="fas fa-columns fa-icon"></i> Column Types and Unique Values</h3>', unsafe_allow_html=True)
            dtypes_unique = pd.DataFrame({
                "Data Type": df.dtypes,
                "Unique Values": df.nunique(),
                "Missing Values": df.isnull().sum()
            })
            st.dataframe(dtypes_unique, use_container_width=True)

# Column Type Conversion Section
with st.container():
    with st.expander("Column Type Conversion"):
        st.markdown('<h2 class="section-header"><i class="fas fa-wrench fa-icon"></i> Modify Column Data Types</h2>', unsafe_allow_html=True)
        col_to_convert = st.selectbox("Select Column to Convert", options=[""] + df.columns.tolist(), help="Choose a column to change its data type.")
        if col_to_convert:
            new_type = st.selectbox("Select New Data Type", ["Numeric", "Categorical (String)", "Datetime"], help="Select the desired data type for the column.")
            if st.button("Convert Type"):
                try:
                    if new_type == "Numeric":
                        df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors="coerce")
                        st.session_state["action_history"].append(f"Converted column '{col_to_convert}' to Numeric")
                    elif new_type == "Categorical (String)":
                        df[col_to_convert] = df[col_to_convert].astype(str)
                        st.session_state["action_history"].append(f"Converted column '{col_to_convert}' to Categorical")
                    elif new_type == "Datetime":
                        df[col_to_convert] = pd.to_datetime(df[col_to_convert], errors="coerce")
                        st.session_state["action_history"].append(f"Converted column '{col_to_convert}' to Datetime")
                    st.session_state["df"] = df
                    st.session_state["refresh"] += 1  # Trigger re-render
                    st.markdown(f'<p class="success"><i class="fas fa-check-circle fa-icon"></i> Column \'{col_to_convert}\' converted to {new_type}.</p>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> Error converting column: {str(e)}</p>', unsafe_allow_html=True)
            st.dataframe(df[[col_to_convert]].head(), use_container_width=True)

# Data Insights Dashboard
with st.container():
    with st.expander("Data Insights Dashboard", expanded=True):
        st.markdown('<h2 class="section-header"><i class="fas fa-tachometer-alt fa-icon"></i> Data Quality Summary</h2>', unsafe_allow_html=True)
        df = st.session_state["df"].copy()
        skewness = df.select_dtypes(include="number").skew().sort_values(ascending=False)
        high_skew = skewness[abs(skewness) > 1]
        numeric_df = df.select_dtypes(include="number")
        z_scores = numeric_df.apply(zscore)
        outlier_flags = (z_scores.abs() > 3)
        total_outliers = outlier_flags.any(axis=1).sum()
        missing_total = df.isnull().sum().sum()
        missing_percent = round((missing_total / df.size) * 100, 2)
        cat_cols = df.select_dtypes(include=["object"]).columns
        high_cardinality = df[cat_cols].nunique()[df[cat_cols].nunique() > 20]
        variances = numeric_df.var()
        low_variance = variances[variances < 0.01]

        st.markdown('<h3 class="subsection-header"><i class="fas fa-info-circle fa-icon"></i> Key Metrics</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Skewed Features", len(high_skew))
        col2.metric("Total Outliers", total_outliers)
        col3.metric("Missing Data (%)", f"{missing_percent}%")
        col4.metric("High Cardinality Features", len(high_cardinality))

        # Action History
        st.markdown('<h3 class="subsection-header"><i class="fas fa-history fa-icon"></i> Action History</h3>', unsafe_allow_html=True)
        if st.session_state["action_history"]:
            action_df = pd.DataFrame({"Action": st.session_state["action_history"]})
            st.dataframe(action_df, use_container_width=True)
        else:
            st.markdown('<p class="warning"><i class="fas fa-info-circle fa-icon"></i> No actions performed yet.</p>', unsafe_allow_html=True)

        # Data Quality Issues and Suggestions
        st.markdown('<h3 class="subsection-header"><i class="fas fa-exclamation-triangle fa-icon"></i> Data Quality Issues and Suggestions</h3>', unsafe_allow_html=True)
        quality_issues = []
        suggestions = []
        if missing_total > 0:
            quality_issues.append(f"Missing values: {missing_total} ({missing_percent}%)")
            suggestions.append("Apply imputation (mode for categorical, KNN for numeric) or drop rows/columns.")
        if len(high_skew) > 0:
            quality_issues.append(f"Skewed numeric features: {len(high_skew)}")
            suggestions.append("Apply log transformation to skewed features.")
        if total_outliers > 0:
            quality_issues.append(f"Rows with outliers: {total_outliers}")
            suggestions.append("Remove outliers or apply robust scaling.")
        if not low_variance.empty:
            quality_issues.append(f"Low variance features: {len(low_variance)}")
            suggestions.append("Remove low variance features to reduce noise.")
        if not high_cardinality.empty:
            quality_issues.append(f"High cardinality categorical features: {len(high_cardinality)}")
            suggestions.append("Encode high cardinality features (e.g., target encoding) or reduce categories.")
        if target and df[target].dtype not in ["int64", "float64"] and df[target].nunique() <= 20:
            class_counts = df[target].value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 2:
                quality_issues.append(f"Class imbalance in target (ratio: {imbalance_ratio:.2f})")
                suggestions.append("Apply SMOTE, random oversampling, undersampling, or class weights.")

        if quality_issues:
            quality_df = pd.DataFrame({
                "Issue": quality_issues,
                "Suggestion": suggestions
            })
            st.dataframe(quality_df, use_container_width=True)
            # Apply Suggested Actions on latest data
            if missing_total > 0:
                if st.button("Apply Recommended Imputation"):
                    cat_cols = df.select_dtypes(include=["object"]).columns
                    num_cols = df.select_dtypes(include=["number"]).columns
                    if len(cat_cols) > 0 and df[cat_cols].isnull().sum().sum() > 0:
                        for col in cat_cols:
                            df[col] = df[col].fillna(df[col].mode()[0])
                    if len(num_cols) > 0 and df[num_cols].isnull().sum().sum() > 0:
                        imputer = KNNImputer(n_neighbors=5)
                        df[num_cols] = imputer.fit_transform(df[num_cols])
                    st.session_state["df"] = df
                    st.session_state["action_history"].append("Applied recommended imputation")
                    st.session_state["refresh"] += 1  # Trigger re-render
                    st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Applied recommended imputation.</p>', unsafe_allow_html=True)
            if len(high_skew) > 0:
                if st.button("Apply Log Transformation"):
                    for col in high_skew.index:
                        if df[col].min() > 0:
                            df[col] = np.log1p(df[col])
                    st.session_state["df"] = df
                    st.session_state["action_history"].append("Applied log transformation to skewed features")
                    st.session_state["refresh"] += 1  # Trigger re-render
                    st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Applied log transformation to skewed features.</p>', unsafe_allow_html=True)
            if total_outliers > 0:
                if st.button("Remove Outliers"):
                    df = df[~outlier_flags.any(axis=1)]
                    st.session_state["df"] = df
                    st.session_state["action_history"].append("Removed rows with outliers")
                    st.session_state["refresh"] += 1  # Trigger re-render
                    st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Removed rows with outliers.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> No major data quality issues detected.</p>', unsafe_allow_html=True)

# Missing Data Handling Section
with st.container():
    with st.expander("Missing Data Handling"):
        st.markdown('<h2 class="section-header"><i class="fas fa-broom fa-icon"></i> Handle Missing Values</h2>', unsafe_allow_html=True)
        df = st.session_state["df"].copy()
        if df.isnull().values.any():
            st.markdown('<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> Dataset contains missing values.</p>', unsafe_allow_html=True)
            st.markdown('<h3 class="subsection-header"><i class="fas fa-list-ul fa-icon"></i> Missing Values Summary</h3>', unsafe_allow_html=True)
            st.dataframe(df.isnull().sum().to_frame("Missing Values"), use_container_width=True)

            cat_cols = df.select_dtypes(include=["object"]).columns
            num_cols = df.select_dtypes(include=["number"]).columns

            # Categorical Missing Values
            if len(cat_cols) > 0 and df[cat_cols].isnull().sum().sum() > 0:
                st.markdown('<h3 class="subsection-header"><i class="fas fa-font fa-icon"></i> Categorical Columns</h3>', unsafe_allow_html=True)
                cat_strategy = st.selectbox(
                    "Select Strategy for Categorical Missing Values:",
                    ["Mode (Recommended)", "Constant Value"],
                    help="Mode is the most frequent value and is typically best for categorical data."
                )
                if cat_strategy == "Constant Value":
                    fill_value = st.text_input("Enter Constant Value", value="Unknown")
                if st.button("Apply Categorical Imputation"):
                    if cat_strategy == "Mode (Recommended)":
                        for col in cat_cols:
                            df[col] = df[col].fillna(df[col].mode()[0])
                        st.session_state["action_history"].append("Filled missing categorical values with mode")
                    else:
                        for col in cat_cols:
                            df[col] = df[col].fillna(fill_value)
                        st.session_state["action_history"].append(f"Filled missing categorical values with '{fill_value}'")
                    st.session_state["df"] = df
                    st.session_state["refresh"] += 1  # Trigger re-render
                    st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Applied categorical imputation.</p>', unsafe_allow_html=True)

            # Numeric Missing Values
            if len(num_cols) > 0 and df[num_cols].isnull().sum().sum() > 0:
                st.markdown('<h3 class="subsection-header"><i class="fas fa-calculator fa-icon"></i> Numeric Columns</h3>', unsafe_allow_html=True)
                num_strategy = st.selectbox(
                    "Select Strategy for Numeric Missing Values:",
                    ["KNN Imputation (Recommended)", "Mean", "Median", "Zero"],
                    help="KNN imputation uses nearest neighbors and is often effective for numeric data."
                )
                if num_strategy == "KNN Imputation (Recommended)":
                    n_neighbors = st.slider("Number of Neighbors for KNN", 1, 10, 5)
                if st.button("Apply Numeric Imputation"):
                    if num_strategy == "KNN Imputation (Recommended)":
                        imputer = KNNImputer(n_neighbors=n_neighbors)
                        df[num_cols] = imputer.fit_transform(df[num_cols])
                        st.session_state["action_history"].append(f"Applied KNN imputation to numeric columns (n_neighbors={n_neighbors})")
                    elif num_strategy == "Mean":
                        df = df.fillna(df.mean(numeric_only=True))
                        st.session_state["action_history"].append("Filled missing numeric values with mean")
                    elif num_strategy == "Median":
                        df = df.fillna(df.median(numeric_only=True))
                        st.session_state["action_history"].append("Filled missing numeric values with median")
                    elif num_strategy == "Zero":
                        df[num_cols] = df[num_cols].fillna(0)
                        st.session_state["action_history"].append("Filled missing numeric values with zero")
                    st.session_state["df"] = df
                    st.session_state["refresh"] += 1  # Trigger re-render
                    st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Applied numeric imputation.</p>', unsafe_allow_html=True)

            # General Options
            st.markdown('<h3 class="subsection-header"><i class="fas fa-cogs fa-icon"></i> General Options</h3>', unsafe_allow_html=True)
            general_strategy = st.selectbox(
                "Other Missing Value Actions:",
                ["Do nothing", "Drop rows", "Drop columns"],
                help="Choose to drop rows/columns or take no action."
            )
            if st.button("Apply General Strategy"):
                if general_strategy == "Drop rows":
                    df = df.dropna()
                    st.session_state["action_history"].append("Dropped rows with missing values")
                    st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Dropped rows with missing values.</p>', unsafe_allow_html=True)
                elif general_strategy == "Drop columns":
                    df = df.dropna(axis=1)
                    st.session_state["action_history"].append("Dropped columns with missing values")
                    st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Dropped columns with missing values.</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="warning"><i class="fas fa-info-circle fa-icon"></i> No action taken on missing values.</p>', unsafe_allow_html=True)
                st.session_state["df"] = df
                st.session_state["refresh"] += 1  # Trigger re-render
        else:
            st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> No missing values in the dataset.</p>', unsafe_allow_html=True)

# Outlier Detection Section
with st.container():
    with st.expander("Outlier Detection and Handling"):
        st.markdown('<h2 class="section-header"><i class="fas fa-exclamation-circle fa-icon"></i> Outliers per Numeric Feature (Z > 3)</h2>', unsafe_allow_html=True)
        df = st.session_state["df"].copy()
        numeric_df = df.select_dtypes(include="number")
        z_scores = numeric_df.apply(zscore)
        outlier_flags = (z_scores.abs() > 3)
        outlier_counts = outlier_flags.sum()
        st.dataframe(outlier_counts[outlier_counts > 0].to_frame("Outlier Count"), use_container_width=True)
        
        if st.checkbox("Show Rows with Outliers", help="View rows containing outliers based on Z-scores."):
            outlier_rows = df[outlier_flags.any(axis=1)]
            st.write(f"Found {outlier_rows.shape[0]} rows with at least one outlier.")
            st.dataframe(outlier_rows.head(10), use_container_width=True)

        if st.checkbox("Remove Outliers", help="Remove rows with outliers based on Z-scores."):
            df = df[~outlier_flags.any(axis=1)]
            st.session_state["df"] = df
            st.session_state["action_history"].append("Removed rows with outliers")
            st.session_state["refresh"] += 1  # Trigger re-render
            st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Removed rows with outliers.</p>', unsafe_allow_html=True)

# Advanced Visualizations Section
with st.container():
    with st.expander("Advanced Visualizations"):
        st.markdown('<h2 class="section-header"><i class="fas fa-chart-bar fa-icon"></i> Feature Visualizations</h2>', unsafe_allow_html=True)
        df = st.session_state["df"].copy()
        x_feature = st.selectbox("Select X Feature", options=[""] + df.columns.tolist(), help="Choose the feature for the x-axis.")
        y_feature = st.selectbox("Select Y Feature", options=[""] + df.columns.tolist(), help="Choose the feature for the y-axis.")

        # Determine best visualization types based on data types
        viz_options = []
        if x_feature and y_feature:
            x_type = df[x_feature].dtype
            y_type = df[y_feature].dtype

            if pd.api.types.is_numeric_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
                viz_options = ["Scatter Plot", "Pair Plot", "Correlation Heatmap"]
            elif pd.api.types.is_categorical_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
                viz_options = ["Box Plot", "Violin Plot", "Histogram (grouped by X)"]
            elif pd.api.types.is_categorical_dtype(x_type) and pd.api.types.is_categorical_dtype(y_type):
                viz_options = ["Histogram", "Bar Plot"]
            elif pd.api.types.is_datetime64_any_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
                viz_options = ["Line Plot", "Scatter Plot (over time)"]
            elif pd.api.types.is_datetime64_any_dtype(x_type) and pd.api.types.is_categorical_dtype(y_type):
                viz_options = ["Bar Plot (over time)"]
            else:
                viz_options = ["Scatter Plot"]  # Default fallback

            viz_type = st.selectbox("Select Visualization Type", options=[""] + viz_options, help="Choose a visualization based on the selected features.")

            if viz_type:
                try:
                    if viz_type == "Scatter Plot":
                        fig = px.scatter(df, x=x_feature, y=y_feature, color=target if target else None, trendline="ols", title=f"{x_feature} vs {y_feature}")
                    elif viz_type == "Pair Plot" and df.select_dtypes(include="number").shape[1] >= 2:
                        numeric_cols = df.select_dtypes(include="number").columns[:4]  # Limit for performance
                        fig = px.scatter_matrix(df, dimensions=numeric_cols, color=target if target else None, title="Pair Plot")
                    elif viz_type == "Correlation Heatmap" and df.select_dtypes(include="number").shape[1] >= 2:
                        corr = df.select_dtypes(include="number").corr()
                        fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap", aspect="auto", color_continuous_scale="RdBu")
                    elif viz_type == "Box Plot":
                        fig = px.box(df, x=x_feature, y=y_feature, color=target if target else None, title=f"{y_feature} by {x_feature}")
                    elif viz_type == "Violin Plot":
                        fig = px.violin(df, x=x_feature, y=y_feature, color=target if target else None, box=True, points="all", title=f"{y_feature} by {x_feature}")
                    elif viz_type == "Histogram (grouped by X)":
                        fig = px.histogram(df, x=y_feature, color=x_feature, title=f"{y_feature} Distribution by {x_feature}")
                    elif viz_type == "Bar Plot":
                        fig = px.bar(df, x=x_feature, y=y_feature, color=target if target else None, title=f"{y_feature} by {x_feature}")
                    elif viz_type == "Line Plot":
                        fig = px.line(df, x=x_feature, y=y_feature, color=target if target else None, title=f"{y_feature} over {x_feature}")
                    elif viz_type == "Bar Plot (over time)":
                        fig = px.bar(df, x=x_feature, y=y_feature, color=target if target else None, title=f"{y_feature} over {x_feature}")
                    else:
                        st.markdown('<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> Visualization not supported for selected features.</p>', unsafe_allow_html=True)
                        fig = None
                except Exception as e:
                    st.markdown(f'<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> Error rendering visualization: {str(e)}</p>', unsafe_allow_html=True)
                    fig = None

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        if target and df[target].nunique() <= 20:
            st.markdown('<h2 class="section-header"><i class="fas fa-chart-pie fa-icon"></i> Target Distribution</h2>', unsafe_allow_html=True)
            fig = px.histogram(df, x=target, color=target, title="Target Distribution")
            st.plotly_chart(fig, use_container_width=True)
            if df[target].dtype not in ["int64", "float64"]:
                class_counts = df[target].value_counts()
                imbalance_ratio = class_counts.max() / class_counts.min()
                if imbalance_ratio > 2:
                    st.markdown('<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> Class imbalance detected (ratio: {:.2f}).</p>'.format(imbalance_ratio), unsafe_allow_html=True)
                    st.markdown("""
                        <h3 class="subsection-header"><i class="fas fa-balance-scale fa-icon"></i> Suggestions for Handling Class Imbalance</h3>
                        <ul><li><b>SMOTE</b>: Generates synthetic samples for the minority class.</li></ul>
                    """, unsafe_allow_html=True)
                    if st.button("Apply SMOTE"):
                        X = df.drop(columns=[target]).select_dtypes(include=["number"]).fillna(0)
                        y = df[target]
                        if len(X) > 1:
                            le = LabelEncoder()
                            y_encoded = le.fit_transform(y)
                            n_neighbors = min(5, len(X) - 1)
                            smote = SMOTE(random_state=42, n_neighbors=n_neighbors)
                            try:
                                X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
                                df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                                df_resampled[target] = le.inverse_transform(y_resampled)
                                st.session_state["df"] = df_resampled
                                st.session_state["action_history"].append("Applied SMOTE to balance classes")
                                st.session_state["refresh"] += 1  # Trigger re-render
                                st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Applied SMOTE to balance classes.</p>', unsafe_allow_html=True)
                            except ValueError as e:
                                st.markdown(f'<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> Error applying SMOTE: {str(e)}</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> Insufficient data for SMOTE (need at least 2 samples).</p>', unsafe_allow_html=True)

# Feature Insights Section
with st.container():
    with st.expander("Feature Insights"):
        st.markdown('<h2 class="section-header"><i class="fas fa-search-plus fa-icon"></i> Feature Insights</h2>', unsafe_allow_html=True)
        df = st.session_state["df"].copy()
        numeric_df = df.select_dtypes(include="number")
        
        if not numeric_df.empty:
            st.markdown('<h2 class="section-header"><i class="fas fa-link fa-icon"></i> Top Feature Correlations</h2>', unsafe_allow_html=True)
            corr_matrix = numeric_df.corr(method="pearson").abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            top_corr = upper_tri.stack().sort_values(ascending=False).reset_index()
            top_corr.columns = ["Feature 1", "Feature 2", "Correlation"]
            st.dataframe(top_corr.head(10), use_container_width=True)

        if target:
            st.markdown(f'<h2 class="section-header"><i class="fas fa-info fa-icon"></i> Mutual Information with `{target}`</h2>', unsafe_allow_html=True)
            X = df.drop(columns=[target]).select_dtypes(include=["number"]).fillna(0)
            y = df[target]
            if len(X.columns) > 0:
                if y.dtype in ["int64", "float64"]:
                    mi_scores = mutual_info_regression(X, y, random_state=42)
                else:
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
                mi_df = pd.DataFrame({"Feature": X.columns, "Mutual Information": mi_scores}).sort_values(by="Mutual Information", ascending=False)
                st.dataframe(mi_df.head(10), use_container_width=True)
                st.markdown("**Note**: Mutual Information captures both linear and non-linear relationships.")
                fig = px.bar(mi_df.head(10), x="Feature", y="Mutual Information", title=f"Mutual Information with {target}")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'<h2 class="section-header"><i class="fas fa-star fa-icon"></i> Feature Importance for `{target}` (Random Forest)</h2>', unsafe_allow_html=True)
            if y.dtype in ["int64", "float64"]:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            if len(X.columns) > 0 and (y.dtype in ["int64", "float64"] or len(y.unique()) > 1):
                model.fit(X, y)
                importance = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                st.dataframe(importance.head(10), use_container_width=True)

                st.markdown('<h2 class="section-header"><i class="fas fa-chart-bar fa-icon"></i> SHAP Feature Importance</h2>', unsafe_allow_html=True)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X.sample(min(100, len(X)), random_state=42))
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X.sample(min(100, len(X)), random_state=42), plot_type="bar", show=False)
                st.pyplot(fig)
                plt.clf()

        st.markdown('<h2 class="section-header"><i class="fas fa-tachometer-alt fa-icon"></i> Low Variance Features</h2>', unsafe_allow_html=True)
        variances = numeric_df.var()
        low_variance = variances[variances < 0.01]
        if not low_variance.empty:
            st.markdown('<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> Low variance features detected (consider removing):</p>', unsafe_allow_html=True)
            st.dataframe(low_variance.to_frame("Variance"), use_container_width=True)
        else:
            st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> No low variance features detected.</p>', unsafe_allow_html=True)

        st.markdown('<h2 class="section-header"><i class="fas fa-tags fa-icon"></i> High Cardinality Categorical Features</h2>', unsafe_allow_html=True)
        cat_cols = df.select_dtypes(include=["object"]).columns
        high_cardinality = df[cat_cols].nunique()[df[cat_cols].nunique() > 20]
        if not high_cardinality.empty:
            st.markdown('<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> High cardinality categorical features detected (consider encoding or reducing):</p>', unsafe_allow_html=True)
            st.dataframe(high_cardinality.to_frame("Unique Values"), use_container_width=True)

# Machine Learning Preparation Section
with st.container():
    with st.expander("Machine Learning Preparation"):
        st.markdown('<h2 class="section-header"><i class="fas fa-robot fa-icon"></i> ML Preparation</h2>', unsafe_allow_html=True)
        df = st.session_state["df"].copy()
        if target:
            features = [col for col in df.columns if col != target]
            X = df[features].select_dtypes(include=["number"]).fillna(0)
            y = df[target]
            
            st.markdown('<h3 class="subsection-header"><i class="fas fa-cogs fa-icon"></i> Task Type and Data Splitting</h3>', unsafe_allow_html=True)
            inferred_task = "Regression" if y.dtype in ["int64", "float64"] else "Classification"
            task_type = st.selectbox("Select Task Type", ["Auto (Inferred: " + inferred_task + ")", "Classification", "Regression"], help="Auto infers based on target type.")
            task_type = inferred_task if task_type.startswith("Auto") else task_type
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.write(f"Training set: {X_train.shape[0]} rows (80%), Test set: {X_test.shape[0]} rows (20%)")
            st.session_state["action_history"].append("Split data into 80% training and 20% testing")
            
            if st.checkbox("Encode Categorical Features", help="Convert categorical features to numeric using LabelEncoder."):
                cat_cols = df[features].select_dtypes(include=["object"]).columns
                if len(cat_cols) > 0:
                    for col in cat_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                    st.session_state["df"] = df
                    st.session_state["action_history"].append("Encoded categorical features using LabelEncoder")
                    st.session_state["refresh"] += 1  # Trigger re-render
                    st.markdown('<p class="success"><i class="fas fa-check-circle fa-icon"></i> Encoded categorical features using LabelEncoder.</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="warning"><i class="fas fa-info-circle fa-icon"></i> No categorical features to encode.</p>', unsafe_allow_html=True)

            if st.checkbox("Scale Numeric Features", help="Apply scaling to numeric features for better model performance."):
                scale_method = st.selectbox("Select Scaling Method", ["StandardScaler", "MinMaxScaler"], help="StandardScaler standardizes features, MinMaxScaler scales to [0,1].")
                num_cols = df[features].select_dtypes(include=["number"]).columns
                if len(num_cols) > 0:
                    if scale_method == "StandardScaler":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    df[num_cols] = scaler.fit_transform(df[num_cols])
                    st.session_state["df"] = df
                    st.session_state["action_history"].append(f"Applied {scale_method} to numeric features")
                    st.session_state["refresh"] += 1  # Trigger re-render
                    st.markdown(f'<p class="success"><i class="fas fa-check-circle fa-icon"></i> Applied {scale_method} to numeric features.</p>', unsafe_allow_html=True)

            if task_type == "Classification" and y.nunique() <= 20:
                st.markdown('<h3 class="subsection-header"><i class="fas fa-balance-scale fa-icon"></i> Class Balance</h3>', unsafe_allow_html=True)
                fig = px.histogram(df, x=target, color=target, title="Class Distribution")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<h3 class="subsection-header"><i class="fas fa-project-diagram fa-icon"></i> Feature-Target Relationships</h3>', unsafe_allow_html=True)
            selected_features = st.multiselect("Select numeric features to plot against target", X.columns.tolist(), help="Visualize relationships between features and target.")
            for feature in selected_features:
                if task_type == "Classification":
                    fig = px.box(df, x=target, y=feature, color=target, title=f"{feature} vs {target}")
                else:
                    fig = px.scatter(df, x=feature, y=target, color=target, trendline="ols", title=f"{feature} vs {target}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<p class="warning"><i class="fas fa-exclamation-triangle fa-icon"></i> Please select a target column to proceed with ML preparation.</p>', unsafe_allow_html=True)

# Download Processed Data
with st.container():
    with st.expander("Download Processed Data"):
        st.markdown('<h2 class="section-header"><i class="fas fa-download fa-icon"></i> Download Processed Dataset</h2>', unsafe_allow_html=True)
        df = st.session_state["df"].copy()
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="processed_dataset.csv",
            mime="text/csv",
            help="Download the processed dataset as a CSV file."
        )

# Display Data if Button Clicked
if st.session_state.get("show_data", False):
    st.markdown('<h2 class="section-header"><i class="fas fa-table fa-icon"></i> Current Dataset</h2>', unsafe_allow_html=True)
    st.dataframe(st.session_state["df"], use_container_width=True)
