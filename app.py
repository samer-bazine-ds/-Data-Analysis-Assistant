import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import numpy as np
from dotenv import load_dotenv
import os
from scipy.stats import zscore



st.set_page_config(page_title="AI Data Assistant", layout="wide")
st.title("📊 Data Analysis Assistant")




uploaded_file = st.file_uploader("Upload your CSV file", type="csv")


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(" Data loaded successfully.")
    st.dataframe(df.head())

    if st.checkbox("Show basic analysis"):
        st.subheader(" Statistical Summary")
        st.write(df.describe())

        st.subheader(" Missing Values")
        st.write(df.isnull().sum())

        st.subheader(" Numeric Columns")
        st.write(df.select_dtypes(include='number').columns.tolist())

        st.subheader(" Categorical Columns")
        st.write(df.select_dtypes(include='object').columns.tolist())

    if st.checkbox("Show basic visuals"):
        numeric_cols = df.select_dtypes(include='number').columns
        categorical_cols = df.select_dtypes(include='object').columns

        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        for col in categorical_cols:
            fig, ax = plt.subplots()
            sns.countplot(y=col, data=df, ax=ax)
            ax.set_title(f"Count Plot of {col}")
            st.pyplot(fig)

        if len(numeric_cols) > 1:
            st.subheader(" Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

    st.markdown("---")
    st.header(" Missing Data Handling")

    if df.isnull().values.any():
        st.write("Your dataset contains missing values.")
        strategy = st.selectbox("Select a strategy to handle missing values:", 
                                ["Do nothing", "Drop rows", "Fill with mean", "Fill with median", "Fill with zero"])

        if strategy == "Drop rows":
            df = df.dropna()
            st.success("Dropped rows with missing values.")
        elif strategy == "Fill with mean":
            df = df.fillna(df.mean(numeric_only=True))
            st.success("Filled missing values with column mean.")
        elif strategy == "Fill with median":
            df = df.fillna(df.median(numeric_only=True))
            st.success("Filled missing values with column median.")
        elif strategy == "Fill with zero":
            df = df.fillna(0)
            st.success("Filled missing values with zero.")
        else:
            st.info("No action taken.")
    else:
        st.success("No missing values in your dataset.")

    st.markdown("---")
    st.header(" Top Feature Correlations")

    numeric_df = df.select_dtypes(include='number')

    if not numeric_df.empty:
        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_corr = (upper_tri.stack().sort_values(ascending=False).reset_index())
        top_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
        st.dataframe(top_corr.head(10), use_container_width=True)
    else:
        st.warning("No numeric features found for correlation analysis.")

    st.markdown("---")
    st.header(" Compare Features with Target Variable")

    target = st.selectbox("Select a target variable", df.columns)

    if df[target].dtype in ['int64', 'float64']:
        correlations = df.corr()[target].drop(target).sort_values(ascending=False)
        st.subheader(f"Top features correlated with `{target}`:")
        st.dataframe(correlations.head(10).to_frame("Correlation"), use_container_width=True)
    else:
        st.warning("Selected target is not numeric — correlation not applicable.")

    st.markdown("---")
    st.header(" Outlier Detection")

    z_scores = numeric_df.apply(zscore)
    outlier_flags = (z_scores.abs() > 3)
    outlier_counts = outlier_flags.sum()

    st.subheader("Outliers per Numeric Feature (Z > 3):")
    st.dataframe(outlier_counts[outlier_counts > 0].to_frame("Outlier Count"))

    if st.checkbox("🔍 Show rows with any outlier"):
        outlier_rows = df[outlier_flags.any(axis=1)]
        st.write(f"Found {outlier_rows.shape[0]} rows with at least one outlier.")
        st.dataframe(outlier_rows.head(10))

    st.markdown("---")
    st.header(" Data Insights Dashboard")

    skewness = df[numeric_df.columns].skew().sort_values(ascending=False)
    high_skew = skewness[skewness > 1]

    top_corr_feats = []
    if df[target].dtype in ['int64', 'float64']:
        corr_vals = df.corr()[target].drop(target).sort_values(ascending=False)
        top_corr_feats = corr_vals[abs(corr_vals) > 0.5]

    total_outliers = outlier_flags.any(axis=1).sum()
    missing_total = df.isnull().sum().sum()
    missing_percent = round((missing_total / df.size) * 100, 2)

    st.subheader(" Key Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric(" Skewed Features", len(high_skew))
    col2.metric(" Total Outliers", total_outliers)
    col3.metric(" Missing Data (%)", f"{missing_percent}%")

    if len(top_corr_feats) > 0:
        st.subheader(" Strongly Correlated Features with Target")
        st.dataframe(top_corr_feats.to_frame("Correlation"), use_container_width=True)

    if len(high_skew) > 0:
        st.subheader(" Skewed Features (Consider Transformation)")
        st.dataframe(high_skew.to_frame("Skewness"), use_container_width=True)

    if df[target].dtype in ['int64', 'float64']:
        st.info(" Consider regression modeling. Top features are already highlighted.")
    elif df[target].dtype == "object":
     st.info(" Classification may be appropriate. Check target class distribution.")
