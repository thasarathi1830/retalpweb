import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from pathlib import Path
import streamlit.components.v1 as components
# from src.sidebar import render_sidebar

# --- Page Configuration ---
st.set_page_config(page_title="Retalp EDA Dashboard", layout="wide")

# --- Create necessary folders ---
Path("data").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

# --- Load data function ---
def load_data(file_path):
    if str(file_path).lower().endswith('.csv'):
        encodings = ['utf-8', 'utf-16', 'latin1', 'cp1252']
        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except Exception:
                continue
        raise ValueError("Could not read CSV file with common encodings.")
    elif str(file_path).lower().endswith('.xlsx'):
        return pd.read_excel(file_path, engine='openpyxl')
    elif str(file_path).lower().endswith('.xls'):
        return pd.read_excel(file_path, engine='xlrd')
    elif str(file_path).lower().endswith('.ods'):
        return pd.read_excel(file_path, engine='odf')
    else:
        raise ValueError("Unsupported file format.")

# --- Sidebar: Upload file ---
uploaded_file = st.sidebar.file_uploader("Upload your data file", type=["csv", "xlsx", "xls", "ods"])

# --- Session state setup ---
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    file_path = Path("data") / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        df = load_data(file_path)
        st.session_state.df = df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    df = st.session_state.df

# --- Sidebar Navigation ---
st.sidebar.markdown("## Choose Section")
section = st.sidebar.radio(
    "Go to",
    ("Data Overview", "Data Cleaning", "Outlier Detection", "Visualizations", "Full Report"),
    index=0
)

# --- Main content ---
if df is not None:
    st.title("📊 Retalp EDA Dashboard")

    if section == "Data Overview":
        st.header("🔍 Data Overview")
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        st.dataframe(df.head(10))
        st.write("**Summary Statistics:**")
        st.dataframe(df.describe(include='all').T)
        st.write("**Missing Values:**")
        st.dataframe(df.isnull().sum().to_frame("Missing Count"))

    elif section == "Data Cleaning":
        st.header("🧹 Data Cleaning")
        cols = st.multiselect("Select columns to drop", df.columns)
        if cols:
            df.drop(columns=cols, inplace=True)
            st.success(f"Dropped columns: {', '.join(cols)}")

        fill_option = st.selectbox("Missing value handling", ["None", "Fill with mean", "Fill with median", "Fill with mode", "Drop rows with NA"])
        if fill_option != "None":
            if fill_option == "Drop rows with NA":
                df.dropna(inplace=True)
            else:
                for col in df.select_dtypes(include=np.number).columns:
                    if fill_option == "Fill with mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif fill_option == "Fill with median":
                        df[col].fillna(df[col].median(), inplace=True)
                    elif fill_option == "Fill with mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)
            st.success(f"Missing values handled: {fill_option}")

        st.dataframe(df.head(10))
        st.session_state.df = df

    elif section == "Outlier Detection":
        st.header("🔎 Outlier Detection & Handling")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            col = st.selectbox("Numeric column for outlier detection", num_cols)
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            st.write(f"**Number of outliers:** {len(outliers)}")
            st.dataframe(outliers)

            action = st.radio("Handle outliers", ["Do nothing", "Cap values", "Remove outliers"])
            if action == "Cap values":
                df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))
                st.success("Outliers capped.")
            elif action == "Remove outliers":
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                st.success("Outliers removed.")
            st.session_state.df = df
        else:
            st.info("No numeric columns for outlier detection.")

    elif section == "Visualizations":
        st.header("📈 Visualizations")
        plot_type = st.selectbox("Choose plot type", ["Bar Plot", "Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap"])

        if plot_type == "Bar Plot":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols):
                col = st.selectbox("Column", cat_cols)
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind='bar', ax=ax, color='#1f77b4')
                st.pyplot(fig)
            else:
                st.info("No categorical columns available.")

        elif plot_type == "Histogram":
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols):
                col = st.selectbox("Column", num_cols)
                bins = st.slider("Bins", 5, 100, 20)
                fig, ax = plt.subplots()
                df[col].plot(kind='hist', bins=bins, ax=ax, color='#ff7f0e', edgecolor='black')
                st.pyplot(fig)
            else:
                st.info("No numeric columns available.")

        elif plot_type == "Box Plot":
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols):
                col = st.selectbox("Column", num_cols)
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax, color='#2ca02c')
                st.pyplot(fig)
            else:
                st.info("No numeric columns available.")

        elif plot_type == "Scatter Plot":
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) >= 2:
                x = st.selectbox("X axis", num_cols)
                y = st.selectbox("Y axis", num_cols, index=1)
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[x], y=df[y], ax=ax, color='#d62728')
                st.pyplot(fig)
            else:
                st.info("Need at least two numeric columns.")

        elif plot_type == "Correlation Heatmap":
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else:
                st.info("Need at least two numeric columns.")

    elif section == "Full Report":
        st.header("🤖 Automated EDA Report")
        if st.button("Generate and Show Report"):
            with st.spinner("Generating report..."):
                profile = ProfileReport(df, title="Full EDA Report", explorative=True)
                components.html(profile.to_html(), height=1000, scrolling=True)

else:
    st.info("👈 Upload a file from the sidebar to start.")


