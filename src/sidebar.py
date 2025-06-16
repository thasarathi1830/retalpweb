# sidebar.py
import streamlit as st
from pathlib import Path

def render_sidebar():
    st.sidebar.title("📁 Upload File")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (.csv, .xlsx, .xls, .ods)",
        type=["csv", "xlsx", "xls", "ods"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    st.sidebar.markdown("Use the navigation below to explore the data.")

    return uploaded_file