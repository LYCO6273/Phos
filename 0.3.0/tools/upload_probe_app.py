"""Minimal Streamlit app used by tools/upload_probe.py."""

import streamlit as st

st.file_uploader("probe", type=["jpg", "jpeg", "png", "dng"])
