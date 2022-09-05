import pandas as pd
from pathlib import Path
import streamlit as st

from config import config
from tagifai import main, utils

# Title
st.title("MLOps Course | Made With ML Reproduction")

# Sections
st.header("🔢 Data")
st.header("📊 Performance")
st.header("🔭 Inference")

# Data
st.header("Data")
projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
df = pd.read_csv(projects_fp)
st.text(f"Projects (count: {len(df)})")
st.write(df)

# Performance
st.header("📊 Performance")
performance_fp = Path(config.CONFIG_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.text("Overall:")
st.write(performance["overall"])
tag = st.selectbox("Choose a tag: ", list(performance["class"].keys()))
st.write(performance["class"][tag])
tag = st.selectbox("Choose a slice: ", list(performance["slices"].keys()))
st.write(performance["slices"][tag])

# Inference
st.header("🔭 Inference")
text = st.text_input("Enter text:", "Transfer learning with transformers for text classification.")
run_id = st.text_input("Enter run ID:", open(Path(config.CONFIG_DIR, "run_id.txt")).read())
prediction = main.predict_tag(text=text, run_id=run_id)
st.write(prediction)

@st.cache()
def load_data():
    projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
    df = pd.read_csv(projects_fp)
    return df
