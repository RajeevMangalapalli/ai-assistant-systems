import pandas as pd
import joblib
import streamlit as st

@st.cache_data
def load_dataset():
    return pd.read_csv(
        r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\Classification_model_output.csv"
    )

@st.cache_resource
def load_model():
    return joblib.load(
        r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Model_implementations\rf_classification_model.joblib"
    )
