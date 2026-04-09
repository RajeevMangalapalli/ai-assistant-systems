import streamlit as st
from PIL import Image

st.title("Music Recommendation Assistant")

page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Real Dataset", "Fake Dataset", "Real Models", "Fake Models", "Chatbot"]
)

if page == "Home":
    st.write("Welcome to Home page")
elif page == "Real Dataset":
    import pages.Real_Dataset as rd
    rd.app()
elif page == "Fake Dataset":
    import pages.Fake_Dataset as fd
    fd.app()
elif page == "Real Models":
    import pages.Real_Models as rm
    rm.app()
elif page == "Fake Models":
    import pages.Fake_Models as fm
    fm.app()
elif page == "Chatbot":
    import pages.Chatbot_Interface as ci
    ci.app()



st.markdown("""
             ## 🎵 Music Recommendation Assistant

This project implements a **music recommendation assistant** that analyzes user mood and listening behavior using **machine learning models trained on real and synthetic data**. The goal is to evaluate how **fake data generation and data augmentation** affect model performance.

### 🔹 Model Implementations
- **Real Models – Classification**  
  Predict discrete mood categories using real-world datasets.
- **Real Models – Regression**  
  Predict continuous mood scores (e.g., valence, energy, danceability) from real data.
- **Fake Models – Classification**  
  Train classifiers on **synthetic datasets generated with Faker**.
- **Fake Models – Regression**  
  Train regression models using faker-generated numerical mood values.

### 🔹 Data Augmentation
- **80% real data + 20% synthetic (fake) data**
- Used to analyze robustness, generalization, and performance changes
- Helps simulate realistic scenarios with limited real data

### 🔹 Objectives
- Compare **real vs. fake vs. augmented datasets**
- Evaluate classification and regression performance
- Assess the usefulness of synthetic data in ML pipelines

### 🔹 Technologies
Python · Streamlit · Scikit-learn · Faker · Pandas · NumPy

            """)

st.divider()

image = Image.open(
    r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\App\assets\homepage_image.png"
)
st.image(image, width='stretch')
st.caption("Image Source: Gemini Nanobanana")