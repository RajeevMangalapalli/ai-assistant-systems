import streamlit as st
import pandas as pd
import joblib
from faker import Faker
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from sklearn.metrics import accuracy_score, classification_report


st.set_page_config(page_title="Music Data Transformations - Augmented Dataset", layout="wide")
st.title("🎵 Data Transformations for Machine Learning Models (Augmented Dataset)")
st.header("Overview of Data Preparation Steps")

nodes = [
    StreamlitFlowNode(id='1', pos=(50, 100), data={'content': 'Raw Data'}, node_type='input', source_position='right'),
    StreamlitFlowNode(id='2', pos=(250, 100), data={'content': 'Cleaned Data'}, node_type='default'),
    StreamlitFlowNode(id='3', pos=(450, 150), data={'content': 'Transformed Data (20% Synthetic Data)'}, node_type='default'),
    StreamlitFlowNode(id='4', pos=(700, 100), data={'content': 'Linear Regression Model'}, node_type='default'),
    StreamlitFlowNode(id='5', pos=(700, 250), data={'content': 'Classification Model'}, node_type='output')
]

edges = [
    StreamlitFlowEdge('1-2', '1', '2', animated=True),
    StreamlitFlowEdge('2-3', '2', '3', animated=True),
    StreamlitFlowEdge('3-4', '3', '4', animated=True),
    StreamlitFlowEdge('3-5', '3', '5', animated=True)
]

if 'flow_state' not in st.session_state:
    st.session_state.flow_state = StreamlitFlowState(nodes, edges)

streamlit_flow(
    'static_flow',
    st.session_state.flow_state,
    fit_view=True,
    show_minimap=False,
    show_controls=False
)

st.divider()


# 1. RAW DATASET

st.header("1. Raw Dataset")

df_raw = pd.read_csv(
    r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\music_info.csv"
)

st.dataframe(df_raw.head())
st.dataframe(df_raw.describe())

st.divider()


# 2. DATA CLEANING

st.header("2. Data Cleaning")

columns_to_drop = ["spotify_id", "tags", "genre"]
df_cleaned = df_raw.drop(columns=columns_to_drop)

st.write("Removed columns:", columns_to_drop)
st.divider()


# 3. TRANSFORMED + AUGMENTED DATA

st.header("3. Transformed Dataset with Augmentation (20%)")

df_transformed = pd.read_csv(
    r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\transformed_music_info.csv"
)

fake = Faker()
num_synth = int(0.2 * len(df_transformed))
synthetic_rows = []

for _ in range(num_synth):
    base = df_transformed.sample(1).iloc[0]
    synthetic_rows.append({
        "danceability": fake.random.uniform(0, 1),
        "energy": fake.random.uniform(0, 1),
        "key": fake.random_int(0, 11),
        "mode": fake.random_int(0, 1),
        "speechiness": fake.random.uniform(0, 1),
        "acousticness": fake.random.uniform(0, 1),
        "instrumentalness": fake.random.uniform(0, 1),
        "liveness": fake.random.uniform(0, 1),
        "valence": fake.random.uniform(0, 1),
        "tempo": fake.random.uniform(60, 200),
        "time_signature": fake.random_int(3, 7),
        "year": fake.random_int(1950, 2025),
        "duration_min": fake.random.uniform(2, 6),
        "loudness_normalized": fake.random.uniform(0, 1),
        "mood_label": base["mood_label"]
    })

df_augmented = pd.concat(
    [df_transformed.iloc[:-num_synth], pd.DataFrame(synthetic_rows)],
    ignore_index=True
)

st.success(f"Augmented dataset size: {df_augmented.shape[0]}")
st.dataframe(df_augmented.head())

st.divider()


# 4. LINEAR REGRESSION MODEL

st.header("4. Linear Regression Model")

df_lr = df_augmented.copy()

df_lr['mood_score'] = (
    0.5 * df_lr['valence'] +
    0.3 * df_lr['energy'] +
    0.2 * df_lr['danceability']
)

lr_model = joblib.load(
    r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Model_implementations\linear_regression_model.joblib"
)

X_lr = df_lr[['duration_min', 'loudness_normalized', 'year',
              'speechiness', 'instrumentalness', 'liveness', 'tempo']]

df_lr['predicted_mood_score'] = lr_model.predict(X_lr)

def categorize(score):
    if score >= 0.6:
        return "Happy"
    elif score <= 0.4:
        return "Sad"
    else:
        return "Neutral"

df_lr['mood_label_from_lr'] = df_lr['predicted_mood_score'].apply(categorize)

st.dataframe(df_lr[['mood_score', 'predicted_mood_score', 'mood_label_from_lr']].head())

st.divider()


# 5. CLASSIFICATION MODEL

st.header("5. Classification Model (Random Forest)")

df_clf = df_augmented.copy()

df_clf['mood_score'] = (
    0.5 * df_clf['valence'] +
    0.3 * df_clf['energy'] +
    0.2 * df_clf['danceability']
)

df_clf['mood_label'] = df_clf['mood_score'].apply(categorize)

rf_model = joblib.load(
    r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Model_implementations\rf_classification_model.joblib"
)

X_clf = df_clf[
    ['danceability', 'energy', 'key', 'loudness', 'mode',
     'speechiness', 'acousticness', 'instrumentalness',
     'liveness', 'valence', 'tempo', 'time_signature',
     'duration_min', 'loudness_normalized']
]


df_clf['mood_label_from_clf'] = rf_model.predict(X_clf)

st.dataframe(df_clf[['mood_score', 'mood_label', 'mood_label_from_clf']].head(10))

st.divider()

# 6. MODEL COMPARISON

st.header("6. Model Comparison: Linear Regression vs Classification")

comparison_data = pd.DataFrame({
    'actual_mood_label': df_clf['mood_label'],
    'lr_predicted_label': df_lr['mood_label_from_lr'],
    'clf_predicted_label': df_clf['mood_label_from_clf']
})

st.dataframe(comparison_data.head(15))

lr_accuracy = accuracy_score(
    comparison_data['actual_mood_label'],
    comparison_data['lr_predicted_label']
)

clf_accuracy = accuracy_score(
    comparison_data['actual_mood_label'],
    comparison_data['clf_predicted_label']
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Linear Regression Accuracy", f"{lr_accuracy:.2%}")
    st.dataframe(pd.DataFrame(
        classification_report(
            comparison_data['actual_mood_label'],
            comparison_data['lr_predicted_label'],
            output_dict=True
        )
    ).transpose())

with col2:
    st.metric("Random Forest Accuracy", f"{clf_accuracy:.2%}")
    st.dataframe(pd.DataFrame(
        classification_report(
            comparison_data['actual_mood_label'],
            comparison_data['clf_predicted_label'],
            output_dict=True
        )
    ).transpose())

agreement_rate = (
    comparison_data['lr_predicted_label']
    == comparison_data['clf_predicted_label']
).mean()

st.metric("Model Agreement Rate", f"{agreement_rate:.2%}")

st.success("✅ Augmented data evaluation completed successfully.")
