import streamlit as st
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from sklearn.metrics import accuracy_score, classification_report

def app():
    st.title("📊 Real Dataset")
    st.write("This page displays information about the real dataset.")


st.set_page_config(page_title="Music Data Transformations - Real Dataset", layout="wide")

st.title("🎵 Data Transformations for Machine Learning Models (Real Dataset)")
st.header("Overview of Data Preparation Steps")
nodes = [
    StreamlitFlowNode(
        id='1',
        pos=(50, 100),
        data={'content': 'Raw Data'},
        node_type='input',
        source_position='right',
        draggable=True
    ),

    StreamlitFlowNode(
        id='2',
        pos=(250, 100),
        data={'content': 'Cleaned Data'},
        node_type='default',
        source_position='right',
        target_position='left',
        draggable=True
    ),

    StreamlitFlowNode(
        id='3',
        pos=(450, 150),
        data={'content': 'Transformed Data'},
        node_type='default',
        source_position='right',
        target_position='left',
        draggable=True
    ),

    StreamlitFlowNode(
        id='4',
        pos=(700, 100),
        data={'content': 'Linear Regression Model'},
        node_type='default',
        target_position='left',
        draggable=True
    ),

    StreamlitFlowNode(
        id='5',
        pos=(700, 250),
        data={'content': 'Classification Model'},
        node_type='output',
        target_position='left',
        draggable=True
    )
]

edges = [
    StreamlitFlowEdge(
        '1-2', '1', '2',
        animated=True,
        marker_end={'type': 'arrow'}
    ),
    StreamlitFlowEdge(
        '2-3', '2', '3',
        animated=True,
        marker_end={'type': 'arrow'}
    ),
    StreamlitFlowEdge(
        '3-4', '3', '4',
        animated=True,
        marker_end={'type': 'arrow'}
    ),
    StreamlitFlowEdge(
        '3-5', '3', '5',
        animated=True,
        marker_end={'type': 'arrow'}
    )
]


st.write("The data from one node is used in the next Node")


if 'static_flow_state' not in st.session_state:
	st.session_state.static_flow_state = StreamlitFlowState(nodes, edges)

streamlit_flow('static_flow',
	st.session_state.static_flow_state,
	fit_view=True,
	show_minimap=False,
	show_controls=False,
	pan_on_drag=False,
	allow_zoom=False)


# 1. Load raw data
st.header("1. Raw Dataset")

df = pd.read_csv(
    r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\music_info.csv"
)

st.write("Preview of the original dataset:")
st.dataframe(df.head())

st.write("Dataset information:")
st.dataframe(df.describe())

st.divider()

# 2. Data Cleaning
st.header("2. Data Cleaning")

st.markdown("""
The following columns were removed because they are **non-numeric**, **IDs**, or **not useful**
for regression or classification models.
""")

columns_to_drop = ["spotify_id", "tags", "genre"]
df_cleaned = df.drop(columns=columns_to_drop)

st.write("Removed columns:", columns_to_drop)

st.divider()

# 3. Final Dataset
st.header("3. Transformed Dataset - to be used in Linear Regression Model")
st.dataframe(df_cleaned.head())

st.divider()

# ============================================================================
# LINEAR REGRESSION MODEL SECTION
# ============================================================================
st.header("4. Linear Regression Model")

# Display mood_score definition
st.subheader("Target Variable Definition")
st.markdown("""
**Mood Score** is defined as a weighted combination of three musical features:
""")
st.code("""
mood_score = (0.5 × valence) + (0.3 × energy) + (0.2 × danceability)
""")

st.markdown("""
- **Valence (50%)**: Measures the musical positiveness (happy vs. sad)
- **Energy (30%)**: Represents the intensity and activity level
- **Danceability (20%)**: Describes how suitable a track is for dancing
""")

# Display feature and target variables
st.subheader("Model Variables")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Feature Variables (X):**")
    feature_list = [
        'duration_min',
        'loudness_normalized',
        'year',
        'speechiness',
        'instrumentalness',
        'liveness',
        'tempo'
    ]
    for feature in feature_list:
        st.write(f"- {feature}")

with col2:
    st.markdown("**Target Variable (y):**")
    st.write("- mood_score")

st.divider()

# Load the transformed data for prediction
df_lr = pd.read_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\transformed_music_info.csv")

# Define mood score as a combination of valence, energy, and danceability
df_lr['mood_score'] = (0.5 * df_lr['valence']) + (0.3 * df_lr['energy']) + (0.2 * df_lr['danceability'])

# Load the pre-trained Linear Regression model
lr_model = joblib.load(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Model_implementations\linear_regression_model.joblib")

# Features and target variable
X_lr = df_lr[['duration_min', 'loudness_normalized', 'year', 'speechiness', 'instrumentalness', 'liveness', 'tempo']]
y_lr = df_lr['mood_score']

# Make predictions using the pre-trained model
y_pred_lr = lr_model.predict(X_lr)

# New column for predicted mood score
df_lr['predicted_mood_score'] = y_pred_lr

# Categorize into happy, sad, neutral based on the predicted mood score
def categorize_mood(score):
    if score >= 0.6:
        return 'Happy'
    elif score <= 0.4:
        return 'Sad'
    else:
        return 'Neutral'

df_lr['mood_label_from_lr'] = df_lr['predicted_mood_score'].apply(categorize_mood)

comparison_df = df_lr[['mood_score', 'predicted_mood_score', 'mood_label_from_lr']].head()

st.subheader("Model Predictions")
st.write("Comparison of Actual vs Predicted Mood Scores:")
st.dataframe(comparison_df)

# New data set obtained after prediction
st.header("Dataset with Predicted Mood Scores")
st.dataframe(df_lr.head())

st.divider()

# ============================================================================
# CLASSIFICATION MODEL SECTION
# ============================================================================
st.header("5. Classification Model (Random Forest)")

# Display mood categorization logic
st.subheader("Mood Categorization Logic")
st.markdown("""
Songs are classified into three mood categories based on their features:
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.success("**😊 Happy**\n\nMood Score ≥ 0.6")
with col2:
    st.warning("**😐 Neutral**\n\nMood Score: 0.4 - 0.6")
with col3:
    st.error("**😢 Sad**\n\nMood Score ≤ 0.4")

# Display model variables
st.subheader("Model Variables")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Feature Variables (X):**")
    classification_features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'time_signature', 'duration_min', 'loudness_normalized'
    ]
    for feature in classification_features:
        st.write(f"- {feature}")

with col2:
    st.markdown("**Target Variable (y):**")
    st.write("- mood_label")
    st.caption("(Categorical: Happy, Neutral, or Sad)")

st.divider()

# Load the transformed data for classification
df_clf = pd.read_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\transformed_music_info.csv")

# Calculate mood score for ground truth labels
df_clf['mood_score'] = (0.5 * df_clf['valence']) + (0.3 * df_clf['energy']) + (0.2 * df_clf['danceability'])
df_clf['mood_label'] = df_clf['mood_score'].apply(categorize_mood)

# Load the pre-trained Random Forest Classification model
rf_model = joblib.load(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Model_implementations\rf_classification_model.joblib")

# Features for classification
X_clf = df_clf[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_min', 'loudness_normalized']]

# Make predictions using the pre-trained model
df_clf['mood_label_from_clf'] = rf_model.predict(X_clf)

# Display classification results
st.subheader("Classification Results")
comparison_class_df = df_clf[['mood_score', 'mood_label', 'mood_label_from_clf']].head(10)
st.write("Comparison of Actual vs Predicted Mood Labels:")
st.dataframe(comparison_class_df)

# Display mood distribution
st.subheader("Mood Distribution")
col1, col2 = st.columns(2)

with col1:
    st.write("**Actual Mood Distribution:**")
    actual_counts = df_clf['mood_label'].value_counts()
    st.dataframe(actual_counts)

with col2:
    st.write("**Predicted Mood Distribution:**")
    predicted_counts = df_clf['mood_label_from_clf'].value_counts()
    st.dataframe(predicted_counts)

st.divider()


# MODEL COMPARISON SECTION

st.header("6. Model Comparison: Linear Regression vs Classification")

st.markdown("""
This section compares the mood predictions from both models:
- **Linear Regression**: Predicts mood score, then categorizes into mood labels
- **Random Forest Classification**: Directly predicts mood labels from features
""")

# Merge the predictions from both models
comparison_data = pd.DataFrame({
    'mood_score': df_clf['mood_score'],
    'actual_mood_label': df_clf['mood_label'],
    'lr_predicted_label': df_lr['mood_label_from_lr'],
    'clf_predicted_label': df_clf['mood_label_from_clf']
})

st.subheader("Side-by-Side Prediction Comparison")
st.dataframe(comparison_data.head(15))

# Calculate accuracy for both models
lr_accuracy = accuracy_score(comparison_data['actual_mood_label'], comparison_data['lr_predicted_label'])
clf_accuracy = accuracy_score(comparison_data['actual_mood_label'], comparison_data['clf_predicted_label'])

col1, col2 = st.columns(2)

with col1:
    st.metric("Linear Regression Accuracy", f"{lr_accuracy:.2%}")
    st.write("**Classification Report (Linear Regression):**")
    lr_report = classification_report(comparison_data['actual_mood_label'], comparison_data['lr_predicted_label'], output_dict=True)
    st.dataframe(pd.DataFrame(lr_report).transpose())

with col2:
    st.metric("Random Forest Classification Accuracy", f"{clf_accuracy:.2%}")
    st.write("**Classification Report (Random Forest):**")
    clf_report = classification_report(comparison_data['actual_mood_label'], comparison_data['clf_predicted_label'], output_dict=True)
    st.dataframe(pd.DataFrame(clf_report).transpose())

# Agreement analysis
agreement = (comparison_data['lr_predicted_label'] == comparison_data['clf_predicted_label']).sum()
agreement_rate = agreement / len(comparison_data)

st.subheader("Model Agreement Analysis")
st.metric("Agreement Rate", f"{agreement_rate:.2%}", help="Percentage of predictions where both models agree")

st.write(f"Both models agree on **{agreement} out of {len(comparison_data)}** predictions.")

# Show cases where models disagree
disagreements = comparison_data[comparison_data['lr_predicted_label'] != comparison_data['clf_predicted_label']]
if len(disagreements) > 0:
    st.write(f"**Cases where models disagree ({len(disagreements)} total):**")
    st.dataframe(disagreements.head(10))

st.divider()

# Final dataset with all predictions
st.header("Final Dataset with All Predictions")
final_df = df_clf.copy()
final_df['mood_label_from_lr'] = df_lr['mood_label_from_lr']
final_df['predicted_mood_score'] = df_lr['predicted_mood_score']
st.dataframe(final_df[['mood_score', 'predicted_mood_score', 'mood_label', 'mood_label_from_lr', 'mood_label_from_clf']].head())

st.divider()
st.markdown("""
### Summary
This final dataset contains predictions from both models:
1. **Linear Regression Model**: Predicts the `mood_score`, which is then categorized into mood labels
2. **Random Forest Classifier**: Directly classifies songs into mood categories using all 14 features

The comparison shows how different modeling approaches can yield different predictions for the same songs.
""")