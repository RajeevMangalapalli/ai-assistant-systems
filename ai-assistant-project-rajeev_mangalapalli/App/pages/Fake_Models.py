
#The actual recommendation model implementations will be here
#as the sliders and stuff to take user input for recommendation parameters
import random
import pandas as pd
import streamlit as st

def app():
    st.title("🧠 Fake Models")
    st.write("Details and performance of fake/simulated models.")

st.title("Fake Models Page")
st.write("This page will contain the recommendation model implementations using the augmented dataset.")

# Create the sliders for user input

happiness_level = st.slider("Happiness level (1-10)", 0,100,1)
energy_level = st.slider("Energy level (1-10)", 0,100,1)
danceability = st.slider("Danceability (1-10)", 0,100,1)

happiness_level_sclaed = happiness_level/100
energy_level_scaled = energy_level/100
danceability_scaled = danceability/100


st.write("Happiness Level:", happiness_level)
st.write("Energy Level:", energy_level)
st.write("Danceability:", danceability)

# Here, you would typically load your trained recommendation model

st.write("Based on the input parameters, the recommended songs will be displayed here.")

#Final output based on the user input sliders and the recommendation model
@st.cache_data
def load_dataset_classification():
    return pd.read_csv(
        r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\Classification_model_output_augmented.csv"
    )
@st.cache_data
def load_dataset_regression():
    return pd.read_csv(
        r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\Linear_Regression_Output.csv"
    )

# From the data set, filter songs based on user input
df_clf = load_dataset_classification()
filtered_songs = df_clf[
    (df_clf["mood_score"] >= happiness_level_sclaed - 0.1) &
    (df_clf["mood_score"] <= happiness_level_sclaed + 0.1) &
    (df_clf["energy"] >= energy_level_scaled - 0.1) &
    (df_clf["energy"] <= energy_level_scaled + 0.1) &
    (df_clf["danceability"] >= danceability_scaled - 0.1) &
    (df_clf["danceability"] <= danceability_scaled + 0.1)
]

if not filtered_songs.empty:
    song = filtered_songs.iloc[random.randint(0, len(filtered_songs) - 1)]
    st.write("Recommended Song:")
    st.write(f"- {song['name']} by {song['artist']} ({song['mood_label']})")
else:
    st.write("No songs found matching the criteria. Please adjust the sliders and try again.")




df_lr = load_dataset_regression()

filtered_songs_lr = df_lr[
    (df_lr["predicted_mood_score"] >= happiness_level_sclaed - 0.1) &
    (df_lr["predicted_mood_score"] <= happiness_level_sclaed + 0.1) &
    (df_lr["energy"] >= energy_level_scaled - 0.1) &
    (df_lr["energy"] <= energy_level_scaled + 0.1) &
    (df_lr["danceability"] >= danceability_scaled - 0.1) &
    (df_lr["danceability"] <= danceability_scaled + 0.1)
]
if not filtered_songs_lr.empty:
    song_lr = filtered_songs_lr.iloc[random.randint(0, len(filtered_songs_lr) - 1)]
    st.write("Recommended Song from Regression Model:")
    st.write(f"- {song_lr['name']} by {song_lr['artist']} ({song_lr['mood_label']})")
else:
    st.write("No songs found matching the criteria in Regression Model. Please adjust the sliders and try again.")