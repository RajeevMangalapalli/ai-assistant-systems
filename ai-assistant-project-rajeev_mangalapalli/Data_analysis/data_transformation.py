#Using the cleaned dataframe from data_analysis.py
#This file contains functions to perform data transformation tasks

#The first transformation is categorical encoding of the "year" column into decades
import pandas as pd
df_cleaned = pd.read_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\cleaned_music_info.csv")

def encode_decade(year):
    if 1960 <= year <= 1969:
        return '1960s'
    elif 1970 <= year <= 1979:
        return '1970s'
    elif 1980 <= year <= 1989:
        return '1980s'
    elif 1990 <= year <= 1999:
        return '1990s'
    elif 2000 <= year <= 2009:
        return '2000s'
    elif 2010 <= year <= 2019:
        return '2010s'
    elif 2020 <= year <= 2023:
        return '2020s'
    else:
        return 'Unknown'

df_cleaned['decade'] = df_cleaned['year'].apply(encode_decade)
print(df_cleaned[['year', 'decade']].head())

#The second transformation is doing the scaling of the "duration_ms" column from milliseconds to minutes
df_cleaned['duration_min'] = df_cleaned['duration_ms'] / 60000

print(df_cleaned[['duration_ms', 'duration_min']].head())

#The third transformation is normalizing the "loudness" column to a 0-1 scale
min_loudness = df_cleaned['loudness'].min()
max_loudness = df_cleaned['loudness'].max()
df_cleaned['loudness_normalized'] = (df_cleaned['loudness'] - min_loudness) / (max_loudness - min_loudness)
print(df_cleaned[['loudness', 'loudness_normalized']].head())

#Adding a mood score
df_cleaned['mood_score'] = df_cleaned['valence'] * 0.5 + df_cleaned['energy'] * 0.3 + df_cleaned['danceability'] * 0.2


#Classifying the songs into mood categories based on valence since we don't have a mood_score column yet
def classify_mood(valence):
    if valence >= 0.6:
        return 'Happy'
    elif valence <= 0.4:
        return 'Sad'
    else:
        return 'Neutral'

df_cleaned['mood_label'] = df_cleaned['valence'].apply(classify_mood)
print(df_cleaned[['valence', 'mood_label']].head())
#Saving the transformed dataframe to a new CSV file
df_cleaned.to_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\transformed_music_info.csv", index=False)
