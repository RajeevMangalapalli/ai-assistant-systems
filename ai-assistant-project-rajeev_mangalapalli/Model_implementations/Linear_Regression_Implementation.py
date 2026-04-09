import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load the transformed dataset
df = pd.read_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\transformed_music_info.csv")


# Define mood score as a combination of valence, energy, and danceability
df['mood_score'] = (0.5 * df['valence']) + (0.3 * df['energy']) + (0.2 * df['danceability'])

#Predict mood_score using Linear Regression and not valence, energy, danceability otherwise it would always match perfectly


# Features and target variable
X = df[['duration_min', 'loudness_normalized', 'year', 'speechiness', 'instrumentalness', 'liveness', 'tempo']]
y = df['mood_score']

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

#New column for predicted mood score
df['predicted_mood_score'] = y_pred

comparison_df = df[['mood_score', 'predicted_mood_score']].head()
print(comparison_df)
#Plotting the graphs of actual vs predicted mood scores

plt.figure(figsize=(10,6))
plt.scatter(df.index, df['mood_score'], color='blue', label='Actual Mood Score')
plt.scatter(df.index, df['predicted_mood_score'], color='red', label='Predicted Mood Score')
plt.xlabel('Index')
plt.ylabel('Mood Score')
plt.title('Actual vs Predicted Mood Scores')
plt.legend()
plt.show()
# Evaluate the model
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error: {mse : .4f}")
print(f"Mean Absolute Error: {mae : .4f}")
print(f"R-squared: {r2 : .4f}")

#Classify the predicted_mood_score into Happy, Sad, Neutral
def categorize_mood(score):
    if score >= 0.6:
        return 'Happy'
    elif score <= 0.4:
        return 'Sad'
    else:
        return 'Neutral'
    
df['mood_label'] = df['predicted_mood_score'].apply(categorize_mood)

# Save the dataframe with predicted mood scores
df.to_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\Linear_Regression_Output.csv", index=False)

#Save the trained model using joblib
joblib.dump(model, r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Model_implementations\linear_regression_model.joblib")