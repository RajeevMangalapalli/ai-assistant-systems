
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the transformed dataset
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\transformed_music_info.csv")

# Classify it into happy, neutral, sad based on danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, decade, duration_min, loudness_normalized

# Features and target variable
X = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_min', 'loudness_normalized']]
y = df['mood_label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Compare predictions with actual values
print("Comparison of Actual and Predicted values:")
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head())
#plot graphs

plt.figure(figsize=(10,6))
sns.countplot(x='mood_label', data=df)
plt.title('Distribution of Mood Labels')
plt.show()
# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Save the features
df.to_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\Classification_model_output.csv", index=False)

#Save the model using joblib
joblib.dump(rf_classifier, r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Model_implementations\rf_classification_model.joblib")
