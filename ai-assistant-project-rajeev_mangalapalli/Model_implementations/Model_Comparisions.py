# music_model_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. Load datasets
# ==============================
df_reg = pd.read_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\Linear_Regression_Output.csv")
df_clf = pd.read_csv(r"C:\Users\rajee\Desktop\VS code\Assistant_Systems\ai-assistant-project-rajeev_mangalapalli\Data_analysis\Classification_Model_Output.csv")

# ==============================
# 2. Define mood label function
# ==============================
def mood_from_score(score):
    if score <= 0.4:
        return 'Sad'
    elif score <= 0.6:
        return 'Neutral'
    else:
        return 'Happy'

# ==============================
# 3. Regression: Predicted Mood Label
# ==============================
df_reg['predicted_mood_label'] = df_reg['predicted_mood_score'].apply(mood_from_score)
# True mood label (already exists)
df_reg['mood_label'] = df_reg['mood_score'].apply(mood_from_score)

# ==============================
# 4. Classification: Predicted Mood Label
# ==============================
# Create predicted mood label based on mood_score (or predicted score if available)
df_clf['pre_mood_label'] = df_clf['mood_score'].apply(mood_from_score)
# True mood label already exists in df_clf['mood_label']

# ==============================
# 5. Plot Regression Mood Distribution
# ==============================
plt.figure(figsize=(10,6))
sns.histplot(df_reg['predicted_mood_label'], bins=30, kde=True, color='skyblue')
plt.title('Linear Regression – Predicted Mood Label Distribution')
plt.xlabel('Predicted Mood Label')
plt.ylabel('Number of Songs')
plt.tight_layout()
plt.savefig("Linear_Regression_Mood_Distribution.png")
plt.show()

# ==============================
# 6. Plot Regression True vs Predicted
# ==============================
plt.figure(figsize=(10,6))
sns.countplot(x='mood_label', hue='predicted_mood_label', data=df_reg)
plt.title('Linear Regression – True vs Predicted Mood Labels')
plt.xlabel('True Mood Label')
plt.ylabel('Number of Songs')
plt.legend(title='Predicted Mood Label')
plt.tight_layout()
plt.savefig("Linear_Regression_True_vs_Predicted.png")
plt.show()

# ==============================
# 7. Plot Classification Mood Distribution
# ==============================
plt.figure(figsize=(10,6))
sns.countplot(x='mood_label', data=df_clf, color='salmon')
plt.title('Random Forest – Mood Prediction Distribution')
plt.xlabel('Mood Label')
plt.ylabel('Number of Songs')
plt.tight_layout()
plt.savefig("Random_Forest_Mood_Distribution.png")
plt.show()

# ==============================
# 8. Calculate Accuracies
# ==============================
lr_accuracy = accuracy_score(df_reg['mood_label'], df_reg['predicted_mood_label'])
clf_accuracy = accuracy_score(df_clf['mood_label'], df_clf['pre_mood_label'])

# ==============================
# 9. Plot Accuracy Comparison
# ==============================
acc_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'Accuracy': [lr_accuracy, clf_accuracy]
})

plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='Accuracy', data=acc_df, palette=['skyblue','salmon'])
plt.ylim(0,1)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig("Model_Accuracy_Comparison.png")
plt.show()

# ==============================
# 10. Print Classification Reports
# ==============================
print("=== Linear Regression Classification Report ===")
print(classification_report(df_reg['mood_label'], df_reg['predicted_mood_label']))
print(f"Accuracy: {lr_accuracy:.4f}")

print("=== Random Forest Classification Report ===")
print(classification_report(df_clf['mood_label'], df_clf['pre_mood_label']))
print(f"Accuracy: {clf_accuracy:.4f}")
