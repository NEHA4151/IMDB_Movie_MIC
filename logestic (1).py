# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from collections import Counter
import re
data_path = "/content/drive/MyDrive/IMDB Dataset.csv"
df = pd.read_csv(data_path)
print("Dataset Shape:", df.shape)
print(df.head())
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df, palette=['#FF6F61','#6B5B95'])
plt.title("Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

df['review_length'] = df['review'].apply(len)

plt.figure(figsize=(8,5))
sns.histplot(data=df, x='review_length', hue='sentiment', bins=50, palette=['#FF6F61','#6B5B95'], kde=True)
plt.title("Distribution of Review Lengths by Sentiment", fontsize=14)
plt.xlabel("Review Length (characters)")
plt.ylabel("Count")
plt.show()
avg_length = df.groupby('sentiment')['review_length'].mean().reset_index()
plt.figure(figsize=(6,4))
sns.barplot(x='sentiment', y='review_length', data=avg_length, palette=['#FF6F61','#6B5B95'])
plt.title("Average Review Length by Sentiment", fontsize=14)
plt.ylabel("Average Length")
plt.show()
def get_top_words(reviews, n=20):
    words = ' '.join(reviews).lower()
    words = re.findall(r'\b\w+\b', words)
    return Counter(words).most_common(n)
top_pos = get_top_words(df[df['sentiment']=='positive']['review'])
words, counts = zip(*top_pos)
plt.figure(figsize=(10,5))
sns.barplot(x=list(counts), y=list(words), palette='Greens_r')
plt.title("Top 20 Words in Positive Reviews", fontsize=14)
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.show()
top_neg = get_top_words(df[df['sentiment']=='negative']['review'])
words, counts = zip(*top_neg)
plt.figure(figsize=(10,5))
sns.barplot(x=list(counts), y=list(words), palette='Reds_r')
plt.title("Top 20 Words in Negative Reviews", fontsize=14)
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.show()
X = df['review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Predictions
y_pred = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nLogistic Regression Performance:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral', cbar=True,
            xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix - Logistic Regression', fontsize=14)
plt.show()

metrics_dict = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}
plt.figure(figsize=(8,5))
colors = ['#FF6F61', '#6B5B95', '#88B04B', '#FFA500']
sns.barplot(x=list(metrics_dict.keys()), y=list(metrics_dict.values()), palette=colors)
plt.ylim(0, 1.1)
plt.title("Logistic Regression Performance Metrics", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Metric", fontsize=12)
for i, v in enumerate(metrics_dict.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=11)
plt.show()
log_model_path = "/content/drive/MyDrive/logistic_model.pkl"
joblib.dump(log_model, log_model_path)
