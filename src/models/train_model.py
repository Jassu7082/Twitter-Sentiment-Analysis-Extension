import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load sample data from sklearn's 20 newsgroups dataset
data = fetch_20newsgroups(subset='all', categories=None, shuffle=True, random_state=42)

# Create DataFrame
df = pd.DataFrame({'text': data.data, 'label': data.target})

# Preprocess text data
stop_words = 'english'
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert X_train and y_train to DataFrames and save to CSV
X_train_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
y_train_df = pd.DataFrame({'label': y_train})
X_train_df.to_csv('../../data/X_train.csv', index=False)
y_train_df.to_csv('../../data/y_train.csv', index=False)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Save the trained model to disk
model_path = '../../src/sentiment_analysis_model.joblib'
joblib.dump(classifier, model_path)

print("Model training complete.")
