import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from text_preprocessing import preprocess_text

# Load data
data_path = '../../data/train.csv'  # Adjust path as per your directory structure
data = pd.read_csv(data_path)

# Check the column names to ensure 'tweet' is correct
print(data.columns)

# Preprocess text data (assuming 'tweet' is the correct column name)
data['processed_text'] = data['tweet'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['processed_text']).toarray()
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save vectorizer and processed data
vectorizer_path = '../../src/models/tfidf_vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_path)

# Optionally, you can save X_train, X_test, y_train, y_test for future use

print("Data preparation and preprocessing complete.")
