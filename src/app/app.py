from flask import Flask, request
from flask_cors import CORS
import joblib
from text_preprocessing import preprocess_text

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "chrome-extension://bgnimaldnhlmdcmmkajkgkedipafnioe"}})

# Load model and vectorizer
model_path = 'src/sentiment_analysis_model.joblib'
vectorizer_path = 'src/models/tfidf_vectorizer.joblib'
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tweet = data['tweet']

    # Preprocess input text
    processed_tweet = preprocess_text(tweet)

    # Vectorize preprocessed text
    vectorized_tweet = vectorizer.transform([processed_tweet]).toarray()

    # Predict sentiment
    prediction = model.predict(vectorized_tweet)[0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"

    return sentiment

if __name__ == '__main__':
    app.run(debug=True)
