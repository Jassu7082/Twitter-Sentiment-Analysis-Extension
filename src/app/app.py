from flask import Flask, render_template, request
import joblib
from text_preprocessing import preprocess_text

app = Flask(__name__)

# Load model and vectorizer
model_path = 'src/sentiment_analysis_model.joblib'
vectorizer_path = 'src/models/tfidf_vectorizer.joblib'
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']

    # Preprocess input text
    processed_tweet = preprocess_text(tweet)

    # Vectorize preprocessed text
    vectorized_tweet = vectorizer.transform([processed_tweet]).toarray()

    # Predict sentiment
    prediction = model.predict(vectorized_tweet)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    return render_template('result.html', tweet=tweet, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
