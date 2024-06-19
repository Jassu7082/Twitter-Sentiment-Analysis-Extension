import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import pandas as pd  # Import pandas here to resolve the NameError

nltk.download('punkt')
nltk.download('stopwords')

# Function for text preprocessing
def preprocess_text(text):
    # Check if text is NaN (Not a Number)
    if isinstance(text, float) and pd.isnull(text):
        return ''  # Replace NaN with empty string or handle as per your requirement
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Remove URLs, special characters, and convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().strip()

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

    # Join tokens back into sentence
    return ' '.join(filtered_tokens)
