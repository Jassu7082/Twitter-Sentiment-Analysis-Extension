import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm.auto import tqdm 
import pickle
import os
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re 
from collections import Counter
from string import punctuation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import torch 
import torch.nn as nn  
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import ConfusionMatrix 
from mlxtend.plotting import plot_confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize objects
lemma = WordNetLemmatizer()
lb = LabelEncoder()

# Load data
print("Loading data...")
df = pd.read_csv('../../data/train.csv')

# Print columns to identify the correct ones
print(df.columns)

# Adjust column names accordingly   
df = df.rename(columns={"name": "Feature2", "tweet": "Feature1", "label": "labels"})
df["tweets"] = df["Feature1"].astype(str) + " " + df["Feature2"].astype(str)
df = df.drop(["Feature1", "Feature2"], axis=1)
df_labels = {key: value for value, key in enumerate(np.unique(df['labels']))}
def getlabel(n):
    for x, y in df_labels.items():
        if y == n:
            return x

def Most_Words_used(tweets, num_of_words):
    all_text = ''.join(df[tweets].values)
    all_text = re.sub('<.*?>', '', all_text)  # HTML tags
    all_text = re.sub(r'\d+', '', all_text)  # numbers
    all_text = re.sub(r'[^\w\s]', '', all_text)  # special characters
    all_text = re.sub(r'http\S+', '', all_text)  # URLs or web links
    all_text = re.sub(r'@\S+', '', all_text)  # mentions
    all_text = re.sub(r'#\S+', '', all_text)  # hashtags

    words = all_text.split()
    punc = list(punctuation)
    words = [word for word in words if word not in punc]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]

    word_counts = Counter(words)
    top_words = word_counts.most_common(num_of_words)
    return top_words

print("Calculating most used words...")
top_words = Most_Words_used('tweets', 50)
xaxis = [word[0] for word in top_words]
yaxis = [word[1] for word in top_words]

def DataPrep(text):
    text = re.sub('<.*?>', '', text)  # HTML tags
    text = re.sub(r'\d+', '', text)  # numbers
    text = re.sub(r'[^\w\s]', '', text)  # special characters
    text = re.sub(r'http\S+', '', text)  # URLs or web links
    text = re.sub(r'@\S+', '', text)  # mentions
    text = re.sub(r'#\S+', '', text)  # hashtags

    tokens = nltk.word_tokenize(text)
    punc = list(punctuation)
    words = [word for word in tokens if word not in punc]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word.lower() in stop_words]
    words = [lemma.lemmatize(word) for word in words]

    text = ' '.join(words)
    return text

print("Cleaning tweets...")
df['cleaned_tweets'] = df['tweets'].apply(DataPrep)
df.drop_duplicates("cleaned_tweets", inplace=True)

df['tweet_len'] = [len(text.split()) for text in df.cleaned_tweets]
df = df[df['tweet_len'] < df['tweet_len'].quantile(0.995)]
x_train, x_val, y_train, y_val = train_test_split(df['cleaned_tweets'], df['labels'], train_size=0.85, random_state=42)

# Vectorization
print("Vectorizing tweets...")
vec = TfidfVectorizer()
vec.fit(x_train)
x_train = vec.transform(x_train).toarray()
x_val = vec.transform(x_val).toarray()
y_train = lb.fit_transform(y_train)
y_val = lb.transform(y_val)

# Save the vectorizer and label encoder
os.makedirs('../../src', exist_ok=True)

# Save the trained TF-IDF vectorizer to a file
with open('src/tfidf_vectorizer.pkl', 'wb') as f:
    print("saved vector")
    pickle.dump(vec, f)

# Save the label encoder to a file
with open('src/label_encoder.pkl', 'wb') as f:
    print("saved encoder")
    pickle.dump(lb, f)
# Create PyTorch dataset
print("Creating PyTorch dataset...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model
print("Defining model...")
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_dim = x_train.shape[1]
output_dim = len(df_labels)
model = SimpleNN(input_dim, output_dim).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training Batches"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        for inputs, labels in tqdm(val_loader, desc="Validation Batches"):
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='weighted')
        val_recall = recall_score(all_labels, all_preds, average='weighted')
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f'Epoch {epoch+1}/{num_epochs} completed')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Acc: {val_acc:.4f}')
        print(f'Val Precision: {val_precision:.4f}')
        print(f'Val Recall: {val_recall:.4f}')
        print(f'Val F1: {val_f1:.4f}')
        
        # Save the model at each epoch
        if epoch==9:
            print("save model")
            torch.save(model.state_dict(), f'src/model_epoch_{epoch+1}.pth')

train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# Evaluation and confusion matrix
print("Evaluating model...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc="Final Evaluation Batches"):
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='3g')
plt.show()
