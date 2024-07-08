from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class SentimentBiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_size):
        super(SentimentBiLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=True)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_size)
    
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1]
        out = self.fc(lstm_out)
        return out

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

embedding_matrix = np.load('embedding_matrix.npy')
model = SentimentBiLSTM(embedding_matrix, hidden_dim=64, output_size=3)
model.load_state_dict(torch.load('sentiment_model.pt'))
model.eval()

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text']
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=167)
    inputs = torch.tensor(padded, dtype=torch.long)
    with torch.no_grad():
        outputs = model(inputs)
        prediction = torch.argmax(outputs, dim=1).item()
        sentiment = ['negative', 'neutral', 'positive'][prediction]
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
