import torch
import re
import pickle
from train_lstm_model import SentimentLSTM  # Make sure this import works or define the class here

# Load vocab and word2idx
with open("vocab.pkl", "rb") as f:
    vocab, word2idx = pickle.load(f)

# Load model
model = SentimentLSTM(len(vocab), 128, 128, 3)
model.load_state_dict(torch.load(r"C:\Users\gurju\OneDrive\Desktop\sentiment_lstm.pth", map_location="cpu"))
model.eval()

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def encode(tokens):
    return [word2idx.get(token, word2idx["<unk>"]) for token in tokens]

def predict_sentiment(text):
    tokens = tokenize(text)
    input_ids = torch.tensor([encode(tokens)], dtype=torch.long)
    lengths = torch.tensor([len(input_ids[0])])
    with torch.no_grad():
        logits = model(input_ids, lengths)
        pred = logits.argmax(dim=1).item()
    return ["Negative", "Neutral", "Positive"][pred]

# Example
tweet = "I absolutely love this new feature!"
print(f"Tweet: {tweet}")
print(f"Predicted Sentiment: {predict_sentiment(tweet)}")