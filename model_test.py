# model_test.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model("lstm_sentiment_model.h5")

# Predict on a new sentence
def predict_sentiment(text):
    from data_preprocessing import clean_text

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    prediction = model.predict(padded)[0][0]
    label = "Positive" if prediction > 0.5 else "Negative"
    return f"Predicted Sentiment: {label} ({prediction:.2f})"

# Test
print(predict_sentiment("I love using this new app, it's amazing!"))
print(predict_sentiment("This is the worst experience I've ever had."))
