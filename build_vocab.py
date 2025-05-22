import pandas as pd
from collections import Counter
import pickle

DATA_PATH = r"C:\Users\gurju\OneDrive\Desktop\TwitterSentimentLSTM\data\three_class_dataset_balanced.csv"
VOCAB_PATH = r"C:\Users\gurju\OneDrive\Desktop\TwitterSentimentLSTM\data\vocab.pkl"

df = pd.read_csv(DATA_PATH)

all_tokens = []
for text in df['text']:
    tokens = str(text).lower().split()
    all_tokens.extend(tokens)

counter = Counter(all_tokens)
min_freq = 1
vocab = [word for word, freq in counter.items() if freq >= min_freq]

vocab = ["<pad>", "<unk>"] + vocab
word2idx = {word: idx for idx, word in enumerate(vocab)}

print(f"Vocab size: {len(vocab)}")
print("First 10 vocab entries:", vocab[:10])
print("word2idx sample:", {word: word2idx[word] for word in vocab[:10]})

with open(VOCAB_PATH, "wb") as f:
    pickle.dump((vocab, word2idx), f)
print(f"Saved vocab to {VOCAB_PATH}")