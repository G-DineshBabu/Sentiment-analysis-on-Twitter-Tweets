import pickle

VOCAB_PATH = r"C:\Users\gurju\OneDrive\Desktop\TwitterSentimentLSTM\data\vocab.pkl"

with open(VOCAB_PATH, "rb") as f:
    vocab, word2idx = pickle.load(f)

print("Vocab size:", len(vocab))
print("First 10 vocab entries:", vocab[:10])
print("word2idx sample:", {word: word2idx[word] for word in vocab[:10]})