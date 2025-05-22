import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def load_and_preprocess_data(file_path, nrows=None):
    df = pd.read_csv(file_path, encoding='latin-1', header=None, nrows=nrows)
    texts = df[5].astype(str).apply(clean_text)
    labels = df[0].values
    # Remove neutral and map labels
    mask = labels != 2
    texts = texts[mask]
    labels = labels[mask]
    labels = (labels == 4).astype(int)
    return texts, labels