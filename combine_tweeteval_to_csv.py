import pandas as pd

# Absolute paths
text_path = r"C:\Users\gurju\OneDrive\Desktop\TwitterSentimentLSTM\data\train_text.txt"
label_path = r"C:\Users\gurju\OneDrive\Desktop\TwitterSentimentLSTM\data\train_labels.txt"

with open(text_path, encoding="utf-8") as f_text, open(label_path) as f_label:
    texts = [line.strip() for line in f_text]
    labels = [line.strip() for line in f_label]

assert len(texts) == len(labels), "Text and label files have different lengths!"

df = pd.DataFrame({"sentiment": labels, "text": texts})
df.to_csv(r"C:\Users\gurju\OneDrive\Desktop\TwitterSentimentLSTM\data\three_class_dataset.csv", index=False)
print("Saved as three_class_dataset.csv")