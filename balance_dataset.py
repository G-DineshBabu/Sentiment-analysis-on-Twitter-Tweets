import pandas as pd

print("Loading data...")
df = pd.read_csv(r"C:\Users\gurju\OneDrive\Desktop\TwitterSentimentLSTM\data\three_class_dataset.csv")
print("Data loaded. Shape:", df.shape)
print("Value counts:\n", df['sentiment'].value_counts())

target_count = df['sentiment'].value_counts().min()
print("Target count for balancing:", target_count)

df_balanced = (
    df.groupby('sentiment', group_keys=False)
      .apply(lambda x: x.sample(target_count, random_state=42))
      .sample(frac=1, random_state=42)
)

print("Balanced data shape:", df_balanced.shape)

output_path = r"C:\Users\gurju\OneDrive\Desktop\TwitterSentimentLSTM\data\three_class_dataset_balanced.csv"
df_balanced.to_csv(output_path, index=False)
print(f"Balanced dataset saved as {output_path}")