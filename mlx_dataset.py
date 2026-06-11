import pandas as pd
import json
from sklearn.model_selection import train_test_split

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. "
    "Provide detailed , accurate and educational medical information."
)

df = pd.read_csv("data/medquad_cleaned.csv")

train_df, valid_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
)

def convert_row(row):
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": row["question"],
            },
            {
                "role": "assistant",
                "content": row["answer"],
            },
        ]
    }

with open("mlx_data/train.jsonl", "w") as f:
    for _, row in train_df.iterrows():
        f.write(json.dumps(convert_row(row)) + "\n")

with open("mlx_data/valid.jsonl", "w") as f:
    for _, row in valid_df.iterrows():
        f.write(json.dumps(convert_row(row)) + "\n")

print("Done")