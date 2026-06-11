import torch
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from datasets import Dataset

MAX_LENGTH=1024

df=pd.read_csv("data/medquad_cleaned.csv")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

system=(
    "You are a helpful medical assistant."
    "Provide accurate and educational medical information."
)

def preprocess_example(question: str, answer: str):
    prompt_messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    full_messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_tokens = tokenizer(prompt_text)
    full_tokens = tokenizer(full_text)

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    prompt_length = len(prompt_tokens["input_ids"])

    labels = input_ids.copy()

    labels[:prompt_length] = [-100] * prompt_length

    if len(input_ids) > MAX_LENGTH:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

sample = df.iloc[0]

result = preprocess_example(
    sample["question"],
    sample["answer"]
)

processed_dataset = []

skipped = 0

for idx, row in df.iterrows():

    item = preprocess_example(
        row["question"],
        row["answer"]
    )

    if item is None:
        skipped += 1
        continue

    processed_dataset.append(item)

    if idx % 1000 == 0:
        print(
            f"Processed: {idx} | "
            f"Kept: {len(processed_dataset)} | "
            f"Skipped: {skipped}"
        )

print("\nFinished")
print("Kept:", len(processed_dataset))
print("Skipped:", skipped)



hf_dataset = Dataset.from_list(
    processed_dataset
)

print(hf_dataset)

split_dataset = hf_dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(len(train_dataset))
print(len(eval_dataset))

train_dataset.save_to_disk(
    "data/train_dataset"
)

eval_dataset.save_to_disk(
    "data/eval_dataset"
)