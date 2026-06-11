import pandas as pd

df = pd.read_csv("data/medquad.csv")

original=len(df)
print("Initial:",len(df))
print(df.columns)

df = df.drop_duplicates()

df["question_normalized"] = (
    df["question"]
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

duplicate_questions = (
    df["question_normalized"]
    .duplicated()
    .sum()
)

df = df.drop_duplicates(
    subset=["question_normalized"],
    keep="first"
)

df = df.dropna(subset=["question", "answer"])

print("final rows: ",len(df) )
print("Removed: ",original-len(df))

print(df.isnull().sum())
df.to_csv("data/medquad_cleaned.csv", index=False)