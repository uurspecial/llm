import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines="skip", quoting=3)  # quoting=3 = csv.QUOTE_NONE
    except Exception as e:
        print("讀取 CSV 發生錯誤：", e)
        return None

    def get_label(row):
        if row.get("winner_model_a") == 1:
            return "A"
        elif row.get("winner_model_b") == 1:
            return "B"
        else:
            return "tie"

    df["label"] = df.apply(get_label, axis=1)

    df["text"] = df.apply(
        lambda x: f"[PROMPT]: {x.get('prompt', '')}\n\n[RESPONSE A]: {x.get('response_a', '')}\n\n[RESPONSE B]: {x.get('response_b', '')}",
        axis=1
    )

    df = df[["text", "label"]].dropna().reset_index(drop=True)

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.05, seed=42)
    return dataset
