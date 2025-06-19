
import json
import pandas as pd

def load_train_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({"text": item["text"], "label": int(item["label"])})
    return pd.DataFrame(data)

def load_test_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({"text": item["text"]})
    return pd.DataFrame(data)
