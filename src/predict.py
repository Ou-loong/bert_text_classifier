
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data_loader import load_test_data
import torch

def predict_and_save(model_path="bert_model", output_path="output/submit.txt"):
    df = load_test_data("data/test.jsonl")
    texts = df["text"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).numpy()

    with open(output_path, "w", encoding="utf-8") as f:
        for label in preds:
            f.write(f"{label}\n")

    print(f"已保存预测结果到 {output_path}")
