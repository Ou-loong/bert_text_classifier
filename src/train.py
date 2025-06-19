
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from src.data_loader import load_train_data
from src.metrics import compute_metrics

def train_bert(model_name="bert-base-uncased"):
    df = load_train_data("data/train.jsonl")
    dataset = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    tokenized = dataset.map(tokenize)
    tokenized = tokenized.train_test_split(test_size=0.1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./bert_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir='./logs',
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("bert_model")
    tokenizer.save_pretrained("bert_model")
