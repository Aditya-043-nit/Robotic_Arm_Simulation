import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "commands.csv"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "intent_model"
DEFAULT_RUN_DIR = BASE_DIR / "models" / "intent_training_runs"


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DistilBERT intent classifier.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy}


def main():
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path).dropna()
    required_columns = {"sentence", "intent"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns {required_columns}. Found: {list(df.columns)}"
        )

    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["sentence"] != "") & (df["intent"] != "")]

    labels = sorted(df["intent"].unique().tolist())
    label_map = {label: idx for idx, label in enumerate(labels)}
    df["label"] = df["intent"].map(label_map)

    stratify = df["label"] if df["label"].nunique() > 1 else None
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["sentence"].tolist(),
        df["label"].tolist(),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)

    id_to_label = {v: k for k, v in label_map.items()}
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_map),
        id2label=id_to_label,
        label2id=label_map,
    )

    training_args = TrainingArguments(
        output_dir=str(args.run_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    model.save_pretrained(str(args.model_dir))
    tokenizer.save_pretrained(str(args.model_dir))

    with open(args.model_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"Saved intent model to: {args.model_dir}")
    print(f"Validation metrics: {eval_metrics}")


if __name__ == "__main__":
    main()
