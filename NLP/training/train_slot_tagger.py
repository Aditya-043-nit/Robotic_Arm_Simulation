import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "slot_commands.jsonl"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "slot_model"
DEFAULT_RUN_DIR = BASE_DIR / "models" / "slot_training_runs"


class TokenTaggingDataset(torch.utils.data.Dataset):
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
    parser = argparse.ArgumentParser(description="Train DistilBERT slot tagger.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_examples(data_path):
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            tokens = example["tokens"]
            tags = example["tags"]
            if len(tokens) != len(tags):
                raise ValueError(f"Token/tag length mismatch in sample: {example}")
            examples.append(example)
    if not examples:
        raise ValueError(f"No examples found in {data_path}")
    return examples


def build_label_map(examples):
    labels = set()
    for example in examples:
        labels.update(example["tags"])
    ordered = ["O"] + sorted([label for label in labels if label != "O"])
    return {label: idx for idx, label in enumerate(ordered)}


def tokenize_and_align_labels(tokenizer, examples, label_map):
    tokens_batch = [example["tokens"] for example in examples]
    tags_batch = [example["tags"] for example in examples]

    encodings = tokenizer(
        tokens_batch,
        is_split_into_words=True,
        truncation=True,
        padding=True,
    )

    aligned_labels = []
    for sample_idx, word_labels in enumerate(tags_batch):
        word_ids = encodings.word_ids(batch_index=sample_idx)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
                continue

            current_label = word_labels[word_idx]
            if word_idx != prev_word_idx:
                label_ids.append(label_map[current_label])
            else:
                if current_label.startswith("B-"):
                    inside_label = "I-" + current_label[2:]
                    label_ids.append(label_map.get(inside_label, label_map[current_label]))
                else:
                    label_ids.append(label_map[current_label])
            prev_word_idx = word_idx
        aligned_labels.append(label_ids)

    return encodings, aligned_labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)

    total = 0
    correct = 0

    for pred_seq, label_seq in zip(preds, labels):
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            total += 1
            if pred_id == label_id:
                correct += 1

    accuracy = float(correct / total) if total else 0.0
    return {"token_accuracy": accuracy}


def main():
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.data_path)
    label_map = build_label_map(examples)
    id_to_label = {v: k for k, v in label_map.items()}

    train_examples, val_examples = train_test_split(
        examples, test_size=args.test_size, random_state=args.seed
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_encodings, train_labels = tokenize_and_align_labels(
        tokenizer, train_examples, label_map
    )
    val_encodings, val_labels = tokenize_and_align_labels(tokenizer, val_examples, label_map)

    train_dataset = TokenTaggingDataset(train_encodings, train_labels)
    val_dataset = TokenTaggingDataset(val_encodings, val_labels)

    model = DistilBertForTokenClassification.from_pretrained(
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
        metric_for_best_model="token_accuracy",
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

    print(f"Saved slot model to: {args.model_dir}")
    print(f"Validation metrics: {eval_metrics}")


if __name__ == "__main__":
    main()
