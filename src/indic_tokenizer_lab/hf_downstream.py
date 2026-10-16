"""Controlled Hugging Face sequence-classification comparison.

Each checkpoint gets the same supervised data, optimizer, epoch limit, and seed.
The validation split selects the best epoch; the test split is evaluated once at
the end. This is intended as a publication-oriented downstream task, unlike the
lightweight TF-IDF screening command in :mod:`downstream`.
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from .text import normalize


def _read_labeled(path: str | Path, text_column: str, label_column: str):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or text_column not in rows[0] or label_column not in rows[0]:
        raise ValueError(f"Expected nonempty CSV columns '{text_column}' and '{label_column}' in {path}")
    texts, labels = [], []
    for row in rows:
        text, label = row.get(text_column), row.get(label_column)
        if text is None or not str(text).strip() or label is None or not str(label).strip():
            continue
        texts.append(normalize(str(text)))
        labels.append(str(label).strip())
    if not texts:
        raise ValueError(f"No labeled text rows found in {path}")
    return texts, labels


class _EncodedDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = {key: torch.tensor(values[index], dtype=torch.long)
               for key, values in self.encodings.items()}
        row["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return row


def _encode(tokenizer, texts: list[str], labels: list[int], max_length: int):
    encodings = tokenizer(texts, truncation=True, max_length=max_length,
                          add_special_tokens=True, padding=False)
    content_encodings = tokenizer(texts, truncation=True, max_length=max_length,
                                   add_special_tokens=False, padding=False)
    lengths = [len(ids) for ids in content_encodings["input_ids"]]
    return _EncodedDataset(dict(encodings), labels), lengths


def _collator(tokenizer):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad token; choose a compatible pretrained checkpoint")

    def collate(rows):
        batch = {}
        max_len = max(len(row["input_ids"]) for row in rows)
        for key in rows[0]:
            if key == "labels":
                batch[key] = torch.stack([row[key] for row in rows])
                continue
            fill = pad_id if key == "input_ids" else 0
            batch[key] = torch.stack([
                torch.nn.functional.pad(row[key], (0, max_len - len(row[key])), value=fill)
                for row in rows
            ])
        return batch
    return collate


def _predict(model, loader, device):
    model.eval()
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    elapsed = time.perf_counter() - started
    return predictions, elapsed


def _fit_condition(checkpoint: str, train_text: list[str], train_y: list[int],
                   val_text: list[str], val_y: list[int], test_text: list[str],
                   test_y: list[int], label_names: list[str], output: Path, *,
                   epochs: int, batch_size: int, max_length: int, learning_rate: float,
                   weight_decay: float, patience: int, seed: int, device: torch.device) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(label_names),
        id2label={i: label for i, label in enumerate(label_names)},
        label2id={label: i for i, label in enumerate(label_names)},
    )
    train_data, train_lengths = _encode(tokenizer, train_text, train_y, max_length)
    val_data, val_lengths = _encode(tokenizer, val_text, val_y, max_length)
    test_data, test_lengths = _encode(tokenizer, test_text, test_y, max_length)
    collate = _collator(tokenizer)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_f1, best_epoch, stale_epochs, best_state = -1.0, 0, 0, None
    history = []
    train_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            label_batch = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, labels=label_batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        # Validation selects epoch only. Never use test scores for early stopping.
        val_predictions, _ = _predict(model, val_loader, device)
        val_f1 = float(f1_score(val_y, val_predictions, average="macro", zero_division=0))
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "validation_macro_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_epoch, stale_epochs = val_f1, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    training_seconds = time.perf_counter() - train_started
    if best_state is not None:
        model.load_state_dict(best_state)
    test_predictions, inference_seconds = _predict(model, test_loader, device)
    # Tokenize timing and lengths give a comparable encoding-efficiency diagnostic.
    tokenization_started = time.perf_counter()
    tokenizer(test_text, truncation=True, max_length=max_length, padding=False)
    tokenization_seconds = time.perf_counter() - tokenization_started
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "best_model")
    tokenizer.save_pretrained(output / "best_model")
    return {
        "checkpoint": checkpoint, "accuracy": float(accuracy_score(test_y, test_predictions)),
        "macro_f1": float(f1_score(test_y, test_predictions, average="macro", labels=list(range(len(label_names))), zero_division=0)),
        "classification_report": classification_report(test_y, test_predictions,
            labels=list(range(len(label_names))), target_names=label_names,
            output_dict=True, zero_division=0),
        "best_epoch_by_validation_macro_f1": best_epoch,
        "best_validation_macro_f1": best_f1, "epochs_run": len(history),
        "training_seconds": training_seconds, "test_inference_seconds": inference_seconds,
        "test_examples_per_second": len(test_y) / max(inference_seconds, 1e-12),
        "test_tokenization_seconds": tokenization_seconds,
        "mean_test_subword_tokens_per_example": float(np.mean(test_lengths)),
        "mean_train_subword_tokens_per_example": float(np.mean(train_lengths)),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "history": history, "predictions": test_predictions,
    }


def run_hf_classification(train_path: str | Path, validation_path: str | Path,
                          test_path: str | Path, checkpoints: dict[str, str], out_dir: str | Path,
                          text_column: str = "tweet", label_column: str = "sentiment",
                          epochs: int = 5, batch_size: int = 8, max_length: int = 128,
                          learning_rate: float = 2e-5, weight_decay: float = 0.01,
                          patience: int = 2, seed: int = 17, device_name: str = "auto",
                          bootstrap_samples: int = 2000) -> dict:
    """Fine-tune identical sequence-classification heads for each model checkpoint."""
    if "baseline" not in checkpoints:
        raise ValueError("Provide --checkpoint baseline=PATH for the original-tokenizer baseline")
    if epochs < 1 or batch_size < 1 or max_length < 2 or patience < 1 or bootstrap_samples < 1:
        raise ValueError("epochs, batch size, max length, patience, and bootstrap samples must be positive")
    incomplete = []
    for name, checkpoint in checkpoints.items():
        local_path = Path(checkpoint)
        if local_path.exists() and local_path.is_dir() and not (local_path / "config.json").is_file():
            incomplete.append(f"{name}={checkpoint}")
    if incomplete:
        paths = "\n".join(f"  - {item}" for item in incomplete)
        raise ValueError(
            "These local checkpoint folders do not contain config.json and are not usable models:\n"
            f"{paths}\n"
            "The adapt-mlm command creates its output folder before training and saves model files "
            "only when training finishes. Rerun each affected adapt-mlm command and wait for its "
            "JSON report to print before running hf-classify."
        )
    train_text, train_labels = _read_labeled(train_path, text_column, label_column)
    val_text, val_labels = _read_labeled(validation_path, text_column, label_column)
    test_text, test_labels = _read_labeled(test_path, text_column, label_column)
    label_names = sorted(set(train_labels))
    label_to_id = {label: i for i, label in enumerate(label_names)}
    for split_name, labels in (("validation", val_labels), ("test", test_labels)):
        unseen = set(labels) - set(label_names)
        if unseen:
            raise ValueError(f"{split_name} contains labels absent from training: {sorted(unseen)}")
    train_y = [label_to_id[label] for label in train_labels]
    val_y = [label_to_id[label] for label in val_labels]
    test_y = [label_to_id[label] for label in test_labels]
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    metrics, predictions = {}, {}
    for name, checkpoint in checkpoints.items():
        result = _fit_condition(checkpoint, train_text, train_y, val_text, val_y,
            test_text, test_y, label_names, out / name, epochs=epochs,
            batch_size=batch_size, max_length=max_length, learning_rate=learning_rate,
            weight_decay=weight_decay, patience=patience, seed=seed, device=device)
        predictions[name] = result.pop("predictions")
        metrics[name] = result

    baseline_correct = np.asarray(predictions["baseline"]) == np.asarray(test_y)
    paired = {}
    rng = np.random.default_rng(seed)
    baseline_f1 = f1_score(test_y, predictions["baseline"], average="macro",
                           labels=list(range(len(label_names))), zero_division=0)
    for name, pred in predictions.items():
        if name == "baseline":
            continue
        adapted_correct = np.asarray(pred) == np.asarray(test_y)
        wins = int(np.sum(adapted_correct & ~baseline_correct))
        losses = int(np.sum(baseline_correct & ~adapted_correct))
        accuracy_delta = float(accuracy_score(test_y, pred) - accuracy_score(test_y, predictions["baseline"]))
        f1_delta = float(f1_score(test_y, pred, average="macro", labels=list(range(len(label_names))),
                                  zero_division=0) - baseline_f1)
        acc_boot, f1_boot = [], []
        for _ in range(bootstrap_samples):
            indexes = rng.integers(0, len(test_y), size=len(test_y))
            y_sample = np.asarray(test_y)[indexes]
            acc_boot.append(accuracy_score(y_sample, np.asarray(pred)[indexes]) -
                            accuracy_score(y_sample, np.asarray(predictions["baseline"])[indexes]))
            f1_boot.append(f1_score(y_sample, np.asarray(pred)[indexes], average="macro",
                                    labels=list(range(len(label_names))), zero_division=0) -
                           f1_score(y_sample, np.asarray(predictions["baseline"])[indexes], average="macro",
                                    labels=list(range(len(label_names))), zero_division=0))
        paired[name] = {"correct_only_condition": wins, "correct_only_baseline": losses,
                        "accuracy_delta_vs_baseline": accuracy_delta,
                        "accuracy_delta_95pct_paired_bootstrap_ci": [float(x) for x in np.percentile(acc_boot, [2.5, 97.5])],
                        "macro_f1_delta_vs_baseline": f1_delta,
                        "macro_f1_delta_95pct_paired_bootstrap_ci": [float(x) for x in np.percentile(f1_boot, [2.5, 97.5])],
                        "mcnemar_exact_p": float(binomtest(wins, wins + losses, .5).pvalue)
                        if wins + losses else 1.0}

    report = {
        "metrics": metrics, "paired_vs_baseline": paired,
        "labels_in_id_order": label_names,
        "protocol": {
            "task": "Hugging Face sequence classification with a newly initialized classification head",
            "validation": "Select best epoch by validation macro-F1 with early stopping",
            "test": "Official test split evaluated once after epoch selection",
            "controls": {"seed": seed, "epochs_max": epochs, "batch_size": batch_size,
                "max_length": max_length, "learning_rate": learning_rate,
                "weight_decay": weight_decay, "patience": patience, "device": str(device)},
            "limits": "A single seed is a pilot. Repeat with predeclared seeds and report their spread for research claims.",
        },
    }
    (out / "hf_classification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "hf_classification_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["row_id", "gold", *checkpoints])
        for index, gold in enumerate(test_labels):
            writer.writerow([index, gold, *(label_names[predictions[name][index]] for name in checkpoints)])
    return report
