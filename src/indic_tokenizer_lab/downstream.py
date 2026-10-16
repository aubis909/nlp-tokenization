"""Controlled shallow downstream evaluation using a shared TF-IDF + LogisticRegression head."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .tokenizers import METHODS, TokenizerAdapter
from .text import normalize


def _read_labeled(path: str | Path, text_column: str, label_column: str):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or text_column not in rows[0] or label_column not in rows[0]:
        raise ValueError(f"Expected CSV columns '{text_column}' and '{label_column}' in {path}")
    return [normalize(str(r[text_column])) for r in rows], [str(r[label_column]).strip() for r in rows]


def run_classification(train_path: str | Path, test_path: str | Path, artifacts_dir: str | Path,
                       out_dir: str | Path, methods: list[str] | None = None,
                       text_column: str = "text", label_column: str = "label") -> dict:
    train_text, y_train = _read_labeled(train_path, text_column, label_column)
    test_text, y_test = _read_labeled(test_path, text_column, label_column)
    labels = sorted(set(y_train) | set(y_test))
    if set(y_test) - set(y_train):
        raise ValueError("Every test label must occur in the training file")
    methods = methods or [m for m in METHODS if (Path(artifacts_dir) / m).exists()]
    results = {}
    predictions = {}
    augmentation_factors = {}
    for method in methods:
        adapter = TokenizerAdapter(Path(artifacts_dir) / method, method)
        factor = 3 if method == "regularized" else 1
        xtr = [" ".join(adapter.encode(s, sample=(method == "regularized")))
               for s in train_text for _ in range(factor)]
        y_method = [label for label in y_train for _ in range(factor)]
        xte = [" ".join(adapter.encode(s)) for s in test_text]
        vectorizer = TfidfVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None,
                                     lowercase=False, ngram_range=(1, 2), min_df=1,
                                     sublinear_tf=True, max_features=200_000)
        X_train = vectorizer.fit_transform(xtr); X_test = vectorizer.transform(xte)
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=17)
        classifier.fit(X_train, y_method)
        pred = classifier.predict(X_test); predictions[method] = pred.tolist()
        results[method] = {"accuracy": float(accuracy_score(y_test, pred)),
                           "macro_f1": float(f1_score(y_test, pred, average="macro", labels=labels, zero_division=0)),
                           "classification_report": classification_report(y_test, pred, labels=labels,
                               output_dict=True, zero_division=0),
                           "vocabulary_features": len(vectorizer.vocabulary_)}
        augmentation_factors[method] = factor
    paired = {}
    if "baseline" in predictions:
        base = np.asarray(predictions["baseline"]) == np.asarray(y_test)
        for method, pred in predictions.items():
            if method == "baseline":
                continue
            other = np.asarray(pred) == np.asarray(y_test)
            wins, losses = int(np.sum(other & ~base)), int(np.sum(base & ~other))
            pvalue = float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.0
            paired[method] = {"correct_only_adapted": wins, "correct_only_baseline": losses,
                              "mcnemar_exact_p": pvalue}
    report = {"metrics": results, "paired_vs_baseline": paired,
              "training_segmentation_augmentation_factor": augmentation_factors,
              "protocol": "Tokenizer-specific TF-IDF unigram/bigram features + balanced logistic regression; identical labeled split and classifier settings."}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "classification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "classification_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["row_id", "gold", *methods])
        for i, gold in enumerate(y_test): writer.writerow([i, gold, *(predictions[m][i] for m in methods)])
    return report


def score_translation(references: str | Path, hypotheses: str | Path, out: str | Path) -> dict:
    """Score newline-aligned translation outputs with SacreBLEU BLEU and chrF++."""
    import sacrebleu
    refs = Path(references).read_text(encoding="utf-8-sig").splitlines()
    hyps = Path(hypotheses).read_text(encoding="utf-8-sig").splitlines()
    if len(refs) != len(hyps):
        raise ValueError(f"Reference/hypothesis count mismatch: {len(refs)} vs {len(hyps)}")
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    report = {"n_segments": len(refs), "bleu": bleu.score, "bleu_signature": str(bleu.signature),
              "chrf_plus_plus": chrf.score, "chrf_signature": str(chrf.get_signature())}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
