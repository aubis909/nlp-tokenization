"""Intrinsic tokenizer metrics, strata, paired inference, and CSV/PNG result artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from .text import language_of, script_of, words, devanagari_to_itrans, transliteration_pair
from .tokenizers import METHODS, TokenizerAdapter

CONTROL = {"<EN>", "<ROM>", "<HI>", "<MIX>", "<OTHER>"}


def code_mixing_index(text: str) -> float:
    counts = Counter(language_of(w) for w in words(text) if re.search(r"\w", w, re.UNICODE))
    counts = Counter({k: v for k, v in counts.items() if k in {"EN", "ROM", "HI"}})
    total = sum(counts.values())
    return 0.0 if total == 0 else 100.0 * (1 - max(counts.values()) / total)


def cross_script_consistency(adapter: TokenizerAdapter, text: str) -> tuple[float, int]:
    """Token-set Jaccard for Devanagari words and their generated ITRANS variants.

    This diagnostic uses only generated transliteration pairs and should not be generalized to
    colloquial Latin spellings without a gold parallel transliteration lexicon.
    """
    scores = []
    def canonical(piece: str) -> str:
        value = piece.replace("▁", "").strip().casefold()
        if value.upper() in CONTROL:
            return ""
        return devanagari_to_itrans(value) if script_of(value) == "deva" else value
    for word in words(text):
        pair = transliteration_pair(word)
        if not pair:
            continue
        left = {canonical(p) for p in adapter.encode(pair[0]) if canonical(p) not in CONTROL and canonical(p)}
        right = {canonical(p) for p in adapter.encode(pair[1]) if canonical(p) not in CONTROL and canonical(p)}
        union = left | right
        scores.append(len(left & right) / len(union) if union else 0.0)
    return (float(np.mean(scores)), len(scores)) if scores else (float("nan"), 0)


def evaluate_one(adapter: TokenizerAdapter, method: str, rows: list[str]) -> list[dict]:
    results = []
    for index, text in enumerate(rows):
        # Compare tokenizer efficiency using deterministic encoding for every arm.
        # Unigram sampling is a stochastic training augmentation, so measure its
        # variability separately rather than changing the primary metrics here.
        pieces = adapter.encode(text, sample=False)
        consistency, pair_count = cross_script_consistency(adapter, text)
        payload = [piece for piece in pieces if piece.strip("▁") not in CONTROL]
        lexical = [w for w in words(text) if re.search(r"\w", w, re.UNICODE)]
        nbytes = len(text.encode("utf-8"))
        scripts = Counter(script_of(w) for w in lexical)
        scripts = Counter({k: v for k, v in scripts.items() if k in {"deva", "latin", "mixed"}})
        results.append({
            "row_id": index, "method": method, "text": text,
            "words": len(lexical), "tokens_total": len(pieces), "tokens_payload": len(payload),
            "fertility": len(payload) / max(1, len(lexical)),
            "sequence_length_per_word": len(payload) / max(1, len(lexical)),
            "bytes_per_token": nbytes / max(1, len(payload)),
            "unknown_rate": sum("<unk>" in p for p in payload) / max(1, len(payload)),
            "cmi_weak": code_mixing_index(text),
            "cross_script_consistency": consistency, "transliteration_pairs": pair_count,
            "dominant_script": scripts.most_common(1)[0][0] if scripts else "other",
        })
        if method == "regularized":
            sampled_counts = [len(adapter.encode(text, sample=True)) for _ in range(10)]
            results[-1]["sampled_token_count_mean"] = float(np.mean(sampled_counts))
            results[-1]["sampled_token_count_std"] = float(np.std(sampled_counts))
    return results


def _paired_stats(base: list[dict], adapted: list[dict], metric: str, bootstrap: int, seed: int):
    a = np.array([r[metric] for r in base], dtype=float)
    b = np.array([r[metric] for r in adapted], dtype=float)
    n = min(len(a), len(b)); delta = b[:n] - a[:n]
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(delta, size=(bootstrap, n), replace=True), axis=1)
    try:
        p = float(wilcoxon(delta, zero_method="zsplit").pvalue) if np.any(delta) else 1.0
    except ValueError:
        p = 1.0
    return {"mean_delta_adapted_minus_baseline": float(delta.mean()),
            "bootstrap_95ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "wilcoxon_p": p, "n_pairs": int(n)}


def run_evaluation(test_rows: list[str], artifacts_dir: str | Path, results_dir: str | Path,
                   methods: list[str] | None = None, bootstrap: int = 5000) -> dict:
    artifacts_dir, results_dir = Path(artifacts_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    methods = methods or [m for m in METHODS if (artifacts_dir / m).exists()]
    all_rows = []
    for method in methods:
        adapter = TokenizerAdapter(artifacts_dir / method, method)
        all_rows.extend(evaluate_one(adapter, method, test_rows))
    csv_path = results_dir / "per_sentence.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        # Some methods add method-specific metrics (e.g. Unigram sampling statistics).
        # Build a stable union so those rows remain exportable alongside other methods.
        fieldnames = list(dict.fromkeys(key for row in all_rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(all_rows)

    summaries = {}
    for method in methods:
        subset = [r for r in all_rows if r["method"] == method]
        summaries[method] = {metric: float(np.mean([r[metric] for r in subset]))
                             for metric in ("fertility", "sequence_length_per_word", "bytes_per_token", "unknown_rate", "cmi_weak")}
        summaries[method]["n_sentences"] = len(subset)
        consistency_rows = [r["cross_script_consistency"] for r in subset if not math.isnan(r["cross_script_consistency"])]
        summaries[method]["cross_script_consistency"] = float(np.mean(consistency_rows)) if consistency_rows else None
        summaries[method]["transliteration_pairs"] = int(sum(r["transliteration_pairs"] for r in subset))
        summaries[method]["by_script"] = {
            group: {"fertility": float(np.mean([r["fertility"] for r in subset if r["dominant_script"] == group]))}
            for group in ("deva", "latin", "mixed")
            if any(r["dominant_script"] == group for r in subset)
        }
    paired = {}
    if "baseline" in methods:
        base = [r for r in all_rows if r["method"] == "baseline"]
        for method in methods:
            if method == "baseline":
                continue
            adapted = [r for r in all_rows if r["method"] == method]
            paired[method] = {metric: _paired_stats(base, adapted, metric, bootstrap, 17)
                              for metric in ("fertility", "bytes_per_token")}
            base_pairs = [r for r in base if r["transliteration_pairs"]]
            adapted_pairs = [r for r in adapted if r["transliteration_pairs"]]
            if base_pairs and adapted_pairs:
                paired[method]["cross_script_consistency"] = _paired_stats(
                    base_pairs, adapted_pairs, "cross_script_consistency", bootstrap, 17)
    if "regularized" in summaries:
        reg = [r for r in all_rows if r["method"] == "regularized"]
        summaries["regularized"]["sampled_token_count_std_mean"] = float(np.mean([r["sampled_token_count_std"] for r in reg]))
    report = {"summary": summaries, "paired_vs_baseline": paired,
              "notes": ["Lower fertility indicates fewer payload tokens per whitespace/punctuation word.",
                        "Token-level perplexity must not be compared across different token vocabularies.",
                        "CMI, Latin-script language ID, and unknown rate are weak proxies; use benchmark gold labels where available.",
                        "Bootstrap intervals and Wilcoxon tests are paired by sentence; adjust for multiple comparisons in publication."]}
    (results_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(summaries, results_dir / "tokenizer_comparison.png")
    return report


def _plot(summary: dict, path: Path) -> None:
    names = list(summary)
    values = [summary[n]["fertility"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4c78a8" if n == "baseline" else "#f58518" for n in names]
    ax.bar(names, values, color=colors)
    ax.set_ylabel("Tokens per word (fertility)"); ax.set_title("Held-out code-mixed tokenization efficiency")
    ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
