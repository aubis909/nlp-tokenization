"""Evaluate manually validated cross-script word pairs against each tokenizer variant."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .text import devanagari_to_itrans, script_of
from .tokenizers import METHODS, TokenizerAdapter

TAGS = {"<EN>", "<ROM>", "<HI>", "<MIX>", "<OTHER>"}


def _surface(piece: str) -> str:
    value = piece.replace("▁", "").strip().casefold()
    if value.upper() in TAGS:
        return ""
    return devanagari_to_itrans(value) if script_of(value) == "deva" else value


def evaluate_pairs(path: str | Path, artifacts: str | Path, out_dir: str | Path,
                   dev_column: str = "devanagari", roman_column: str = "roman",
                   methods: list[str] | None = None, bootstrap: int = 5000) -> dict:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    if not pairs or dev_column not in pairs[0] or roman_column not in pairs[0]:
        raise ValueError(f"Pair CSV needs '{dev_column}' and '{roman_column}' columns")
    if any(not row.get(dev_column) or not row.get(roman_column) for row in pairs):
        raise ValueError("Every alignment row must have both script forms")
    methods = methods or [m for m in METHODS if (Path(artifacts) / m).exists()]
    by_method, detail = {}, []
    for method in methods:
        adapter = TokenizerAdapter(Path(artifacts) / method, method)
        jaccards, exact = [], []
        for index, row in enumerate(pairs):
            deva, roman = row[dev_column].strip(), row[roman_column].strip()
            left = [_surface(p) for p in adapter.encode(deva)]
            right = [_surface(p) for p in adapter.encode(roman)]
            left = [p for p in left if p]; right = [p for p in right if p]
            aset, bset = set(left), set(right)
            jaccard = len(aset & bset) / len(aset | bset) if aset | bset else 0.0
            same = left == right
            jaccards.append(jaccard); exact.append(int(same))
            detail.append({"pair_id": row.get("pair_id", index), "method": method,
                           "devanagari": deva, "roman": roman,
                           "dev_tokens": " ".join(left), "roman_tokens": " ".join(right),
                           "token_set_jaccard": jaccard, "canonical_sequence_match": same})
        by_method[method] = {"mean_token_set_jaccard": float(np.mean(jaccards)),
                             "canonical_sequence_match_rate": float(np.mean(exact)),
                             "n_pairs": len(pairs)}
    paired = {}
    if "baseline" in methods:
        b = [r for r in detail if r["method"] == "baseline"]
        for method in methods:
            if method == "baseline":
                continue
            a = [r for r in detail if r["method"] == method]
            delta = np.array([x["token_set_jaccard"] - y["token_set_jaccard"] for x, y in zip(a, b)])
            rng = np.random.default_rng(17)
            draws = np.mean(rng.choice(delta, size=(bootstrap, len(delta)), replace=True), axis=1)
            p = float(wilcoxon(delta, zero_method="zsplit").pvalue) if np.any(delta) else 1.0
            paired[method] = {"mean_delta_vs_baseline": float(delta.mean()),
                              "bootstrap_95ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
                              "wilcoxon_p": p, "n_pairs": len(delta)}
    report = {"summary": by_method, "paired_vs_baseline": paired,
              "definition": "Surface-normalized subword-set Jaccard and ordered-sequence match after rendering Devanagari tokens to ITRANS. Human validation of the pair CSV is required."}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "pair_consistency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "pair_consistency.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)
    return report
