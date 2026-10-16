"""Weakly supervised dynamic contextual segmentation (research prototype).

The boundary classifier conditions a split decision on characters plus neighboring words. Its
training labels come from transparent suffix/script-transition heuristics, not human annotation.
This makes it reproducible, but DCT numbers must be reported as weak-supervision results.
"""

from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

from .morphology import suffix_segments
from .text import script_of, words


def _char_features(word: str, index: int, previous: str, following: str) -> dict[str, str]:
    return {
        "left1": word[index - 1:index], "right1": word[index:index + 1],
        "left2": word[max(0, index - 2):index], "right2": word[index:index + 2],
        "pair": word[index - 1:index + 1], "script": script_of(word),
        "prev_script": script_of(previous), "next_script": script_of(following),
        "prev_word": previous[-4:].casefold(), "next_word": following[:4].casefold(),
        "is_transition": str(script_of(word[:index]) != script_of(word[index:])),
        "at_suffix_boundary": str(any(word.endswith(s) and len(word) - len(s) == index
                                       for s in ("ना", "ने", "ता", "ती", "ते", "ला", "ची", "चे", "में", "से"))),
    }


class BoundaryDCT:
    def __init__(self):
        self.model = None
        self.fallback_reason: str | None = None
        self.label_counts = {0: 0, 1: 0}

    def fit(self, sentences: list[str]) -> "BoundaryDCT":
        features, labels = [], []
        for sentence in sentences:
            ws = words(sentence)
            lexical = [w for w in ws if re.search(r"\w", w, re.UNICODE)]
            for wi, word in enumerate(lexical):
                if len(word) < 2:
                    continue
                prev = lexical[wi - 1] if wi else "<BOS>"
                nxt = lexical[wi + 1] if wi + 1 < len(lexical) else "<EOS>"
                weak = set()
                parts = suffix_segments(word)
                if len(parts) > 1:
                    weak.add(len(parts[0]))
                for pos in range(1, len(word)):
                    # Weak labels preserve script transitions and frequent suffix boundaries.
                    label = int(pos in weak or script_of(word[:pos]) != script_of(word[pos:]))
                    features.append(_char_features(word, pos, prev, nxt)); labels.append(label)
        self.label_counts = {0: labels.count(0), 1: labels.count(1)}
        # A tiny or script-homogeneous corpus can produce only one weak-label class.
        # Do not invent training labels: preserve the transparent weak rules as a
        # deterministic fallback and record why the learned classifier was skipped.
        if not features or len(set(labels)) < 2:
            self.fallback_reason = (
                "no_candidate_boundaries" if not features
                else "weak_labels_contain_only_one_class"
            )
            self.model = None
            return self
        self.fallback_reason = None
        self.model = make_pipeline(
            DictVectorizer(sparse=True),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=17),
        )
        self.model.fit(features, labels)
        return self

    def segment(self, sentence: str, threshold: float = 0.55) -> list[str]:
        if self.model is None and self.fallback_reason is None:
            raise RuntimeError("DCT model has not been trained or loaded")
        lexical = words(sentence)
        content = [w for w in lexical if re.search(r"\w", w, re.UNICODE)]
        out, wi = [], 0
        for word in lexical:
            if not re.search(r"\w", word, re.UNICODE):
                continue
            prev = content[wi - 1] if wi else "<BOS>"
            nxt = content[wi + 1] if wi + 1 < len(content) else "<EOS>"
            wi += 1
            if len(word) < 2:
                out.append(word); continue
            if self.model is None:
                # Same auditable suffix/script-transition weak rules used to create
                # the labels for the learned model, used when a classifier cannot
                # be estimated from the available corpus.
                weak = set()
                parts = suffix_segments(word)
                if len(parts) > 1:
                    weak.add(len(parts[0]))
                weak.update(p for p in range(1, len(word))
                            if script_of(word[:p]) != script_of(word[p:]))
                starts, last = [0], 0
                starts.extend(sorted(weak))
                starts.append(len(word))
                for end in starts[1:]:
                    if end > last:
                        out.append(word[last:end]); last = end
                continue
            feats = [_char_features(word, p, prev, nxt) for p in range(1, len(word))]
            probs = self.model.predict_proba(feats)[:, 1]
            starts, last = [0], 0
            for offset, prob in enumerate(probs, start=1):
                if prob >= threshold:
                    starts.append(offset)
            starts.append(len(word))
            for end in starts[1:]:
                out.append(word[last:end]); last = end
        return out

    def save(self, path: str | Path) -> None:
        joblib.dump({"model": self.model, "fallback_reason": self.fallback_reason,
                     "label_counts": self.label_counts}, path)

    @classmethod
    def load(cls, path: str | Path) -> "BoundaryDCT":
        saved = joblib.load(path)
        obj = cls()
        # Accept artifacts written by earlier versions, which stored the estimator directly.
        if isinstance(saved, dict) and "model" in saved:
            obj.model = saved["model"]
            obj.fallback_reason = saved.get("fallback_reason")
            obj.label_counts = saved.get("label_counts", {0: 0, 1: 0})
        else:
            obj.model = saved
        return obj
