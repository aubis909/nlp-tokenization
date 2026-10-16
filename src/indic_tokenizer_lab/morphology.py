"""Lightweight morphology constraints and an optional Morfessor adapter."""

from __future__ import annotations

import re
from pathlib import Path

from .text import devanagari_to_itrans, script_of, words

# Transparent weak suffix inventory; corpus/language specific and not a gold analyzer.
HI_SUFFIXES = ("यों", "ियों", "ाओं", "ों", "ें", "ने", "ना", "ती", "ता", "ते", "कर", "गी", "गा", "के", "की", "का", "में", "से", "को")
MR_SUFFIXES = ("ाच्या", "ाची", "ाचे", "ातून", "ामध्ये", "ाने", "ाला", "ात", "ानी", "णे", "तो", "ते", "ली", "ला", "ची", "चे")


def suffix_segments(word: str) -> list[str]:
    """Split one Devanagari word at the longest known suffix, retaining both pieces."""
    if script_of(word) != "deva":
        return [word]
    for suffix in sorted(set(HI_SUFFIXES + MR_SUFFIXES), key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return [word[:-len(suffix)], suffix]
    return [word]


def weak_morph_segment(text: str) -> str:
    pieces = []
    for token in words(text):
        parts = suffix_segments(token)
        pieces.extend(parts if len(parts) > 1 else [token])
    return " ".join(pieces)


class MorfessorAdapter:
    """Unsupervised Morfessor model; falls back to the transparent suffix heuristic."""

    def __init__(self):
        self.model = None

    def fit(self, sentences: list[str]) -> "MorfessorAdapter":
        try:
            import morfessor
        except ImportError:
            return self
        model = morfessor.BaselineModel()
        counts: dict[str, int] = {}
        for sentence in sentences:
            for token in words(sentence):
                if script_of(token) == "deva":
                    counts[token] = counts.get(token, 0) + 1
        model.load_data([(count, token) for token, count in counts.items()])
        model.train_batch()
        self.model = model
        return self

    def segment(self, text: str) -> str:
        if self.model is None:
            return weak_morph_segment(text)
        output = []
        for token in words(text):
            if script_of(token) == "deva":
                try:
                    output.extend(self.model.viterbi_segment(token)[0])
                except (AttributeError, ValueError):
                    output.extend(suffix_segments(token))
            else:
                output.append(token)
        return " ".join(output)

    def save(self, path: str | Path) -> None:
        import pickle
        with Path(path).open("wb") as stream:
            pickle.dump(self.model, stream)

    @classmethod
    def load(cls, path: str | Path) -> "MorfessorAdapter":
        import pickle
        obj = cls()
        with Path(path).open("rb") as stream:
            obj.model = pickle.load(stream)
        return obj
