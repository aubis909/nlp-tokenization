"""SentencePiece model training and consistent runtime adapters."""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm

from .data import write_lines
from .dct import BoundaryDCT
from .morphology import MorfessorAdapter
from .text import augment_with_transliteration, phone_proxy, tag_language

METHODS = ("baseline", "aligned", "morphology", "language_tagged", "phonetic", "contextual", "regularized")
LANGUAGE_TAGS = ["<EN>", "<ROM>", "<HI>", "<MIX>", "<OTHER>"]


def _preprocess(text: str, method: str, morph: MorfessorAdapter | None = None) -> str:
    if method == "morphology":
        return morph.segment(text) if morph else text
    if method == "language_tagged":
        return tag_language(text)
    if method == "phonetic":
        return phone_proxy(text)
    return text


def train_one(method: str, train_rows: list[str], out_dir: str | Path,
              vocab_size: int = 8000, model_type: str = "unigram") -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose from {METHODS}")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if method == "contextual":
        model = BoundaryDCT().fit(train_rows)
        model.save(out_dir / "dct.joblib")
        return {"method": method, "kind": "contextual_boundary_model", "weak_labels": True,
                "classifier_trained": model.model is not None,
                "fallback_reason": model.fallback_reason,
                "weak_label_counts": model.label_counts}
    if method == "regularized" and model_type != "unigram":
        raise ValueError("The regularized method requires SentencePiece Unigram; use --model-type unigram")

    morph = MorfessorAdapter().fit(train_rows) if method == "morphology" else None
    if morph:
        morph.save(out_dir / "morfessor.pkl")
    training_lines: list[str] = []
    for row in train_rows:
        if method == "aligned":
            training_lines.extend(augment_with_transliteration(row))
        else:
            training_lines.append(_preprocess(row, method, morph))
    text_path = out_dir / "spm_training.txt"
    write_lines(training_lines, text_path)
    model_prefix = str(out_dir / "tokenizer")
    symbols = LANGUAGE_TAGS if method == "language_tagged" else []
    spm.SentencePieceTrainer.train(
        input=str(text_path), model_prefix=model_prefix, vocab_size=vocab_size,
        model_type=model_type, character_coverage=0.9999, hard_vocab_limit=False,
        shuffle_input_sentence=False, num_threads=1, normalization_rule_name="identity",
        user_defined_symbols=symbols, unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    )
    metadata = {"method": method, "kind": "sentencepiece", "model_type": model_type,
                "vocab_size_requested": vocab_size, "vocab_size_actual": spm.SentencePieceProcessor(model_file=model_prefix + ".model").get_piece_size(),
                "language_tags": symbols, "phonetic_proxy": method == "phonetic",
                "morphology_adapter": "morfessor" if morph and morph.model is not None else "weak_suffix_rules" if morph else None}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    text_path.unlink(missing_ok=True)
    return metadata


class TokenizerAdapter:
    def __init__(self, model_dir: str | Path, method: str):
        self.model_dir, self.method = Path(model_dir), method
        if method == "contextual":
            self.contextual = BoundaryDCT.load(self.model_dir / "dct.joblib")
            self.processor = None; self.morph = None
        else:
            self.processor = spm.SentencePieceProcessor(model_file=str(self.model_dir / "tokenizer.model"))
            self.morph = MorfessorAdapter.load(self.model_dir / "morfessor.pkl") if method == "morphology" and (self.model_dir / "morfessor.pkl").exists() else None

    def encode(self, text: str, sample: bool = False) -> list[str]:
        if self.method == "contextual":
            return self.contextual.segment(text)
        prepared = _preprocess(text, self.method, self.morph)
        if self.method == "regularized" and sample:
            return self.processor.sample_encode_as_pieces(prepared, nbest_size=-1, alpha=0.1)
        return self.processor.encode(prepared, out_type=str)
