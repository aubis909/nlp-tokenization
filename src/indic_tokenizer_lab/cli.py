"""Command line entry point for preparing corpora, training methods, and evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .data import read_text, sha256_file, split_corpus, write_lines
from .evaluate import run_evaluation
from .downstream import run_classification, score_translation
from .hf_downstream import run_hf_classification
from .hf_adapt import continue_pretraining
from .pair_eval import evaluate_pairs
from .translation import translate_csv, score_directional_csv
from .seq2seq_finetune import finetune_translation
from .tokenizers import METHODS, train_one


def _prepare(args):
    rows = read_text(args.input, args.text_column, args.limit, args.random_sample, args.seed)
    train, test = split_corpus(rows, args.test_fraction, args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_lines(train, out / "train.txt"); write_lines(test, out / "test.txt")
    manifest = {"source": str(Path(args.input).resolve()), "sha256": sha256_file(args.input),
                "text_column": args.text_column, "rows_after_deduplication": len(train) + len(test),
                "train_rows": len(train), "test_rows": len(test), "seed": args.seed,
                "test_fraction": args.test_fraction,
                "sampling": "reservoir_random" if args.random_sample else "first_rows",
                "normalization": "NFC; URL/user placeholders; lowercase"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _train(args):
    train = read_text(args.train)
    chosen = args.method or list(METHODS)
    for method in chosen:
        result = train_one(method, train, Path(args.out) / method, args.vocab_size, args.model_type)
        print(json.dumps(result, ensure_ascii=False))


def _evaluate(args):
    test = read_text(args.test)
    methods = args.method or None
    report = run_evaluation(test, args.artifacts, args.out, methods, args.bootstrap)
    print(json.dumps(report["summary"], indent=2))


def _classify(args):
    report = run_classification(args.train, args.test, args.artifacts, args.out, args.method,
                                args.text_column, args.label_column)
    print(json.dumps(report["metrics"], indent=2))


def _hf_classify(args):
    checkpoints = {}
    for item in args.checkpoint:
        if "=" not in item:
            raise ValueError("Each --checkpoint must be NAME=MODEL_OR_CHECKPOINT_PATH")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not path or name in checkpoints:
            raise ValueError(f"Invalid or duplicate checkpoint condition: {item}")
        checkpoints[name] = path
    report = run_hf_classification(args.train, args.validation, args.test, checkpoints,
        args.out, args.text_column, args.label_column, args.epochs, args.batch_size,
        args.max_length, args.learning_rate, args.weight_decay, args.patience,
        args.seed, args.device, args.bootstrap)
    print(json.dumps({"metrics": {name: {key: value for key, value in values.items()
        if key not in {"classification_report", "history"}}
        for name, values in report["metrics"].items()},
        "paired_vs_baseline": report["paired_vs_baseline"]}, indent=2))


def _translation(args):
    report = score_translation(args.references, args.hypotheses, args.out)
    print(json.dumps(report, indent=2))


def _adapt(args):
    report = continue_pretraining(args.base_model, args.tokenizer_dir, args.train, args.eval, args.out,
                                  args.max_new_tokens, args.max_steps, args.batch_size,
                                  args.max_length, args.learning_rate, args.seed, args.lora)
    print(json.dumps(report, indent=2))


def _pairs(args):
    report = evaluate_pairs(args.input, args.artifacts, args.out, args.dev_column,
                            args.roman_column, args.method, args.bootstrap)
    print(json.dumps(report["summary"], indent=2))


def _translate(args):
    report = translate_csv(args.model, args.input, args.out, args.source_column,
                           args.reference_column, args.direction_column, args.src_lang_column,
                           args.tgt_lang_column, args.prompt_column, args.src_lang, args.tgt_lang, args.batch_size,
                           args.max_length, args.max_new_tokens, args.num_beams, args.device)
    print(json.dumps(report, indent=2))


def _score_directions(args):
    report = score_directional_csv(args.input, args.out, args.direction_column,
                                   args.reference_column, args.prediction_column)
    print(json.dumps(report, indent=2))


def _finetune_translation(args):
    targets = tuple(part.strip() for part in args.lora_targets.split(",") if part.strip())
    report = finetune_translation(args.base_model, args.tokenizer_dir, args.train, args.eval, args.out,
                                  args.source_column, args.target_column, args.src_lang_column,
                                  args.tgt_lang_column, args.prompt_column, args.vocab_extension_size,
                                  args.max_steps, args.batch_size, args.max_length, args.learning_rate,
                                  args.seed, args.lora, targets)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indic-tokenizer-lab", description="Indic code-mixed tokenizer experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="normalize, deduplicate, and deterministically split a corpus")
    p.add_argument("--input", required=True); p.add_argument("--text-column", default="text")
    p.add_argument("--out", default="data/processed"); p.add_argument("--test-fraction", type=float, default=.2)
    p.add_argument("--seed", type=int, default=17); p.add_argument("--limit", type=int)
    p.add_argument("--random-sample", action="store_true",
                    help="reservoir-sample --limit rows across the full input instead of taking its first rows")
    p.set_defaults(func=_prepare)
    t = sub.add_parser("train", help="train tokenizer variants using train.txt only")
    t.add_argument("--train", default="data/processed/train.txt"); t.add_argument("--out", default="artifacts")
    t.add_argument("--method", choices=METHODS, action="append", help="repeat to select; default trains all")
    t.add_argument("--vocab-size", type=int, default=8000); t.add_argument("--model-type", choices=("unigram", "bpe"), default="unigram")
    t.set_defaults(func=_train)
    e = sub.add_parser("evaluate", help="run held-out intrinsic metrics and paired comparisons")
    e.add_argument("--test", default="data/processed/test.txt"); e.add_argument("--artifacts", default="artifacts")
    e.add_argument("--out", default="results"); e.add_argument("--method", choices=METHODS, action="append")
    e.add_argument("--bootstrap", type=int, default=5000)
    e.set_defaults(func=_evaluate)
    c = sub.add_parser("classify", help="evaluate tokenizer features on a labeled CSV benchmark")
    c.add_argument("--train", required=True, help="CSV with text and label columns")
    c.add_argument("--test", required=True, help="held-out labeled CSV")
    c.add_argument("--text-column", default="text"); c.add_argument("--label-column", default="label")
    c.add_argument("--artifacts", default="artifacts"); c.add_argument("--out", default="results")
    c.add_argument("--method", choices=METHODS, action="append")
    c.set_defaults(func=_classify)
    hc = sub.add_parser("hf-classify", help="fine-tune and compare pretrained transformer classifiers")
    hc.add_argument("--train", required=True, help="labeled training CSV")
    hc.add_argument("--validation", required=True, help="validation CSV used for early stopping")
    hc.add_argument("--test", required=True, help="official held-out test CSV; evaluated after selection")
    hc.add_argument("--text-column", default="tweet"); hc.add_argument("--label-column", default="sentiment")
    hc.add_argument("--checkpoint", action="append", required=True,
                    help="repeat NAME=MODEL_PATH; include baseline=BASE_MODEL and adapted conditions")
    hc.add_argument("--out", default="results/hf_classification")
    hc.add_argument("--epochs", type=int, default=5); hc.add_argument("--batch-size", type=int, default=8)
    hc.add_argument("--max-length", type=int, default=128); hc.add_argument("--learning-rate", type=float, default=2e-5)
    hc.add_argument("--weight-decay", type=float, default=0.01); hc.add_argument("--patience", type=int, default=2)
    hc.add_argument("--seed", type=int, default=17); hc.add_argument("--device", default="auto")
    hc.add_argument("--bootstrap", type=int, default=2000,
                    help="paired bootstrap replicates for metric differences vs baseline")
    hc.set_defaults(func=_hf_classify)
    tr = sub.add_parser("score-translation", help="score newline-aligned translation outputs with BLEU and chrF++")
    tr.add_argument("--references", required=True); tr.add_argument("--hypotheses", required=True)
    tr.add_argument("--out", default="results/translation.json"); tr.set_defaults(func=_translation)
    a = sub.add_parser("adapt-mlm", help="extend a Hugging Face MLM vocabulary and continue masked-LM training")
    a.add_argument("--base-model", required=True, help="Hugging Face model ID or local checkpoint")
    a.add_argument("--tokenizer-dir", required=True, help="trained SentencePiece method directory")
    a.add_argument("--train", required=True); a.add_argument("--eval", required=True)
    a.add_argument("--out", required=True); a.add_argument("--max-new-tokens", type=int, default=1000,
                    help="maximum added pieces; set 0 to continue MLM training with the original tokenizer")
    a.add_argument("--max-steps", type=int, default=1000); a.add_argument("--batch-size", type=int, default=8)
    a.add_argument("--max-length", type=int, default=128); a.add_argument("--learning-rate", type=float, default=5e-5)
    a.add_argument("--lora", action="store_true", help="train query/value LoRA adapters plus only newly added embedding rows")
    a.add_argument("--seed", type=int, default=17); a.set_defaults(func=_adapt)
    pe = sub.add_parser("evaluate-pairs", help="measure token consistency on a human-checked pair CSV")
    pe.add_argument("--input", required=True, help="CSV with one manually verified transliteration pair per row")
    pe.add_argument("--dev-column", default="devanagari"); pe.add_argument("--roman-column", default="roman")
    pe.add_argument("--artifacts", default="artifacts"); pe.add_argument("--out", default="results/pairs")
    pe.add_argument("--method", choices=METHODS, action="append"); pe.add_argument("--bootstrap", type=int, default=5000)
    pe.set_defaults(func=_pairs)
    trn = sub.add_parser("translate", help="generate with a Hugging Face seq2seq model for one or many directions")
    trn.add_argument("--model", required=True, help="model ID or local seq2seq checkpoint")
    trn.add_argument("--input", required=True, help="CSV with source and optional direction/src_lang/tgt_lang/reference columns")
    trn.add_argument("--out", required=True); trn.add_argument("--source-column", default="source")
    trn.add_argument("--reference-column", default="reference"); trn.add_argument("--direction-column", default="direction")
    trn.add_argument("--src-lang-column", default="src_lang"); trn.add_argument("--tgt-lang-column", default="tgt_lang")
    trn.add_argument("--prompt-column", default="prompt", help="optional per-row prefix for prompt-conditioned code-mix models")
    trn.add_argument("--src-lang", help="default source code, e.g. eng_Latn for NLLB")
    trn.add_argument("--tgt-lang", help="default target code, e.g. hin_Deva for NLLB")
    trn.add_argument("--batch-size", type=int, default=8); trn.add_argument("--max-length", type=int, default=512)
    trn.add_argument("--max-new-tokens", type=int, default=128); trn.add_argument("--num-beams", type=int, default=4)
    trn.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device name")
    trn.set_defaults(func=_translate)
    sd = sub.add_parser("score-directions", help="score each direction separately from translated CSV output")
    sd.add_argument("--input", required=True); sd.add_argument("--out", default="results/directional_translation.json")
    sd.add_argument("--direction-column", default="direction"); sd.add_argument("--reference-column", default="reference")
    sd.add_argument("--prediction-column", default="prediction"); sd.set_defaults(func=_score_directions)
    ft = sub.add_parser("finetune-translation", help="adapt tokenizer embeddings and fine-tune a pretrained seq2seq translation model")
    ft.add_argument("--base-model", required=True); ft.add_argument("--tokenizer-dir", required=True)
    ft.add_argument("--train", required=True, help="parallel CSV with source and target columns")
    ft.add_argument("--eval", required=True); ft.add_argument("--out", required=True)
    ft.add_argument("--source-column", default="source"); ft.add_argument("--target-column", default="target")
    ft.add_argument("--src-lang-column", default="src_lang"); ft.add_argument("--tgt-lang-column", default="tgt_lang")
    ft.add_argument("--prompt-column", default="prompt"); ft.add_argument("--vocab-extension-size", type=int, default=1000)
    ft.add_argument("--max-steps", type=int, default=2000); ft.add_argument("--batch-size", type=int, default=8)
    ft.add_argument("--max-length", type=int, default=256); ft.add_argument("--learning-rate", type=float, default=3e-5)
    ft.add_argument("--lora", action="store_true"); ft.add_argument("--lora-targets", default="q_proj,v_proj")
    ft.add_argument("--seed", type=int, default=17); ft.set_defaults(func=_finetune_translation)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
