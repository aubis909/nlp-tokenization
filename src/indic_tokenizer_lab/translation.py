"""Direction-aware seq2seq inference and per-direction BLEU/chrF evaluation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def translate_csv(model_name: str, input_path: str | Path, output_path: str | Path,
                  source_column: str = "source", reference_column: str = "reference",
                  direction_column: str = "direction", source_lang_column: str = "src_lang",
                  target_lang_column: str = "tgt_lang", prompt_column: str = "prompt",
                  source_lang: str | None = None,
                  target_lang: str | None = None, batch_size: int = 8, max_length: int = 512,
                  max_new_tokens: int = 128, num_beams: int = 4, device: str = "auto") -> dict:
    """Generate translations grouped by direction so multilingual BOS codes are correct."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    input_path, output_path = Path(input_path), Path(output_path)
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or source_column not in reader.fieldnames:
            raise ValueError(f"Input CSV must contain a '{source_column}' column")
        rows = list(reader)
    if not rows:
        raise ValueError("Translation input CSV has no rows")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    chosen_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model.to(chosen_device); model.eval()

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        src = (row.get(source_lang_column) or source_lang or "").strip()
        tgt = (row.get(target_lang_column) or target_lang or "").strip()
        groups[(src, tgt)].append(i)
    predictions = [""] * len(rows)
    code_map = getattr(tokenizer, "lang_code_to_id", {})
    for (src, tgt), indexes in groups.items():
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src or None
        forced_bos = None
        if tgt:
            if tgt in code_map:
                forced_bos = code_map[tgt]
            else:
                token_id = tokenizer.convert_tokens_to_ids(tgt)
                if token_id is None or token_id == tokenizer.unk_token_id:
                    raise ValueError(f"Target language code '{tgt}' is not in tokenizer {model_name}")
                forced_bos = token_id
        for start in range(0, len(indexes), batch_size):
            batch_indexes = indexes[start:start + batch_size]
            texts = [(rows[i].get(prompt_column, "") or "") + rows[i][source_column]
                     for i in batch_indexes]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                               max_length=max_length).to(chosen_device)
            kwargs = {"max_new_tokens": max_new_tokens, "num_beams": num_beams}
            if forced_bos is not None:
                kwargs["forced_bos_token_id"] = forced_bos
            with torch.inference_mode():
                generated = model.generate(**inputs, **kwargs)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for index, translation in zip(batch_indexes, decoded):
                predictions[index] = translation.strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["direction", "source", "reference", "prediction", "src_lang", "tgt_lang"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"direction": row.get(direction_column, "") or "unspecified",
                             "source": row.get(source_column, ""),
                             "reference": row.get(reference_column, ""),
                             "prediction": predictions[i],
                             "src_lang": (row.get(source_lang_column) or source_lang or ""),
                             "tgt_lang": (row.get(target_lang_column) or target_lang or "")})
    report = {"model": model_name, "input_rows": len(rows), "output_csv": str(output_path.resolve()),
              "directions": {f"{src or 'model-default'}->{tgt or 'model-default'}": len(ids)
                             for (src, tgt), ids in groups.items()}, "device": chosen_device,
              "generation": {"batch_size": batch_size, "max_length": max_length,
                             "max_new_tokens": max_new_tokens, "num_beams": num_beams}}
    output_path.with_suffix(".metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def score_directional_csv(input_path: str | Path, out_path: str | Path,
                          direction_column: str = "direction", reference_column: str = "reference",
                          prediction_column: str = "prediction") -> dict:
    """Score each translation direction separately and include an unweighted direction macro mean."""
    import sacrebleu
    with Path(input_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {direction_column, reference_column, prediction_column}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"Translation CSV requires columns {sorted(needed)}")
        rows = list(reader)
    by_direction: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row[reference_column].strip():
            by_direction[row[direction_column] or "unspecified"].append(row)
    if not by_direction:
        raise ValueError("No nonempty references were found")
    scores = {}
    for direction, group in sorted(by_direction.items()):
        refs = [r[reference_column] for r in group]
        hyps = [r[prediction_column] for r in group]
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
        scores[direction] = {"n_segments": len(group), "bleu": bleu.score,
                             "bleu_signature": str(bleu.signature), "chrf_plus_plus": chrf.score,
                             "chrf_signature": str(chrf.get_signature())}
    macro = {metric: sum(v[metric] for v in scores.values()) / len(scores)
             for metric in ("bleu", "chrf_plus_plus")}
    report = {"per_direction": scores, "unweighted_direction_macro": macro,
              "note": "Do not pool directions: report the separate scores and use the macro only as a supplementary summary."}
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
