"""Fine-tune a pretrained seq2seq translation model with one tokenizer adaptation arm."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .hf_adapt import _candidate_tokens


def finetune_translation(base_model: str, tokenizer_dir: str | Path, train_csv: str | Path,
                         eval_csv: str | Path, output_dir: str | Path,
                         source_column: str = "source", target_column: str = "target",
                         source_lang_column: str = "src_lang", target_lang_column: str = "tgt_lang",
                         prompt_column: str = "prompt", vocab_extension_size: int = 1000,
                         max_steps: int = 2000, batch_size: int = 8, max_length: int = 256,
                         learning_rate: float = 3e-5, seed: int = 17, lora: bool = False,
                         lora_targets: tuple[str, ...] = ("q_proj", "v_proj")) -> dict:
    """Extend a pretrained seq2seq tokenizer, initialize rows, and fine-tune on parallel examples."""
    from tokenizers import AddedToken
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              DataCollatorForSeq2Seq)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    candidates = _candidate_tokens(Path(tokenizer_dir) / "tokenizer.model", vocab_extension_size)
    decompositions = {t: tokenizer(t, add_special_tokens=False)["input_ids"] for t in candidates}
    tokens_added = tokenizer.add_tokens(
        [AddedToken(t, normalized=False, single_word=False, lstrip=False, rstrip=False)
         for t in candidates], special_tokens=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    old_size = model.get_input_embeddings().num_embeddings
    model.resize_token_embeddings(len(tokenizer))
    embedding = model.get_input_embeddings().weight.data
    with torch.no_grad():
        for token in candidates:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id < old_size:
                continue
            old_ids = [i for i in decompositions[token] if i < old_size]
            initial = embedding[torch.tensor(old_ids, device=embedding.device)].mean(dim=0) if old_ids else embedding[:old_size].mean(dim=0)
            embedding[token_id].copy_(initial)
    output_layer = model.get_output_embeddings()
    if output_layer is not None and output_layer.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        output_weight = output_layer.weight.data
        with torch.no_grad():
            for token in candidates:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id >= old_size:
                    old_ids = [i for i in decompositions[token] if i < old_size]
                    initial = output_weight[torch.tensor(old_ids, device=output_weight.device)].mean(dim=0) if old_ids else output_weight[:old_size].mean(dim=0)
                    output_weight[token_id].copy_(initial)

    hook = None
    if lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8,
                             lora_alpha=16, lora_dropout=0.05,
                             target_modules=list(lora_targets)))
        weight = model.get_input_embeddings().weight
        weight.requires_grad_(True)
        def freeze_old_rows(gradient):
            gradient[:old_size] = 0
            return gradient
        hook = weight.register_hook(freeze_old_rows)

    def load_parallel(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {source_column, target_column}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Parallel CSV requires columns {sorted(required)}")
            data = list(reader)
        if not data or any(not r.get(source_column) or not r.get(target_column) for r in data):
            raise ValueError(f"Parallel CSV {path} has no data or contains empty source/target cells")
        return data

    def encode_rows(rows):
        encoded = []
        for row in rows:
            src = (row.get(source_lang_column) or "").strip()
            tgt = (row.get(target_lang_column) or "").strip()
            if hasattr(tokenizer, "src_lang"):
                tokenizer.src_lang = src or None
            if hasattr(tokenizer, "tgt_lang"):
                tokenizer.tgt_lang = tgt or None
            prompt = row.get(prompt_column, "") or ""
            item = tokenizer(prompt + row[source_column], text_target=row[target_column],
                             truncation=True, max_length=max_length)
            encoded.append({"input_ids": item["input_ids"], "attention_mask": item["attention_mask"],
                            "labels": item["labels"]})
        return encoded

    train_rows, eval_rows = load_parallel(train_csv), load_parallel(eval_csv)
    train_data, eval_data = encode_rows(train_rows), encode_rows(eval_rows)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=learning_rate, weight_decay=0.0)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator)
    losses, step = [], 0
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); step += 1; losses.append(float(loss.detach().cpu()))
            if step >= max_steps:
                break
    model.eval(); validation_losses = []
    with torch.no_grad():
        for batch in DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=collator):
            batch = {k: v.to(device) for k, v in batch.items()}
            validation_losses.append(float(model(**batch).loss.detach().cpu()))
    if hook is not None:
        hook.remove(); model = model.merge_and_unload()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tokenizer.save_pretrained(out)
    report = {"base_model": base_model, "new_tokens_requested": len(candidates),
              "new_tokens_added": tokens_added,
              "final_vocab_size": len(tokenizer), "train_pairs": len(train_rows),
              "eval_pairs": len(eval_rows), "steps": step,
              "mean_training_loss": float(np.mean(losses)),
              "validation_token_loss": float(np.mean(validation_losses)),
              "lora": lora, "lora_targets": list(lora_targets) if lora else [],
              "seed": seed, "device": str(device),
              "note": "Select checkpoints by held-out generation BLEU/chrF per direction; token loss is not comparable across vocabularies."}
    (out / "translation_finetune_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
