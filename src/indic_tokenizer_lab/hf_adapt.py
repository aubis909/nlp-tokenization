"""Extend a pretrained Hugging Face MLM vocabulary and continue masked-language training.

Existing token IDs are preserved. New rows start from the mean of their old-token decomposition
and are subsequently trained. This is a vocabulary-extension baseline, not a replacement tokenizer.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _candidate_tokens(spm_model: str | Path, limit: int,
                      excluded_tokens: set[str] | None = None) -> list[str]:
    if limit <= 0:
        return []
    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=str(spm_model))
    found, seen = [], set(excluded_tokens or ())
    for token_id in range(4, processor.get_piece_size()):
        piece = processor.id_to_piece(token_id)
        if piece.startswith("<") or piece.startswith("["):
            continue
        rendered = processor.decode_pieces([piece]).strip().replace("▁", "")
        if len(rendered) < 2 or rendered in seen or not any(ch.isalpha() for ch in rendered):
            continue
        seen.add(rendered); found.append(rendered)
        if len(found) >= limit:
            break
    return found


def _read_rows(path: str | Path) -> list[str]:
    from .data import read_text
    return read_text(path)


def continue_pretraining(base_model: str, tokenizer_dir: str | Path, train_path: str | Path,
                         eval_path: str | Path, output_dir: str | Path, max_new_tokens: int = 1000,
                         max_steps: int = 1000, batch_size: int = 8, max_length: int = 128,
                         learning_rate: float = 5e-5, seed: int = 17, lora: bool = False) -> dict:
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

    if max_new_tokens < 0 or max_steps < 1 or batch_size < 1 or max_length < 2:
        raise ValueError("max_new_tokens must be >= 0; max_steps/batch_size must be >= 1; max_length must be >= 2")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    sp_model = Path(tokenizer_dir) / "tokenizer.model"
    if not sp_model.exists():
        raise ValueError("Model extension needs a SentencePiece tokenizer artifact, not contextual DCT")
    # Skip pieces already present in the base tokenizer so a requested limit
    # means the same number of genuinely new vocabulary rows across methods.
    existing_tokens = set(tokenizer.get_vocab())
    candidates = _candidate_tokens(sp_model, max_new_tokens, existing_tokens)
    # Record how each candidate was encoded before extension for informed row initialization.
    decompositions = {token: tokenizer(token, add_special_tokens=False)["input_ids"] for token in candidates}
    from tokenizers import AddedToken
    added = [AddedToken(t, normalized=False, single_word=False, lstrip=False, rstrip=False) for t in candidates]
    n_added = tokenizer.add_tokens(added, special_tokens=False)
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    old_size = model.get_input_embeddings().num_embeddings
    model.resize_token_embeddings(len(tokenizer))
    emb = model.get_input_embeddings().weight.data
    with torch.no_grad():
        for token in candidates:
            tid = tokenizer.convert_tokens_to_ids(token)
            if tid < old_size:
                continue
            old_ids = [i for i in decompositions[token] if i < old_size]
            if old_ids:
                emb[tid].copy_(emb[torch.tensor(old_ids, device=emb.device)].mean(dim=0))
            else:
                emb[tid].copy_(emb[:old_size].mean(dim=0))

    if lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type=TaskType.MASKED_LM, r=8,
                             lora_alpha=16, lora_dropout=0.05, target_modules=["query", "value"]))
        # Train only the appended input-embedding rows alongside LoRA weights.
        embedding_weight = model.get_input_embeddings().weight
        embedding_weight.requires_grad_(True)
        def freeze_old_embedding_rows(gradient):
            gradient[:old_size] = 0
            return gradient
        embedding_hook = embedding_weight.register_hook(freeze_old_embedding_rows)
    else:
        embedding_hook = None

    # The added-token HF tokenizer is used for model inputs to preserve base-model compatibility.
    train_rows, eval_rows = _read_rows(train_path), _read_rows(eval_path)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    def encode(rows):
        items = []
        for sentence in rows:
            encoded = tokenizer(sentence, truncation=True, max_length=max_length, add_special_tokens=True)
            if len(encoded["input_ids"]) >= 2:
                items.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})
        return items

    train_data, eval_data = encode(train_rows), encode(eval_rows)
    if not train_data or not eval_data:
        raise ValueError("Need nonempty train and evaluation text for continued pretraining")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=learning_rate, weight_decay=0.0)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator)
    losses, step, epoch = [], 0, 0
    while step < max_steps:
        epoch += 1
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); step += 1; losses.append(float(loss.detach().cpu()))
            if step >= max_steps:
                break

    model.eval(); val_losses = []
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=collator)
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            val_losses.append(float(model(**batch).loss.detach().cpu()))
    validation_loss = float(np.mean(val_losses))
    if embedding_hook is not None:
        embedding_hook.remove()
        model = model.merge_and_unload()
    model.save_pretrained(out); tokenizer.save_pretrained(out)
    report = {"base_model": base_model, "new_tokens_added": n_added, "requested_new_tokens": len(candidates),
              "old_vocab_size": old_size, "final_vocab_size": len(tokenizer), "steps": step,
              "mean_training_masked_lm_loss": float(np.mean(losses)),
              "validation_masked_lm_loss": validation_loss,
              "validation_masked_token_perplexity_proxy": math.exp(min(30.0, validation_loss)),
              "device": str(device), "seed": seed, "lora": lora,
              "interpretation": "Masked-LM validation loss is a within-tokenizer diagnostic, not a directly comparable text-level perplexity across tokenizers."}
    (out / "adaptation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
