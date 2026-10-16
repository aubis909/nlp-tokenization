# Script-aware tokenization for Indic code-mixed NLP

This repository provides a reproducible experiment runner for Indic and Marathi-English code-mixed tokenization. Its recommended primary study is narrower: compare an original-tokenizer continued-pretraining control, ordinary SentencePiece vocabulary extension, and transliteration-view-augmented vocabulary extension on Marathi-English code-mixed text. Evaluate tokenization efficiency and a fine-tuned pretrained-transformer sentiment model on official MeSent splits. Other tokenizers, languages, and translation directions remain exploratory extensions.

The experiment tests a trade-off; it does not assume fewer tokens mean better language understanding. Tokenizer adaptation is an active research area, and combining known components is not automatically novel. Treat the methods below as candidate contributions to validate through a current literature review and controlled experiments, not as established novelty claims.

## What is implemented

| Method | Implementation | Main caveat |
|---|---|---|
| `baseline` | SentencePiece Unigram or BPE on the training corpus | Standard reference point |
| `aligned` | Adds ITRANS-rendered Devanagari views to SentencePiece training; inference text remains unchanged | Alignment is heuristic and can be lossy/ambiguous |
| `morphology` | Morfessor unsupervised segmentation for Devanagari where available; transparent suffix-rule fallback | Not a validated Hindi/Marathi morphological analyzer |
| `language_tagged` | Weak script/lexicon tags (`<HI>`, `<EN>`, `<ROM>`, etc.) guide subword learning | Latin-script language ID is only a small lexicon heuristic |
| `phonetic` | Cross-script grapheme/ITRANS phone proxy before SentencePiece | Not a production G2P system; no reliable round-trip guarantee |
| `contextual` | Character-boundary logistic model using local characters and neighboring words | Weak labels come from script transitions and suffix rules, not human annotations |
| `regularized` | SentencePiece Unigram sampling used as stochastic downstream-training augmentation | The tokenizer efficiency metrics use deterministic encoding for every arm; sampled token-count mean/std are reported separately |

All text normalization, data splits, settings, hashes, and output files are recorded. The weak CMI and Latin-language labels are diagnostics only. Use gold LID and benchmark labels for publication claims.

## 1. Open in VS Code and install

Install Python 3.10 or newer and VS Code with the Python extension. Open this repository folder in VS Code. In the integrated PowerShell terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks virtual-environment activation, select `.venv\Scripts\python.exe` with **Python: Select Interpreter** in VS Code, then run commands as `python -m indic_tokenizer_lab.cli ...` without activating it. Install PyTorch using the command recommended by the official PyTorch selector if you need a CUDA-specific build; CPU experiments work but train more slowly.

## 2. Start with the included toy corpus

This tiny file checks the end-to-end workflow only. It is not a benchmark and must not be used for research conclusions.

```powershell
python -m indic_tokenizer_lab.cli prepare --input data/examples/toy_hinglish.txt --out data/processed
python -m indic_tokenizer_lab.cli train --train data/processed/train.txt --out artifacts --vocab-size 256 --model-type unigram
python -m indic_tokenizer_lab.cli evaluate --test data/processed/test.txt --artifacts artifacts --out results
```

The default training command builds every method, including contextual segmentation. For quick iteration, choose arms explicitly by repeating `--method`:

```powershell
python -m indic_tokenizer_lab.cli train --method baseline --method aligned --method morphology --method language_tagged --method regularized --train data/processed/train.txt --out artifacts --vocab-size 4000
```

The regularized method requires Unigram (the default). Its tokenizer training recipe is the same as the baseline; its experimental treatment is sampled segmentation during downstream training. Artifacts go in `artifacts/<method>/`; held-out rows, metrics, paired tests, and the comparison plot go in `results/`.

## 3. Prepare a real corpus

Obtain the dataset from its official publisher and check its license, research-use conditions, and redistribution rules. This project deliberately does not download social-media corpora automatically. Keep source data out of Git. Supply UTF-8 `.txt` (one sentence per line), `.csv`, `.tsv`, or `.jsonl`. CSV/TSV/JSONL defaults to a `text` field.

For example, for a CSV with a `sentence` column:

```powershell
python -m indic_tokenizer_lab.cli prepare --input data/raw/corpus.csv --text-column sentence --out data/processed --test-fraction 0.2 --seed 17
```

`prepare` normalizes to NFC, replaces URLs and user handles with placeholders, lowercases, removes exact duplicate lines, and makes a deterministic split. The script records the source file hash in `data/processed/manifest.json`. For the paper, decide normalization before looking at test performance, freeze the split, and keep tokenizer training restricted to `train.txt`. If multiple corpora or languages are used, report them separately and retain official train/dev/test splits where the benchmark provides them.

## 4. Train and compare tokenizer variants

```powershell
python -m indic_tokenizer_lab.cli train --train data/processed/train.txt --out artifacts --vocab-size 8000 --model-type unigram
python -m indic_tokenizer_lab.cli evaluate --test data/processed/test.txt --artifacts artifacts --out results --bootstrap 5000
```

The result files are:

- `results/per_sentence.csv`: sentence-level counts, script stratum, weak CMI, fertility, bytes/token, and unknown-piece rate.
- `results/summary.json`: means, script breakdowns, generated Devanagari-to-ITRANS cross-script token-set consistency, paired bootstrap confidence intervals, and Wilcoxon tests against baseline.
- `results/tokenizer_comparison.png`: fertility comparison.

Fertility is payload subword pieces per whitespace/punctuation word. The method `language_tagged` reports both total token count (including tags) and payload count; comparisons use payload fertility. SentencePiece token-level perplexities are not compared across vocabularies. Control the vocabulary budget and training data across arms. Report normalized sequence length, token count, bytes/token, and actual inference latency on the same machine and batching setup if making efficiency claims.

All arms use deterministic encoding for primary intrinsic comparisons. For `regularized`, the CSV also reports the mean and standard deviation of token counts from ten sampled segmentations per sentence; these describe sampling variability and are not direct tokenizer-quality scores.

Cross-script consistency is computed only for Devanagari words paired with this project’s ITRANS transliteration and is a diagnostic, not a semantic equivalence score. Replace or supplement it with a human-checked transliteration lexicon that includes common informal spellings such as `kya`, `kyaa`, and `kyaaN` before drawing conclusions about real Hinglish.

To score your checked pair file (`pair_id,devanagari,roman`), run:

```powershell
python -m indic_tokenizer_lab.cli evaluate-pairs --input data/benchmark/transliteration_pairs.csv --artifacts artifacts --out results/pairs
```

This writes per-pair tokenizations, normalized token-set Jaccard, ordered sequence match rate, and paired statistics against the baseline. Annotate ambiguous words and retain the annotator instructions with the paper materials.

## 5. Transformer downstream experiment (recommended primary task)

The TF-IDF `classify` command in the next section is a quick screening baseline. For the main task, use `hf-classify`: it fine-tunes the same pretrained sequence-classification architecture independently for every checkpoint, selects the best epoch using validation macro-F1, then evaluates the held-out test file. It writes accuracy, macro-F1, per-class scores, token counts, training/inference speed, peak GPU memory, paired bootstrap confidence intervals, exact McNemar tests, and predictions. It saves each fine-tuned model under the output folder.

The repository includes L3Cube-MeSent code-mixed Marathi-English train, validation, and test CSVs under `data/benchmark/me_sent/`. The files use `tweet` and `sentiment`. Never use the test split to select epochs or tune settings.

Train ordinary and aligned SentencePiece tokenizers using only the adaptation-corpus training split:

```powershell
python -m indic_tokenizer_lab.cli train --method baseline --method aligned --train data/processed/mecorpus_5k/train.txt --out artifacts/mecorpus_5k --vocab-size 8000 --model-type unigram
```

Continue MLM pretraining from the same base checkpoint for three conditions. The first keeps the original Hugging Face tokenizer (`--max-new-tokens 0`), the second adds pieces from the ordinary SentencePiece baseline, and the third adds pieces from the transliteration-augmented tokenizer. Keep the corpus split and training budget identical:

```powershell
python -m indic_tokenizer_lab.cli adapt-mlm --base-model xlm-roberta-base --tokenizer-dir artifacts/mecorpus_5k/baseline --train data/processed/mecorpus_5k/train.txt --eval data/processed/mecorpus_5k/test.txt --out models/mecorpus_5k/original_continued --max-new-tokens 0 --max-steps 1000 --batch-size 8 --max-length 128 --seed 17
python -m indic_tokenizer_lab.cli adapt-mlm --base-model xlm-roberta-base --tokenizer-dir artifacts/mecorpus_5k/baseline --train data/processed/mecorpus_5k/train.txt --eval data/processed/mecorpus_5k/test.txt --out models/mecorpus_5k/spm_baseline --max-new-tokens 1000 --max-steps 1000 --batch-size 8 --max-length 128 --seed 17
python -m indic_tokenizer_lab.cli adapt-mlm --base-model xlm-roberta-base --tokenizer-dir artifacts/mecorpus_5k/aligned --train data/processed/mecorpus_5k/train.txt --eval data/processed/mecorpus_5k/test.txt --out models/mecorpus_5k/aligned --max-new-tokens 1000 --max-steps 1000 --batch-size 8 --max-length 128 --seed 17
```

Run the transformer classifier with identical labeled examples and settings:

```powershell
python -m indic_tokenizer_lab.cli hf-classify --train data/benchmark/me_sent/codemix-sent-train.csv --validation data/benchmark/me_sent/codemix-sent-val.csv --test data/benchmark/me_sent/codemix-sent-test.csv --text-column tweet --label-column sentiment --checkpoint baseline=models/mecorpus_5k/original_continued --checkpoint spm-baseline=models/mecorpus_5k/spm_baseline --checkpoint aligned=models/mecorpus_5k/aligned --out results/hf_mesent_seed17 --epochs 5 --batch-size 8 --max-length 128 --patience 2 --seed 17 --bootstrap 2000
```

`baseline` is the original Hugging Face tokenizer with continued MLM training and no new tokens; this controls for the extra MLM training applied to extension conditions. `spm-baseline` and `aligned` differ in the SentencePiece vocabulary source. If GPU memory is insufficient, try `--batch-size 4` and use it for every condition. CPU works but can be slow. Run multiple predeclared seeds with separate output paths before drawing research conclusions. Keep adaptation text disjoint from MeSent test rows and record dataset version, hashes, license, and split policy. Bootstrap intervals and p-values do not replace multiple seeds or a second domain. A gain in token efficiency with lower sentiment quality is a trade-off, not an overall win.

## 6. Evaluate on lightweight labeled NLP benchmarks

Convert the official training and test partitions into UTF-8 CSV files with `text,label` columns (or pass column names). Then run:

```powershell
python -m indic_tokenizer_lab.cli classify --train data/benchmark/train.csv --test data/benchmark/test.csv --artifacts artifacts --out results/classification
```

This uses a deliberately identical TF-IDF unigram/bigram plus balanced logistic-regression classifier per tokenizer. It reports accuracy, macro-F1, class-wise scores, predictions, and an exact paired McNemar test against the baseline. It is an inexpensive controlled downstream probe, not a replacement for benchmark-prescribed models or a pretrained-transformer experiment. Do not tune on the official test labels.

You can use this interface with permitted GLUECoS or COMI-LINGUA classification/LID splits when their data format and labels are mapped carefully. PHINC is a translation corpus; it cannot be scored by the classifier command.

## 7. Run and evaluate translation in both directions

The translation CSV interface handles each route independently, so the experiment can cover Hinglish↔English, Hinglish↔Hindi, Hinglish↔Marathi, Hindi↔English, Marathi↔English, and Hindi↔Marathi. Use one row per segment with `direction,source,reference,src_lang,tgt_lang`; `prompt` is optional for instruction/prompt-conditioned checkpoints. Source/target language codes are passed per row, which supports multiple directions in one run. A blank language code uses the model defaults. You can also pass `--src-lang`/`--tgt-lang` once for a single-direction file.

For multilingual checkpoints such as NLLB, standard code examples are `eng_Latn`, `hin_Deva`, and `mar_Deva` (confirm supported codes for the exact model). Example English-to-Hindi run:

```powershell
python -m indic_tokenizer_lab.cli translate --model facebook/nllb-200-distilled-600M --input data/mt/en_hi.csv --src-lang eng_Latn --tgt-lang hin_Deva --out results/translations/en_hi.csv
python -m indic_tokenizer_lab.cli score-directions --input results/translations/en_hi.csv --out results/translation_scores/en_hi.json
```

Reverse the codes for Hindi-to-English (`hin_Deva` → `eng_Latn`); use `mar_Deva` for Marathi. For a multi-direction CSV, put codes on each row and run the same command without global language flags. Predictions and per-direction language codes are saved with the source and reference.

Hinglish is not a standard NLLB language code. For Hinglish↔English/Hindi/Marathi, use a seq2seq checkpoint trained or fine-tuned for the required code-mixed direction and provide its prompt or per-row language codes if it expects them. NLLB can serve as an English↔Hindi/Marathi baseline, but should not be reported as a Hinglish generator unless it has been adapted and verified for that output style. This command runs generation with a supplied checkpoint; it does not fine-tune an MT model.

Score every direction separately. `score-directions` gives BLEU and chrF++ per direction plus an unweighted direction macro; it does not pool unlike language pairs into one corpus score.

To fine-tune a pretrained seq2seq model on parallel data while comparing a tokenizer arm, prepare separate training and development CSVs with `source,target` and optional `src_lang,tgt_lang,prompt` columns. Include both row directions in both splits for a shared bidirectional model, or train one checkpoint per direction. This extends the model’s original vocabulary with the selected method’s pieces, initializes new input/output embedding rows from old-token decompositions, and trains on the parallel set:

```powershell
python -m indic_tokenizer_lab.cli finetune-translation --base-model facebook/nllb-200-distilled-600M --tokenizer-dir artifacts/aligned --train data/mt/train.csv --eval data/mt/dev.csv --out models/nllb-aligned --lora --lora-targets q_proj,v_proj
```

Use `--vocab-extension-size 0` for a fine-tuned, unchanged-vocabulary baseline. Repeat with `artifacts/baseline`, `artifacts/morphology`, etc. under the same model, data, directions, and training budget. For Hinglish target style, the parallel targets must be genuinely code-mixed and the selected model must be trained to produce that style. Fine-tuning across all six routes requires parallel examples for every route; a model name alone cannot provide that coverage. Full fine-tuning can require substantial GPU memory; LoRA is optional and may need `--lora-targets` adjusted to the checkpoint architecture.

## 8. Score machine translation outputs directly

Prepare two UTF-8 text files with one aligned segment per line: references and system hypotheses. Use the exact same lines and detokenization for each system.

```powershell
python -m indic_tokenizer_lab.cli score-translation --references data/mt/references.txt --hypotheses data/mt/system.txt --out results/translation.json
```

This reports SacreBLEU BLEU and chrF++. It scores outputs; it does not train a translation system or call Google Translate. If Google Translate is included as an external baseline, record the API/product, date, language direction, request settings, and any fees, and keep it separate from tokenizer-controlled systems. COMET can be added when a suitable checkpoint and its license/compute are available; it is not installed by default because model downloads and GPU/CPU cost vary.

## 9. Extend a pretrained masked-language model

For the project’s model-adaptation stage, the code adds SentencePiece-derived pieces to a pretrained Hugging Face tokenizer while preserving old token IDs, resizes the masked-LM embedding table, initializes each new embedding from the mean of its old-token decomposition, and continues masked-language training:

```powershell
python -m indic_tokenizer_lab.cli adapt-mlm --base-model xlm-roberta-base --tokenizer-dir artifacts/aligned --train data/processed/train.txt --eval data/processed/test.txt --out models/xlmr-aligned --max-new-tokens 1000 --max-steps 1000 --batch-size 8 --max-length 128
```

Add `--lora` to train query/value LoRA adapters together with gradients only for newly added embedding rows. This can reduce the trainable transformer parameters, but still requires GPU memory for the base model and does not guarantee a quality improvement.

Run the same base model, corpus split, token budget, steps, batch size, sequence length, and seed for the baseline and each extension. Use a small publicly available model first to confirm memory and runtime. This script is for masked-language-model checkpoints such as XLM-R/mBERT, not arbitrary causal-LM checkpoints. It keeps the pretrained tokenizer and appends useful pieces; it does not replace the model’s tokenizer with a newly trained SentencePiece model. Its masked-LM validation loss is a within-setup diagnostic, not directly comparable perplexity across different tokenizations. Evaluate a task benchmark as the primary language-quality measure. Vocabulary resize alone does not train useful embeddings; continued training is essential.

## 10. Recommended experiment protocol

1. Choose one language/domain first (for example Hinglish Roman + Devanagari), one base model, and one official held-out benchmark. Add Marathi-English as a separately reported replication if resources permit.
2. Establish baseline tokenization on the untouched test split. Record source versions, corpus hashes, licenses, package versions, hardware, and seeds.
3. Compare the baseline and one method at a time, then planned ablations (for example aligned without synthetic romanization, morphology with/without Morfessor, tagged vs untagged, deterministic vs sampled Unigram).
4. For language quality, continue pretraining with identical settings and evaluate one common downstream task with its official metric. Report intrinsic token efficiency and quality together; do not assume one implies the other.
5. Report confidence intervals and paired tests, include per-script and code-mixing strata based on gold labels if available, correct for multiple comparisons, inspect error cases, and publish exact data/model provenance.
6. Before calling a contribution novel, search current ACL Anthology, arXiv, and relevant language-specific proceedings for tokenizer extension, transliteration-aware tokenization, morphology-aware tokenization, and code-mixed segmentation. The supplied project notes themselves warn that tokenizer adaptation and continued vocabulary training already exist.

Starting points for the literature review (read full papers and check work published after them): [Teaching Old Tokenizers New Words (EACL Findings 2026)](https://aclanthology.org/2026.findings-eacl.341/) studies continued BPE vocabulary extension and pruning; [IndicSuperTokenizer (2025)](https://arxiv.org/abs/2511.03237) reports language-specific pre-tokenization and a subword/superword design across Indic languages; [The Nanda Family (EACL 2026)](https://aclanthology.org/2026.eacl-long.288/) describes Hindi vocabulary extension plus continued pretraining with Romanized and code-mixed data; [Tokenization Matters (2025)](https://arxiv.org/abs/2504.16977) evaluates tokenizer choices on Indic NER, including Marathi. These works make “we adapted a tokenizer for an Indic language” too broad to carry a novelty claim by itself.

## 11. Project layout

```text
src/indic_tokenizer_lab/   preprocessing, algorithms, evaluation, CLI
data/examples/             tiny smoke-test corpus
data/raw/                  put licensed source corpora here (ignored by Git)
data/processed/             fixed split and provenance manifest (ignored by Git)
artifacts/                 tokenizer and contextual model artifacts (ignored by Git)
results/                   metrics and plots (ignored by Git)
models/                    optional Hugging Face continued-pretraining outputs
```

See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for preregisterable hypotheses, arm controls, a publication-quality minimum run, and a paper outline.

## 12. Troubleshooting

- **No GPU / CUDA out of memory:** reduce `--batch-size`, `--max-length`, or `--max-steps` for adaptation; keep settings identical across compared runs. CPU is supported.
- **`hf-classify` says a model folder has no `config.json`:** its `adapt-mlm` run did not finish saving (the output directory may still exist but be empty). Rerun that adaptation and wait for its JSON report before classifying. Successful outputs contain `config.json`, model weights, and tokenizer files.
- **SentencePiece vocabulary warning:** small corpora cannot support large vocabularies. Use a smaller `--vocab-size`, or train on the permitted larger corpus.
- **Morfessor not found / fails:** the method falls back to the included weak suffix rules. The artifact metadata indicates which path was used.
- **Bad Roman/Devanagari alignment:** inspect transliteration pairs and include a manually reviewed alignment subset before interpreting results. ITRANS is a scheme, not a guarantee of colloquial transliteration equivalence.
- **Benchmark columns differ:** map the official schema into `text` and `label` columns without changing the official partition or labels; preserve a mapping script in the project.

## 13. Dependencies and reproducibility

The package uses SentencePiece, Hugging Face Transformers/Tokenizers, Indic Transliteration, Morfessor, scikit-learn, SciPy, pandas/NumPy, Matplotlib, PyTorch, and SacreBLEU. `pip install -e .` installs them. Model checkpoints and corpora may have separate terms; review those before use or publication. Save `pip freeze` alongside results:

```powershell
python -m pip freeze > results/environment.txt
```
