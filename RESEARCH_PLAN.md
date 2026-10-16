# Research plan and publication decision

## Proposed study

**Working title:** Cross-script tokenizer adaptation for Marathi-English code-mixed sentiment: consistency, quality, and efficiency trade-offs

**Primary research question:** Under matched continued-pretraining and downstream fine-tuning settings, does adding SentencePiece vocabulary learned from transliteration-view-augmented Marathi-English code-mixed text improve Roman/Devanagari tokenization efficiency and MeSent sentiment quality over (1) continued pretraining with the original XLM-R tokenizer and (2) continued pretraining after ordinary SentencePiece vocabulary extension?

This is a controlled empirical study, not a novelty guarantee. The current `aligned` method adds generated ITRANS-rendered views during SentencePiece training; it does not force two transliterations to share token IDs and does not establish that this idea is unprecedented. MeSent provides an official Marathi-English code-mixed sentiment train/validation/test benchmark, but that dataset alone cannot support claims about Hindi, all transliteration styles, translation, or other domains. Verify the adaptation corpus is disjoint from benchmark test data before experiments.

## Hypotheses to preregister

- **H1 (primary):** Transliteration-view augmentation increases token-set consistency on a manually checked, held-out Roman/Devanagari word-pair set over ordinary SentencePiece at a matched vocabulary budget.
- **H2:** Any reduction in fertility or sequence length remains when held-out text is stratified by writing system and code-mixing level, rather than being driven by one high-frequency subset.
- **H3:** Lower token count alone does not guarantee improved MeSent macro-F1. Sentiment quality may be unchanged or worse.
- **Exploratory:** Morphology constraints, language/script tags, sampled Unigram segmentations, and contextual boundary prediction have distinct trade-offs under equal data and compute budgets.

Do not change hypotheses after seeing held-out scores. If the exploratory methods are included, correct for multiple comparisons or clearly label them exploratory.

## Experimental arms

| Arm | Implemented method                      | Required control                                                                                                    |
| --- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| A0  | Original XLM-R tokenizer, continued MLM | `adapt-mlm --max-new-tokens 0`; controls for extra MLM training                                                     |
| A1  | Ordinary SentencePiece vocabulary extension | Same base checkpoint, source corpus, MLM steps, and added-token limit                                             |
| A2  | Transliteration-view SentencePiece extension | Same settings as A1; generated views are the treatment; report transliteration limitations                      |
| A3+ | Morphology, tags, phonetic proxy, contextual boundaries, and sampling | Exploratory; wait until the primary three-arm comparison is stable                              |

Use one arm per ablation so the effect of each design choice can be attributed. Match the pretrained base checkpoint, added-token count (report actual counts), corpus split, MLM steps, supervised data, optimizer, sequence limit, seed, and evaluation protocol as closely as possible. The CLI `hf-classify` fine-tunes the same sequence-classification model for every arm, selects epochs on validation macro-F1, evaluates test once, and reports token counts, speed, memory, paired bootstrap intervals, and McNemar tests. It saves result checkpoints. The TF-IDF `classify` command remains a screening baseline only. Repeat seeds and replicate on a second domain before a paper claim.

## Data and split policy

The primary run uses Marathi-English code-mixed text and MeSent sentiment. Hindi, Hinglish-to-English, and translation directions are future extensions unless suitable parallel or labeled benchmarks are available. Report every route separately; do not assume one system supports all routes. Most multilingual MT checkpoints do not have a Hinglish target code, so that style requires code-mixed parallel training/evaluation and a compatible checkpoint. Potential resources include L3Cube-HingCorpus, COMI-LINGUA, GLUECoS, PHINC, and Marathi-English resources such as MeCorpus/MyBoli. Obtain each dataset from its official source and verify access, privacy, and redistribution terms. Preserve official benchmark partitions. Do not train tokenizer vocabularies on benchmark test text unless the paper explicitly defines a transductive setting; the default project flow trains only on the local `train.txt` split.

For transliteration consistency, construct a small human-checked test lexicon with multiple valid Roman spellings for each Hindi/Marathi word and include ambiguous Latin words and English words as hard negatives. The included ITRANS-generated pair score is only a pilot diagnostic. It does not measure whether a colloquial form is a correct transliteration.

## Outcomes

**Primary:** token-set consistency for human-validated cross-script pairs. Report both generated-pair and human-validated-pair results; the latter should decide the research conclusion.

**Intrinsic:** fertility, normalized sequence length, byte/token, vocabulary coverage/unknown rate, and measured throughput/latency with fixed hardware and batch settings. Report by script and code-mixing strata.

**Downstream:** official MeSent sentiment accuracy, macro-F1, and per-class scores from `hf-classify`; the TF-IDF classifier is a screening baseline only. Translation is outside the primary study. The translation command generates with a supplied seq2seq checkpoint and scores outputs; it does not fine-tune an MT model.

**Language-model diagnostic:** masked-LM validation loss from the vocabulary-extension script. Do not compare MLM token loss or token perplexity across differing token vocabularies as though it were a common text-level measure. Prefer a shared downstream task for cross-method quality comparison.

**Statistics:** sentence-paired bootstrap intervals and Wilcoxon signed-rank tests are implemented for intrinsic metrics; classification includes exact paired McNemar testing. Predefine primary metrics, adjust secondary comparisons for multiplicity, and report effect sizes and intervals, not only p-values.

## Minimum publication-quality run

1. Freeze dataset versions, licenses, train/dev/test IDs, text normalization, seeds, and verify no adaptation/test overlap.
2. Manually audit at least a meaningful held-out sample of cross-script word pairs and label English/Indic ambiguity.
3. Run original-tokenizer continuation, ordinary SentencePiece vocabulary extension, and transliteration-view extension with matched MLM steps and actual added-token counts; run at least three seeds if resources permit.
4. Evaluate primary consistency, intrinsic metrics, and MeSent on official splits. Select epochs on validation only, evaluate test once, and add ablations only after the primary comparison is stable.
5. Re-run the strongest result on a second dataset/domain or language, and include negative results if gains do not transfer.
6. Publish training scripts, configuration, package/hardware information, data hashes/IDs, metric code, and limitations. Never fill tables with expected or illustrative values.

## Related work to verify before making a novelty claim

- Purason et al. (2026), [Teaching Old Tokenizers New Words](https://aclanthology.org/2026.findings-eacl.341/), directly studies tokenizer vocabulary extension and pruning.
- Rana et al. (2025), [IndicSuperTokenizer](https://arxiv.org/abs/2511.03237), studies language-specific pretokenization and subword/superword tokens across Indic languages.
- Nanda team (2026), [Nanda Family](https://aclanthology.org/2026.eacl-long.288/), reports Hindi vocabulary extension and continued pretraining with Romanized/code-mixed data.
- Pattnayak et al. (2025), [Tokenization Matters](https://arxiv.org/abs/2504.16977), evaluates tokenizer choices on Indic NER tasks including Marathi.

These make generic claims such as “we are the first to adapt an Indic tokenizer” untenable without a narrower gap. Novelty should be decided only after checking these papers, their references, subsequent citations, and other proceedings current to the submission date.

## Paper structure

1. Abstract with actual measured findings only.
2. Introduction: code-mixed cross-script ambiguity, precise research question, and verified contribution.
3. Related work: tokenizer extension, Indic tokenization, transliteration, and code-mixed benchmarks.
4. Data: licensing, language/script proportions, splits, and transliteration-pair audit.
5. Methods: baselines, each tokenizer arm, vocabulary/model controls, and implementation details.
6. Evaluation: metrics, benchmark protocols, significance tests, and compute measurement.
7. Results: intrinsic, cross-script, downstream, ablation, and efficiency results.
8. Error analysis: false alignments, ambiguous Latin strings, inflection, spelling variation, and language-boundary errors.
9. Limitations, ethics/data privacy, and reproducibility statement.
10. Conclusion: answer the RQ without equating compression with understanding.
