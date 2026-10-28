# Offline dataset preparation receipts

This directory preserves the existing small preparation metadata and logs. It contains no dataset cache, model weights, or new data-loading run. [official-data-manifest.json](official-data-manifest.json) records the official repositories, fixed revisions, split sizes, source files, and previously established canonical row digests. [provenance-and-tokenizer-resolution.json](provenance-and-tokenizer-resolution.json) explains how the separate checks relate.

The initial [89567e69958a receipt](sweep-unified-moe58986-89567e69958a.json) has exit code **1**. Its [raw log](sweep-unified-moe58986-89567e69958a.log) records that all three official splits loaded and their row counts/digests matched before a later tokenizer assertion failed. The checker incorrectly expected `list[int]` from the tokenizer's default `return_dict=True` behavior. This initial command did not pass overall.

The later [96741e4c4062 receipt](sweep-unified-moe58986-96741e4c4062.json) has exit code **0**. Its [raw log](sweep-unified-moe58986-96741e4c4062.log) and [tokenizer interface report](tokenizer-interface-report.json) record the explicit `return_dict=False` correction and two first-question input checks. This later run did not reload the full datasets or generate model answers or scores. The initial failure and the narrower later success remain separate evidence.

The previously captured official [GSM8K YAML](official-task-definitions/gsm8k.yaml) and [AIME25 YAML](official-task-definitions/aime25.yaml) contain no `process_docs` or `include` configuration. AIME25 specifies `process_results` for scoring. This inspection establishes the configured task behavior; equivalence of the actual logged document objects still requires their canonical digests to match.

For the collected final pair, [audit_logged_docs.py](../../analysis/audit_logged_docs.py) reads each selected complete accuracy attempt's JSONL, rejects duplicate IDs within one filter or different `doc` objects across filters, orders unique integer `doc_id` values, and hashes the full unmodified `doc`. It uses UTF-8 JSON with sorted keys, `ensure_ascii=False`, comma/colon separators, and one newline after every document including the last. It performs no field projection, string normalization, task processing, scoring, dataset loading, or cache recheck.

The stored test digests are:

- GSM8K, 1319 documents: `32c548f08195e19e33408b844dd7be6aa4bcae457d957bc05909ff4bd4a00595`.
- AIME25, 30 documents: `93226328a3bc794c9b7cf8ccd90beb65491bafa0ac58b132f3c11fe9976131b9`.

A successful audit establishes equality of the four logged test-document sets with these prepared splits. It does not reconstruct the GSM8K training/few-shot source from test logs or establish overall model accuracy/performance. These receipt files remain the original preparation records. The subsequently completed pair2 has a [separate logged-document audit](../../reports/pair2-logged-doc-audit.json) with status `matched` for all four test sets.
