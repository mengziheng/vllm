# Validation evidence for vLLM PR #58986

The complete matched-native `mtp3-pair2` comparison finished successfully for [vLLM PR #58986](https://github.com/vllm-project/vllm/pull/58986). Both arms completed all 14 workloads on the same allocation of four H20 GPUs: full GSM8K and AIME25 accuracy plus twelve serving benchmarks per arm. The fixed vLLM source is `eb34e115d041ac685fa38eadf8eae6c51d61dc23`; both arms use the same model snapshot and rebuilt native runtime, TP4, Triton MoE, MTP3, and `VLLM_BATCH_INVARIANT=0`. Baseline removes the target tuned configuration; tuned retains it.

Quality results are mixed. GSM8K strict-match correct answers changed from **1225/1319 to 1219/1319**, and flexible-extract changed from **1226/1319 to 1220/1319**. AIME25 changed from **19/30 to 21/30**. These are the observed counts from one official accuracy run per arm. GSM8K used raw completions with five examples; AIME25 used chat completions with zero examples. The [machine summary](reports/pair2-summary.json) preserves official metrics, standard errors, sample counts, raw paths, and all benchmark repetitions. The [four official accuracy stdout blocks](reports/official-accuracy-stdout-all.txt) are copied from the selected successful commands.

The three-run medians of **total input-plus-output token throughput** changed as follows; these percentages compare tuned with baseline:

- Input/output 8192/1: 6791.86 → 6755.25 token/s, **−0.54%**.
- Input/output 2048/512: 3567.37 → 4097.71 token/s, **+14.87%**.
- Input/output 512/1024: 1908.55 → 2302.18 token/s, **+20.62%**.
- Input/output 512/128: 3992.04 → 4683.60 token/s, **+17.32%**.

Each of the 24 benchmark runs completed 128 measured requests and 16 successful warmups. The [24 official benchmark stdout blocks](reports/official-bench-stdout-all.txt) retain each selected output's exact bytes with relative source-path labels. The machine summary also retains output-token throughput and request throughput separately, their three raw values, and median/minimum/maximum; the range of three runs is not a confidence interval.

The [final raw directory](raw/fullpair-v10-pair2-48781766/) includes the complete [job/pair records](raw/fullpair-v10-pair2-48781766/job/pair/), [passed pair audit](raw/fullpair-v10-pair2-48781766/job/pair/pair-audit.json), [pool result](raw/fullpair-v10-pair2-48781766/pool-job-status.json), [launcher records](raw/fullpair-v10-pair2-48781766/launcher/), and [collection manifest](raw/fullpair-v10-pair2-48781766/collection-manifest.json). The collected archive's recorded SHA256 is `cda4c62a58cc8bfaa6f012f96871372d58db7c00e7db0472dd6d62bbc731ab28`; the archive itself, model weights, compiled libraries, and dependency caches are excluded from this branch. Original raw files remain unchanged.

The [logged-document audit](reports/pair2-logged-doc-audit.json) matched all four full test-document sets to the stored official preparation digests. GSM8K has 1319 unique documents and 2638 log rows per arm because its two official answer filters share one task logfile; AIME25 has 30 documents and 30 rows per arm. Cross-filter document copies must match before deduplication. The [offline preparation receipts](evidence/offline-dataset-preparation/README.md) retain the initial overall exit-1 check and the distinct later exit-0 tokenizer correction, and explain the canonical serialization and the earlier GSM8K training-split evidence.

The accuracy stderr includes `fatal: not a git repository` from lm-evaluation-harness's optional Git metadata lookup. That lookup is caught; it is not a model-request failure. The four evaluation commands independently exited 0 and produced complete official results. Source identity is established separately by the [client provenance](raw/fullpair-v10-pair2-48781766/job/pair/client-version/attempt-001/results/client-provenance.json), saved server proofs, and native receipts.

Runtime evidence includes the [allocation receipt](raw/fullpair-v10-pair2-48781766/job/allocation-sweep-unified-moe58986-bbbd3419b670.json), [hardware snapshot](raw/fullpair-v10-pair2-48781766/live-process-evidence/hardware-before-bench-48781766.csv), [baseline process maps](raw/fullpair-v10-pair2-48781766/live-process-evidence/native-maps-baseline-48781766.json), and [tuned process maps](raw/fullpair-v10-pair2-48781766/live-process-evidence/native-maps-tuned-48781766.json). [native/](native/), [build-scripts/](build-scripts/), and the [CPU preparation reports](preparation-reports/) retain build/source pins, compiler records, real CPU load/schema checks, and the frozen native receipt.

The [frozen v10 change record](evaluation/V10-CHANGES.md) preserves the earlier recovery proposal. A different allocation correctly rejected reuse of v9 results; [that rejection](raw/fullpair-v10-a80de6bd/) and the [original v9 output](raw/fullpair-v9-d8617fa4/) remain historical evidence. All reported final results above use the fresh pair2 allocation. The [isolated admission material](pool-admission/) records its script, eight final fixtures, development red/green logs, proof, and actual submissions; no shared pool file was changed. The [historical PR description](reports/PR-description-before-native-validation.md) preserves the previous PR text.

The two analysis runs have exact command/stdout/stderr/exit records beside [pair2-summary.json](reports/pair2-summary.json) and [pair2-logged-doc-audit.json](reports/pair2-logged-doc-audit.json). The official stdout concatenations are reproducible from the selected successful attempts with:

```sh
python analysis/collect_selected_stdout.py raw/fullpair-v10-pair2-48781766/job/pair --kind benchmark --output reports/official-bench-stdout-all.txt
python analysis/collect_selected_stdout.py raw/fullpair-v10-pair2-48781766/job/pair --kind accuracy --output reports/official-accuracy-stdout-all.txt
```

Each concatenation has a `.sources.json` index with byte offsets and payload lengths, plus its own exact generation command/stdout/stderr/exit records. Labels are added outside each original stdout payload; the payload bytes are unchanged. Dataset/source attribution is in [ATTRIBUTION.md](ATTRIBUTION.md). [PUBLICATION-CHECKS.json](PUBLICATION-CHECKS.json), [INCREMENTAL-CHECKS.json](INCREMENTAL-CHECKS.json), and [CANDIDATE-FILES.json](CANDIDATE-FILES.json) record publication checks and the candidate inventory.
