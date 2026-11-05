# PR #58986: consolidate the GLM MoE regression tests

The November 5 review asked to combine the new tests with an existing suite. The same three cases now live in `tests/kernels/moe/test_moe.py`; `tests/benchmarks/test_moe.py` is removed. Imports are local to the moved tests. Both NVIDIA and AMD MoE CI dependency lists include `benchmarks/kernels/benchmark_moe.py`; the old benchmark CI entry is removed. The PR adds 84 lines across four files.

The test bodies, assertions and parametrization match the previous tests, as recorded in [the AST comparison](relocation-equivalence.json). The source is PR head `ef428c29` plus the test/CI relocation; this follow-up does not change runtime source or the H20 configuration.

## Actual test run

[Original stdout](results-cpu-platform/stdout.txt), [stderr](results-cpu-platform/stderr.txt), [command](results-cpu-platform/command.json), [exit code](results-cpu-platform/exit.json), and [source/environment provenance](provenance-cpu-platform.json) are retained. The real existing module collected 1129 cases and selected the three GLM cases: **3 passed, 1126 deselected, 3 warnings in 0.39s**. CUDA remained uninitialized before and after the test. Real Transformers config constructors and the current benchmark module were imported from the source snapshot.

The CPU-only launcher used the shared project environment, no GPU allocation and the supported `VLLM_TARGET_DEVICE=cpu` setting. The exact child invocation is in command.json. Equivalent targeted pytest selection from a configured checkout is:

```bash
VLLM_TARGET_DEVICE=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -B -m pytest tests/kernels/moe/test_moe.py \
  --confcutdir=tests/kernels/moe -p no:cacheprovider \
  -k glm5next -vv --tb=short
```

The [first attempt](results/stdout.txt) omitted the CPU selector, so module-level `VllmConfig()` could not infer a device and collection exited 2 before running a test. It is preserved with its command, stderr and exit code. The successful attempt only adds the supported CPU selector, with the same source and assertions. No tests were extracted, mocked, skipped, marked xfail or retried. Three pre-existing flaky marks on unselected tests produce warnings because plugin autoload is disabled; no retry plugin ran.

Common v032 native-library symlinks were used for real Python imports. This does not validate freshly built native extensions or GPU execution for the current head. Missing generated version/core extension and CUDA-runtime warnings are preserved. The container exited 0 without OOM and had no GPU request. Task output, temporary files, caches and logs used the independent data03 filesystem, separate from Docker data00, with a 2 GiB output budget and log rotation.

[Applicable pre-commit checks](precommit.stdout.txt) passed, including Ruff, formatting, type checks, SPDX and CI selection. Old full H20 accuracy and performance logs remain unchanged at their original immutable links; they were run on `eb34e115` and were not rerun for this test relocation or the subsequent merge of main.
