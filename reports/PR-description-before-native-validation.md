<!-- markdownlint-disable -->

## Overview

Add the H20 block-FP8 Triton MoE configuration for GLM-5.3-Flash at TP4 (E=288, per-rank N=512, 128x128 weight blocks), and support its nested text configuration in `benchmark_moe.py`. This review follow-up starts from `4c1cbbe8823ff5263dc3833b816b6238e4f2ff48`, which merges main at `39e6a1aafcaea1b90c515dacfc3a20835756758c`, preserving Gemma 4 benchmark support.

## Review follow-up (2026-10-28)

Commit: `eb34e115` (on top of `4c1cbbe8`). The resulting diff is 250 added lines: 195 configuration data, 9 benchmark parsing lines, 45 regression-test lines, and 1 CI dependency line. The ideal total-diff target of fewer than 100 lines is not met; the configuration has been retained in its generated format.

- Moved the 35-line tuning-guide addition out of the code diff and retained the usage, commands, and limitations in the collapsible section below. The existing H20 JSON and 9-line GLM parameter branch are unchanged. The JSON remains in its generated format; dropping token-count entries would change nearest-entry selection boundaries.
- Added three regression cases using real Transformers GLM config constructors, and included the benchmark helper in the existing benchmark-test CI source dependencies. The tests distinguish the MoE width (2048) from the dense MLP width (12288) and check outer FP8 block metadata.
- Ran the target tests with the current source, the same source with only the 9-line GLM branch removed, and the current source again: 3 passed; 2 failed / 1 passed as expected; 3 passed. These are CPU parameter-parsing checks with CUDA uninitialized. Public v0.32 native libraries were reused; missing `_version` / `vllm._C` warnings are retained in the raw output. This is not a clean main wheel, native ABI, or model-serving validation.
- Applicable pre-commit checks passed on the changed test, Buildkite YAML, and documentation removal, including test-to-CI selection, Ruff, formatting, mypy, and SPDX checks. The unchanged benchmark code and JSON retain their previously recorded checks.
- A TP4 current-source startup check was attempted in the shared H20 pool and stopped before readiness: the source overlay lacked the build-generated FlashMLA Python wrapper. CPU follow-up generated that wrapper from the current CMake pin, then found a definite interface mismatch with the borrowed v0.32 native library: the wrapper passes 14 arguments to `sparse_decode_fwd`, while the registered native operator accepts 13. No completion, model score, or serving throughput was produced. A matching `_flashmla_C` build and generated wrapper are required before the formal paired tests. The failed input and logs were preserved; the GPU attempt was not retried with an altered input.
- Standalone paired `lm_eval==0.4.13` and `vllm bench serve` tools are prepared outside the upstream diff. Full current-source model accuracy and serving performance are still pending; no new end-to-end numbers are claimed by this follow-up. Historical v0.30 results below keep their original identity.

<details>
<summary>CPU regression commands and raw output (current / without GLM branch / current)</summary>

Environment: Python 3.12.13, torch 2.13.0+cu130, Triton 3.7.1, Transformers 5.19.0, Ray 2.53.0, pytest 8.3.2. The CPU-only container requested no GPU devices. `/subject` and `/common` were read-only mounts; output and temporary files were on a data filesystem separate from Docker.

Each subprocess imported the actual benchmark module and vLLM Python source, then ran the following pytest invocation. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and `--confcutdir` isolate the target test from unrelated model fixtures; no tests were skipped or mocked.

```bash
PYTHONPATH=/subject/current:/common/site-packages:/work \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/local/bin/python3 -B -m pytest \
  /subject/current/tests/benchmarks/test_moe.py \
  --confcutdir=/subject/current/tests/benchmarks -p no:cacheprovider -vv --tb=short
```

For the negative control, replace `/subject/current` with `/subject/baseline`; that copy differs only by removal of the nine GLM parsing lines. The initial driver's case-sensitive check stopped after the valid negative-control run; those raw pytest results were preserved and reused, and only the final current-source subprocess was completed afterward.

### current-before

```text
IMPORT_EVIDENCE {"python": "3.12.13 (001c8cb:001c8cb, Sep 11 2026, 20:08:55) [GCC 12.2.0]", "executable": "/usr/local/bin/python3", "vllm": "/subject/current/vllm/__init__.py", "benchmark": "/subject/current/benchmarks/kernels/benchmark_moe.py", "constructors": ["transformers.models.glm5_next.configuration_glm5_next", "transformers.models.glm5_next.configuration_glm5_next"], "versions": {"torch": "2.13.0+cu130", "triton": "3.7.1", "transformers": "5.19.0", "ray": "2.53.0", "pytest": "8.3.2"}, "cuda_initialized_before": false}
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.3.2, pluggy-1.7.0 -- /usr/local/bin/python3
rootdir: /subject/current/tests/benchmarks
collecting ... collected 3 items

../subject/current/tests/benchmarks/test_moe.py::test_glm5next_moe_model_params[nested] PASSED [ 33%]
../subject/current/tests/benchmarks/test_moe.py::test_glm5next_moe_model_params[text-only] PASSED [ 66%]
../subject/current/tests/benchmarks/test_moe.py::test_glm5next_outer_fp8_weight_block_size PASSED [100%]

============================== 3 passed in 0.03s ===============================
CUDA_INITIALIZED_AFTER False

/subject/current/vllm/__init__.py:7: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from .version import __version__, __version_tuple__  # isort:skip
Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")
W1028 14:57:00.701000 8 site-packages/torch/utils/cpp_extension.py:178] No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
```

### baseline-without-glm

```text
IMPORT_EVIDENCE {"python": "3.12.13 (001c8cb:001c8cb, Sep 11 2026, 20:08:55) [GCC 12.2.0]", "executable": "/usr/local/bin/python3", "vllm": "/subject/baseline/vllm/__init__.py", "benchmark": "/subject/baseline/benchmarks/kernels/benchmark_moe.py", "constructors": ["transformers.models.glm5_next.configuration_glm5_next", "transformers.models.glm5_next.configuration_glm5_next"], "versions": {"torch": "2.13.0+cu130", "triton": "3.7.1", "transformers": "5.19.0", "ray": "2.53.0", "pytest": "8.3.2"}, "cuda_initialized_before": false}
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.3.2, pluggy-1.7.0 -- /usr/local/bin/python3
rootdir: /subject/baseline/tests/benchmarks
collecting ... collected 3 items

../subject/baseline/tests/benchmarks/test_moe.py::test_glm5next_moe_model_params[nested] FAILED [ 33%]
../subject/baseline/tests/benchmarks/test_moe.py::test_glm5next_moe_model_params[text-only] FAILED [ 66%]
../subject/baseline/tests/benchmarks/test_moe.py::test_glm5next_outer_fp8_weight_block_size PASSED [100%]

=================================== FAILURES ===================================
____________________ test_glm5next_moe_model_params[nested] ____________________
/subject/baseline/tests/benchmarks/test_moe.py:29: in test_glm5next_moe_model_params
    assert get_model_params(config) == (288, 8, 2048, 4096)
E   assert (288, 8, 12288, 4096) == (288, 8, 2048, 4096)
E     
E     At index 2 diff: 12288 != 2048
E     
E     Full diff:
E       (
E           288,
E           8,
E     -     2048,
E     +     12288,
E           4096,
E       )
__________________ test_glm5next_moe_model_params[text-only] ___________________
/subject/baseline/tests/benchmarks/test_moe.py:29: in test_glm5next_moe_model_params
    assert get_model_params(config) == (288, 8, 2048, 4096)
E   assert (288, 8, 12288, 4096) == (288, 8, 2048, 4096)
E     
E     At index 2 diff: 12288 != 2048
E     
E     Full diff:
E       (
E           288,
E           8,
E     -     2048,
E     +     12288,
E           4096,
E       )
=========================== short test summary info ============================
FAILED ../subject/baseline/tests/benchmarks/test_moe.py::test_glm5next_moe_model_params[nested] - assert (288, 8, 12288, 4096) == (288, 8, 2048, 4096)
  
  At index 2 diff: 12288 != 2048
  
  Full diff:
    (
        288,
        8,
  -     2048,
  +     12288,
        4096,
    )
FAILED ../subject/baseline/tests/benchmarks/test_moe.py::test_glm5next_moe_model_params[text-only] - assert (288, 8, 12288, 4096) == (288, 8, 2048, 4096)
  
  At index 2 diff: 12288 != 2048
  
  Full diff:
    (
        288,
        8,
  -     2048,
  +     12288,
        4096,
    )
========================= 2 failed, 1 passed in 0.06s ==========================
CUDA_INITIALIZED_AFTER False

/subject/baseline/vllm/__init__.py:7: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from .version import __version__, __version_tuple__  # isort:skip
Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")
W1028 14:57:11.182000 17 site-packages/torch/utils/cpp_extension.py:178] No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
```

### current-after

```text
IMPORT_EVIDENCE {"python": "3.12.13 (001c8cb:001c8cb, Sep 11 2026, 20:08:55) [GCC 12.2.0]", "executable": "/usr/local/bin/python3", "vllm": "/subject/current/vllm/__init__.py", "benchmark": "/subject/current/benchmarks/kernels/benchmark_moe.py", "constructors": ["transformers.models.glm5_next.configuration_glm5_next", "transformers.models.glm5_next.configuration_glm5_next"], "versions": {"torch": "2.13.0+cu130", "triton": "3.7.1", "transformers": "5.19.0", "ray": "2.53.0", "pytest": "8.3.2"}, "cuda_initialized_before": false}
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.3.2, pluggy-1.7.0 -- /usr/local/bin/python3
rootdir: /subject/current/tests/benchmarks
collecting ... collected 3 items

../subject/current/tests/benchmarks/test_moe.py::test_glm5next_moe_model_params[nested] PASSED [ 33%]
../subject/current/tests/benchmarks/test_moe.py::test_glm5next_moe_model_params[text-only] PASSED [ 66%]
../subject/current/tests/benchmarks/test_moe.py::test_glm5next_outer_fp8_weight_block_size PASSED [100%]

============================== 3 passed in 0.03s ===============================
CUDA_INITIALIZED_AFTER False

/subject/current/vllm/__init__.py:7: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from .version import __version__, __version_tuple__  # isort:skip
Failed to import from vllm._C: ModuleNotFoundError("No module named 'vllm._C'")
W1028 14:58:20.641000 8 site-packages/torch/utils/cpp_extension.py:178] No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
```

</details>

<details>
<summary>Current-source TP4 startup attempt and native-interface diagnosis (not model evaluation)</summary>

The test used the same current Python source as the starting PR head, H20 TP4, explicit Triton, MTP3, max model length 16384, max sequences 32, batch tokens 4096, memory utilization 0.92, seed 42, and read-only local weights. The source import probe and actual working directory both point to `/subject/current`. The container exited without OOM and never reached `/health` readiness. This startup check does not establish model quality or MoE numerical correctness.

Raw server output:

```text
/subject/current/vllm/__init__.py:7: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from .version import __version__, __version_tuple__  # isort:skip
(APIServer pid=9) INFO 10-28 15:13:29 [startup.py:38] 
(APIServer pid=9) INFO 10-28 15:13:29 [startup.py:38]        █     █     █▄   ▄█
(APIServer pid=9) INFO 10-28 15:13:29 [startup.py:38]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version dev
(APIServer pid=9) INFO 10-28 15:13:29 [startup.py:38]   █▄█▀ █     █     █     █  model   /model
(APIServer pid=9) INFO 10-28 15:13:29 [startup.py:38]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=9) INFO 10-28 15:13:29 [startup.py:38] 
(APIServer pid=9) INFO 10-28 15:13:29 [startup.py:96] non-default args: {'model_tag': '/model', 'host': '127.0.0.1', 'port': 18386, 'model': '/model', 'seed': 42, 'max_model_len': 16384, 'served_model_name': ['glm53-pr58986-smoke'], 'tensor_parallel_size': 4, 'language_model_only': True, 'max_num_batched_tokens': 4096, 'max_num_seqs': 32, 'moe_backend': 'triton', 'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 3}}
(APIServer pid=9) INFO 10-28 15:13:42 [model.py:744] Resolved architecture: Glm5NextForConditionalGeneration
(APIServer pid=9) INFO 10-28 15:13:42 [model.py:2170] Using max model len 16384
(APIServer pid=9) INFO 10-28 15:13:44 [model.py:964] All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.
(APIServer pid=9) INFO 10-28 15:13:44 [model.py:1021] Disabled mm_prefix attention mode because multimodal inputs are configuration-disabled. Attention backends without mm_prefix support may now be selected.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/subject/current/vllm/entrypoints/cli/main.py", line 123, in <module>
    main()
  File "/subject/current/vllm/entrypoints/cli/main.py", line 117, in main
    args.dispatch_function(args)
  File "/subject/current/vllm/entrypoints/cli/serve.py", line 156, in cmd
    uvloop.run(run_server(args))
  File "/usr/local/lib/python3.12/site-packages/uvloop/__init__.py", line 96, in run
    return __asyncio.run(
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "uvloop/loop.pyx", line 1515, in uvloop.loop.Loop.run_until_complete
  File "/usr/local/lib/python3.12/site-packages/uvloop/__init__.py", line 48, in wrapper
    return await main
           ^^^^^^^^^^
  File "/subject/current/vllm/entrypoints/launchers/api_server/entry.py", line 176, in run_server
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
  File "/subject/current/vllm/entrypoints/launchers/api_server/entry.py", line 190, in run_server_worker
    async with build_async_engine_client(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/contextlib.py", line 210, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/subject/current/vllm/entrypoints/launchers/api_server/entry.py", line 60, in build_async_engine_client
    async with build_async_engine_client_from_engine_args(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/contextlib.py", line 210, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/subject/current/vllm/entrypoints/launchers/api_server/entry.py", line 82, in build_async_engine_client_from_engine_args
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/subject/current/vllm/engine/arg_utils.py", line 2280, in create_engine_config
    model_config = self.create_model_config()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/subject/current/vllm/engine/arg_utils.py", line 1994, in create_model_config
    return ModelConfig(
           ^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in __init__
    s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
  File "/subject/current/vllm/config/model.py", line 922, in __post_init__
    self._verify_quantization()
  File "/subject/current/vllm/config/model.py", line 1430, in _verify_quantization
    method = me_quant.get_quantization_config(name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/subject/current/vllm/model_executor/layers/quantization/__init__.py", line 124, in get_quantization_config
    from vllm.models.deepseek_v41 import (
  File "/subject/current/vllm/models/deepseek_v41/__init__.py", line 13, in <module>
    from .nvidia.dspark import DSparkDeepseekV4ForCausalLM
  File "/subject/current/vllm/models/deepseek_v41/nvidia/dspark.py", line 51, in <module>
    from .model import (
  File "/subject/current/vllm/models/deepseek_v41/nvidia/model.py", line 78, in <module>
    from vllm.models.deepseek_v4.nvidia.model import (
  File "/subject/current/vllm/models/deepseek_v4/__init__.py", line 44, in <module>
    from .nvidia.dspark import (  # type: ignore[assignment]
  File "/subject/current/vllm/models/deepseek_v4/nvidia/dspark.py", line 54, in <module>
    from .model import (
  File "/subject/current/vllm/models/deepseek_v4/nvidia/model.py", line 77, in <module>
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention
  File "/subject/current/vllm/models/deepseek_v4/attention.py", line 65, in <module>
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
  File "/subject/current/vllm/v1/attention/backends/mla/sparse_swa.py", line 37, in <module>
    from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta, get_mla_metadata
  File "/subject/current/vllm/v1/attention/ops/flashmla.py", line 83, in <module>
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
ModuleNotFoundError: No module named 'vllm.third_party.flashmla.flash_mla_interface'
```

The missing file is generated by the current `cmake/external_projects/flashmla.cmake`, from FlashMLA revision `e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b`. The correctly generated current wrapper differs from the borrowed v0.32 wrapper. CPU-only operator-schema inspection, without initializing CUDA, reports:

```json
{
  "pinned_wrapper": [
    {
      "function": "sparse_decode_fwd",
      "python_positional_arguments": 14,
      "last_argument": "enable_batch_invariant",
      "line": 193
    }
  ],
  "borrowed_native_schema": "_flashmla_C::sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, Tensor? topk_length, Tensor? attn_sink, Tensor(a)? tile_scheduler_metadata, Tensor(b)? num_splits, Tensor? extra_kv, Tensor? extra_indices, Tensor? extra_topk_length, int d_v, float sm_scale, Tensor(c!)? out_) -> (Tensor(c!), Tensor, Tensor(a)?, Tensor(b)?)",
  "borrowed_native_argument_count": 13,
  "interface_compatible": false,
  "required_rebuild_target": "_flashmla_C at vLLM current CMake FlashMLA pin e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b, including matching generated Python wrapper",
  "gpu_v2_submitted": false,
  "cuda_initialized_during_schema_check": false
}
```

The next runtime prerequisite is a matching native FlashMLA build and generated wrapper; a replacement Python wrapper alone is insufficient. No second GPU attempt or large rebuild was started.

</details>

## Claims

- The configuration reduces expert-padding work with 16- and 32-row tiles in measured regions and preserves default settings elsewhere. The runtime configuration is byte-identical to the previous PR head.
- On the refreshed H20/CUDA 13 kernel comparison, the current configuration is 1.98x at 128 tokens, 1.56x at 640, 1.32x at 1024, and 1.15x at 2560; unchanged regions remain within 0.4% of the default. These are complete MoE-call CUDA Graph microbenchmarks, not new full-model throughput measurements.
- The existing corrected v0.30/CUDA 12.9 full-model results remain +18.24% mixed, +26.32% long-output, and +10.20% short-output throughput under the documented conditions below. They are historical measurements, not CUDA 13 or current-main model validation.

## Validation

### Refreshed rebase validation on H20 / CUDA 13

Main Python/Triton sources at `ae065918640` plus this change, PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.19.0, Ray 2.53.0. Native extensions are prebuilt v0.32 CUDA 13 artifacts; this is a validation of the relevant main MoE path, not a clean full-main wheel build or full-model serving on main. Baseline and PR arms use exactly the same source and native libraries, changing only the selected MoE configuration.

- 35 token counts x 3 seeds = **105/105** comparisons against the default pass at `atol=0.002`, `rtol=0.01`; maximum absolute difference **0.0001220703125**, maximum relative RMSE **0.000232742**.
- **210/210** captured-graph comparisons pass (default and PR, each against the default eager output). Lookup boundaries include 576/577, 1216/1217, 2304/2305, and 3328/3329.
- Additional independent FP32 GEMM reference checks at 1 and 128 tokens pass for both arms with the same tolerance. The reference retains the production FP8 quantization helper and independently computes GEMMs and SiLU/reduction semantics; it is not an independent quantizer implementation.
- Inputs are synthetic BF16 activations, FP8 expert weights, and routing scores using the dimensions read from the actual model configuration. The timed boundary is `fused_topk` plus activation quantization, expert alignment, both expert GEMMs, SiLU, and weighting/reduction. Router projection, TP communication, and host-side serving latency are outside this kernel benchmark.
- Numerical checks finish before timing. Timing calls the existing `benchmark_config` with 30 iterations, 10 CUDA Graph calls per iteration, and seeds 1051/2077/3091; arm order rotates. Values below are medians over the three seeds. No foreign compute process was detected on the assigned GPU; other GPUs on the host were in use.
- Actual checkpoint configuration parsing returns `(E, topk, intermediate_size, hidden_size) = (288, 8, 2048, 4096)` and block shape `[128,128]`.
- Applicable local pre-commit checks passed for the source used in this historical kernel measurement. The 2026-10-28 follow-up checks are recorded separately below. Upstream `pre-run-check` is blocked by the contributor authorization gate; pre-commit and full CI have not run there. The docs preview also stops at that same gate before rendering ([build log](https://app.readthedocs.org/projects/vllm/builds/35371614/)).

| Tokens | Default (us) | PR config (us) | Speedup |
|---:|---:|---:|---:|
| 1 | 47.26 | 47.26 | 1.000x |
| 96 | 1065.04 | 545.27 | 1.953x |
| 128 | 1117.77 | 565.54 | 1.976x |
| 512 | 1182.13 | 721.35 | 1.639x |
| 577 | 1181.48 | 750.80 | 1.574x |
| 640 | 1189.86 | 764.14 | 1.557x |
| 768 | 1198.84 | 791.53 | 1.515x |
| 1024 | 1221.39 | 926.97 | 1.318x |
| 1152 | 1210.58 | 1049.48 | 1.154x |
| 1217 | 1215.47 | 1215.37 | 1.000x |
| 2305 | 1807.33 | 1697.96 | 1.064x |
| 2560 | 2148.10 | 1867.32 | 1.150x |
| 3072 | 2400.99 | 2136.05 | 1.124x |
| 3329 | 2428.38 | 2428.92 | 1.000x |
| 4096 | 2601.36 | 2600.19 | 1.000x |

Existing CLI benchmark/retuning commands:

```bash
.venv/bin/python benchmarks/kernels/benchmark_moe.py \
  --model /path/to/GLM-5.3-Flash --tp-size 4 --dtype fp8_w8a8 \
  --batch-size 1 96 128 512 577 640 768 1024 1152 1217 2305 2560 3072 3329 4096

.venv/bin/python benchmarks/kernels/benchmark_moe.py \
  --model /path/to/GLM-5.3-Flash --tp-size 4 --dtype fp8_w8a8 \
  --tune --save-dir ./moe-tuned
```

The CLI command benchmarks the loaded configuration. The paired numerical and timing results above use the existing functions directly to compare the default and loaded configurations on identical inputs. Containerized Ray runs require sufficient shared memory; this environment uses 16 GiB. An initial 4 GiB attempt failed during Ray startup and produced no tuning results.

### Original search and full preset search

The original work included 2,345 candidate evaluations across 23 search runs and 21 distinct token counts. Across those runs it explored M={8,16,32,64,128}, N={64,128,256}, K={64,128}, GROUP={1,32}, warps={4,8}, and stages={2,3,4}; these are the union of targeted searches, not their full Cartesian product. The final selected configuration mostly changes M and GROUP because those choices survived the paired rechecks.

Following the review suggestion, the official `benchmark_moe.py --tune` has completed its full 1,920-configuration preset at each of the 24 checked-in token counts. All 24 searches exited successfully, and configuration files and resource receipts were collected; the unmodified tuner skips unsupported configurations. The selected parameters differ from the current PR at 19 points.

The candidates were then rechecked at those 19 points and 41 affected nearest-entry lookup boundaries, using the default, current PR, and candidate as three paired arms. All 180 candidate numerical comparisons and 540 three-arm CUDA Graph comparisons pass at the original `atol=0.002`, `rtol=0.01`; maximum candidate absolute difference is 0.0001220703125 and relative RMSE is 0.000284956. FP32 GEMM reference checks at 1, 16, and 128 tokens also pass, retaining production quantization. Timing uses the same 30 iterations, 10 Graph calls per iteration, and three seeds as above.

Compared with the current PR, the search candidates are about 1.135x at 4 tokens and 1.056x at 1280, but are not uniformly better: at the 1408 and 3840 token lookup boundaries their median latencies are about 3.9% and 3.3% higher. The configuration in this PR therefore remains unchanged; any selected-subset improvement or added lookup point is deferred until affected-boundary and model validation and human review are complete. These results do not claim new full-model gains.

Validation initially hit a CUDA-module-load crash caused by a truncated Triton cached cubin, confirmed against a complete binary with the same cache key. The original failure was retained, and only unfinished shapes were completed using fresh per-job caches. A separate Docker-disk-full attempt failed before test execution and was retained; the remaining checks ran through the same GPU pool and fixed environment on ready replicas. These execution failures are not numerical failures and have not been rewritten as successful original runs.

## Details

The small-batch improvement overlaps the heuristic proposed in [#47593](https://github.com/vllm-project/vllm/pull/47593), which remains an explicit historical comparison and is credited in the commit. The measured 32-row regions add gains beyond that rule. Existing H20 configuration PRs target other expert counts/shapes; the current duplicate check found no other open E=288/N=512 H20 block-FP8 configuration.

The SM90 mHC correctness repair remains in [#59129](https://github.com/vllm-project/vllm/pull/59129). Historical full-model measurements keep that repair in both arms. This PR does not include or claim the repair as an additional performance gain.

Configuration lookup, the batch-invariant exception, retuning, and explicit Triton backend selection are documented in the usage section below. Backend selection itself is unchanged.

AI assistance was used for investigation, implementation, tuning, test execution, and this description. The line-by-line human review and personal 128-token rerun reported on 2026-10-24 apply to the preceding revision. The new regression tests, CI selection, documentation relocation, and CPU validation in this follow-up were agent-executed; renewed human review of these added lines is not asserted. The broader historical benchmark and tuning runs were also agent-executed.

<details>
<summary>Historical CUDA 12.9 measurements, full-model pairing, quality checks, and limitations</summary>

## Purpose

GLM-5.3-Flash at TP4 uses 288 FP8 experts with a per-rank intermediate size of 512 and 128×128 weight blocks. H20 currently has no matching tuned configuration, so the default 64-row tile pads many expert assignments unnecessarily. Add a shape-specific configuration that selects 16- and 32-row tiles where measured faster and preserves defaults elsewhere. Also teach `benchmark_moe.py` to read the nested text configuration of the GLM-5.3-Flash architectures, so this shape can be benchmarked and retuned through the existing tool.

This does not replace the general heuristic in [#47593](https://github.com/vllm-project/vllm/pull/47593). That PR's 16/64-row rule is an explicit control in the experiments: the small-batch improvement overlaps it and is credited to that approach. The new 32-row regions provide additional gains at the tested 640–1216 and 2560–3328 token sizes. Existing H20 configuration PRs [#44152](https://github.com/vllm-project/vllm/pull/44152), [#44273](https://github.com/vllm-project/vllm/pull/44273) and [#54500](https://github.com/vllm-project/vllm/pull/54500) address other expert counts, shapes or quantization schemes; no open PR for this E=288, N=512 block-FP8 configuration was found.

The SM90 mHC correctness integration previously included here has been split into [independent PR #59129](https://github.com/vllm-project/vllm/pull/59129). This PR now changes only the H20 MoE configuration and benchmark model-config handling. The two changes can be reviewed and merged separately. The full-model results below require the mHC fix in both benchmark arms; they are not measurements of an unrepaired dependency.

## Test Plan

Retune/benchmark the model shape on H20 with the existing tool:

```bash
.venv/bin/python benchmarks/kernels/benchmark_moe.py \
  --model /path/to/GLM-5.3-Flash --tp-size 4 --dtype fp8_w8a8 \
  --batch-size 1 16 64 96 128 512 640 768 1024 1152 1280 2048 2560 2816 3072 4096
```

Independent measurements call that tool's `benchmark_config` directly, using 30 timing iterations (10 CUDA graph calls per iteration), seeds 1051/2077/3091, rotating arm order and changing routing between measurement iterations. All three arms receive the same seeds. `try_get_optimal_moe_config` selects the checked-in config through its normal nearest-entry lookup. Numerical comparisons against the previous default cover 31 batch sizes including lookup boundaries, three seeds each, BF16 activations and block-FP8 expert weights. This direct-function harness imports Ray but does not start a cluster.

## Test Result

- 93/93 numerical comparisons pass with atol=0.002 and rtol=0.01; all results finite. Maximum absolute difference 0.00015259, maximum relative RMSE 0.0002718.
- Applicable pre-commit checks pass for the release and main branches.
- Kernel timing results are microbenchmarks, not end-to-end throughput claims. Ratios below are inverse-latency throughput ratios, medians over three seeds.

| Tokens | Default µs | PR [#47593](https://github.com/vllm-project/vllm/pull/47593) µs | Tuned µs | vs default | vs prior |
|---:|---:|---:|---:|---:|---:|
| 1 | 43.85 | 43.88 | 44.03 | 0.996× | 0.997× |
| 16 | 224.28 | 223.72 | 224.23 | 1.000× | 0.998× |
| 64 | 505.66 | 506.94 | 505.88 | 1.000× | 1.002× |
| 96 | 1067.81 | 548.96 | 545.20 | 1.959× | 1.007× |
| 128 | 1108.53 | 575.58 | 573.39 | 1.933× | 1.004× |
| 512 | 1177.01 | 734.73 | 726.83 | 1.619× | 1.011× |
| 640 | 1185.11 | 1184.41 | 764.64 | 1.550× | 1.549× |
| 768 | 1202.94 | 1194.65 | 804.24 | 1.496× | 1.485× |
| 1024 | 1224.81 | 1233.43 | 983.35 | 1.246× | 1.254× |
| 1152 | 1245.12 | 1240.72 | 1130.48 | 1.101× | 1.098× |
| 1280 | 1268.78 | 1268.64 | 1268.07 | 1.001× | 1.000× |
| 2048 | 1480.73 | 1482.38 | 1480.17 | 1.000× | 1.001× |
| 2560 | 2190.99 | 2195.69 | 1936.64 | 1.131× | 1.134× |
| 2816 | 2383.78 | 2381.01 | 2061.45 | 1.156× | 1.155× |
| 3072 | 2471.71 | 2464.39 | 2130.63 | 1.160× | 1.157× |
| 4096 | 2664.28 | 2664.46 | 2664.75 | 1.000× | 1.000× |

### Refreshed full-model serving on v0.30

The earlier +27.10% / +9.46% / +8.02% results used an unrepaired mHC dependency. They remain historical observations and are superseded for the current code by the corrected comparison below. C0 is v0.30 `ced6857af` plus the three existing correctness backports in mengziheng/vllm#1; O1 is this MoE configuration; C1 is the mHC repair in mengziheng/vllm#3. Compare **C0+C1 → C0+O1+C1**, keeping the header fix in both arms and switching only the O1 JSON. The benchmark helper is not imported by serving.

Both arms use the same 62-shard checkpoint (config SHA256 `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`), 4×H20, TP4, FP8 weights/BF16 activations, MTP3, max length 16384, max sequences 32, batch tokens 4096, memory utilization 0.92, OMP threads 1. Weight memory is 77.62 GiB/card and KV cache 5.72 GiB/card (221976 tokens). Software is vLLM 0.30.0+cu129 native artifacts, PyTorch 2.13.0+cu129, Triton 3.7.1, driver 535.261.03 with CUDA 12.9 compatibility libraries.

Representative server and client commands (substitute the local checkpoint path):

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m vllm.entrypoints.cli.main serve /path/to/GLM-5.3-Flash \
  --served-model-name glm53-flash --host 127.0.0.1 --port 18230 \
  --tensor-parallel-size 4 --max-model-len 16384 --max-num-seqs 32 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.92 \
  --language-model-only --seed 42 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm --base-url http://127.0.0.1:18230 \
  --model /path/to/GLM-5.3-Flash --served-model-name glm53-flash \
  --dataset-name random --random-input-len 2048 --random-output-len 512 \
  --random-range-ratio '{"input":0.2,"output":0.5}' \
  --max-concurrency 32 --num-prompts 128 --seed 8971 \
  --temperature 0 --ignore-eos --num-warmups 16 \
  --save-result --save-detailed --result-dir results \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --extra-body '{"cache_salt":"replace-with-a-fresh-per-run-value"}'
```


Each workload uses three repeats, 128 measured requests and 16 warmups. Seed = `42 + repeat + input_len*3 + output_len*5 + concurrency*7`, repeat 1–3. A fresh `cache_salt` isolates each run. The pairing audit requires identical completed counts, total tokens and per-request input/output lengths. Report ratios of three-run medians, never a best run.

Run order: initial O1+C1, C0+C1 control, then O1+C1 return. The initial short run logged inference JIT and is retained; before any control results were available, the final control-to-return comparison was designated primary to check startup effects and timing drift. This is block-wise, not per-repeat interleaving.

Comparison: c1-control → c1-return. Values are medians of complete paired repeats.

| Workload (input/output/concurrency) | Previous output tok/s | New output tok/s | Change | P99 TTFT ms (old → new) | P99 TPOT ms (old → new) |
|---|---:|---:|---:|---:|---:|
| 2048 / 512 / 32 | 962.39 | 1137.92 | +18.24% | 11271.37 → 9205.95 | 38.02 → 33.33 |
| 512 / 1024 / 32 | 1619.19 | 2045.35 | +26.32% | 9531.62 → 7236.48 | 26.25 → 19.03 |
| 512 / 128 / 32 | 820.00 | 903.60 | +10.20% | 2697.35 → 2351.59 | 57.26 → 49.48 |

All **3,456/3,456** measured requests in the three nine-run blocks succeeded. Each final pair has identical request counts, token totals and per-request input/output lengths. Final output-throughput gains are **26.32% long-output, 10.20% short-output and 18.24% mixed-workload**; all three median P99 TTFT and TPOT values decrease. These gains revalidate O1 on a corrected baseline; they are not an incremental speedup attributed to C1.

The initial candidate block is retained:

Comparison: c1-control → c1. Values are medians of complete paired repeats.

| Workload (input/output/concurrency) | Previous output tok/s | New output tok/s | Change | P99 TTFT ms (old → new) | P99 TPOT ms (old → new) |
|---|---:|---:|---:|---:|---:|
| 2048 / 512 / 32 | 962.39 | 1136.24 | +18.06% | 11271.37 → 10028.50 | 38.02 → 38.12 |
| 512 / 1024 / 32 | 1619.19 | 2053.96 | +26.85% | 9531.62 → 7305.97 | 26.25 → 18.61 |
| 512 / 128 / 32 | 820.00 | 802.43 | -2.14% | 2697.35 → 4716.01 | 57.26 → 48.61 |

Initial short-output runs were 549.20/935.19/802.43 tok/s with P99 TTFT 5547.68/2450.01/4716.01 ms. Return short-output runs are listed below. JIT warnings do not by themselves establish the cause; short-output startup/shape variability remains a limitation, not a production latency guarantee. Return-vs-initial median drift is 512/128: +12.61%, 512/1024: -0.42%, 2048/512: +0.15%.

Long-output preemptions were control 27/30/23 and return 35/32/32; mixed workloads were control 34/58/38 and return 36/45/40. Short workloads had none. All runs use the same KV capacity. Each workload is concurrency32, saturation/random traffic, not an SLO-constrained production capacity measurement.

| Workload | Control output tok/s (r1/r2/r3) | Return output tok/s (r1/r2/r3) |
|---|---|---|
| 512/128/32 | 820.00/801.23/847.37 | 897.22/925.37/903.60 |
| 512/1024/32 | 1622.76/1619.15/1619.19 | 1866.43/2045.35/2057.20 |
| 2048/512/32 | 1031.36/955.37/962.39 | 1166.90/1137.92/1112.09 |


### Model quality and scope

O1+C1: GSM8K first 128, five-shot raw-completion diagnostic, greedy, concurrency 8, output cap 768: **121/128 correct**, all 128 requests successful, zero truncation. All six retrieval checks at 511/2047/2048/2049/4097/8193 tokens pass. The corrected control also scores 121/128 with zero failed or truncated requests. Prompts and dataset hashes match. Only 39/128 output texts are identical across these two runs; each arm uniquely answers three questions correctly. Previous C0 diagnostics scored 117/128 and 118/128; O1 scored 120/128 and 123/128. These subsets do not demonstrate an accuracy improvement or full-model equivalence. Whole-model exact repeatability is not asserted.

The full model was tested on the requested v0.30 release with the correctness backports. The upstream port was separately validated on H20 using main Python/Triton sources at `4a190dc51` plus this change: 35 batch sizes × three seeds = 105 numerical comparisons, including batch sizes 577, 1217, 2305 and 3329 immediately across configuration lookup boundaries. All pass with the same tolerance; maximum absolute difference 0.00015259 and maximum relative RMSE 0.0002718.

Main's MoE alignment API has an additional optional argument, so its MoE native extension was rebuilt from unmodified main sources with CUDA 12.9, GCC 12 and PyTorch 2.13. The other native extensions came from the release wheel. This validates the relevant main FP8 MoE path, not a full main-branch wheel or full-model serving on main. The initial attempt to reuse the release MoE binary failed on that ABI mismatch and was not counted as a successful test.


The mHC header repair and its regression test are submitted separately. GPU model validation remains v0.30. During the PR split, the complete performance-branch tree was verified identical to the previously validated O1-only commit `f2e71f998`. GPU tests were not rerun for this packaging-only split. B200, expert parallelism, other TP sizes/quantization formats, production traffic and long soak tests were not performed. Saturation throughput is not a production capacity/SLO guarantee.

</details>


<details>
<summary>Triton MoE configuration usage and retuning</summary>

When GLM-5.3-Flash uses the Triton fused MoE backend on NVIDIA H20 at TP4,
vLLM automatically loads the bundled configuration for 288 experts, a per-rank
intermediate size of 512, and FP8 weights with 128×128 quantization blocks.
No additional flag is needed for this configuration. It tunes the existing
Triton backend; it does not change which MoE backend vLLM selects.
Batch-invariant execution (`VLLM_BATCH_INVARIANT=1`) uses its fixed configuration
instead of tuned configurations.

To search the benchmark's parameter space for this model from a source checkout:

```bash
python benchmarks/kernels/benchmark_moe.py \
  --model /path/to/GLM-5.3-Flash --tp-size 4 --dtype fp8_w8a8 \
  --tune --save-dir ./moe-tuned
```

The benchmark reads the model configuration and uses synthetic inputs; it does
not load the model weights. Its batch size is the number of tokens passed to
MoE, rather than the number of serving requests. Use `--batch-size` to select
the token counts to tune. Set `VLLM_TUNED_CONFIG_FOLDER` to the generated
configuration directory to evaluate the results in vLLM. Validate numerical
correctness and model behavior before adopting new settings, and measure
performance with the intended hardware, software versions, and serving workload.

For example, start a new server process with the generated configuration and
explicitly select Triton:

```bash
VLLM_TUNED_CONFIG_FOLDER=/absolute/path/to/moe-tuned \
  vllm serve /path/to/GLM-5.3-Flash \
  --tensor-parallel-size 4 --moe-backend triton
```

</details>

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.
- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Historical validation and its limitations are recorded above. The new CPU regression tests passed as described; human confirmation of this follow-up and the requested current-source end-to-end accuracy/performance evidence remain pending.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>


