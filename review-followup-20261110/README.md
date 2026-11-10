# November 10 review follow-up

Source commit: `40cab637f7d0c2ef03c6e696c98fa87451a62407`. PR base: `fa3b9d1377219247060d3c8b49eae199c9e7e154`.

The two current review requests remove the dedicated GLM tests and their MoE CI dependencies. The final PR has 36 additions in two files: the benchmark parser and the H20 JSON. Test and CI files match the base; the benchmark/config files match the previous head.

## Local validation

[Original pre-commit output](precommit.stdout.txt) records the applicable checks passing. [The receipt](validation.json) records the command, revisions and scope checks. Git whitespace validation passed. These checks were agent-executed; no GPU run was performed in this follow-up.

## Full PR audit

The remaining branch reads both GLM5Next text configs while retaining outer quantization metadata. The tuned file has 24 token-count entries and seven distinct launch configurations. Shape/key selection, required fields, FP8 tile constraints, nearest-key selection, user override precedence and batch-invariant behavior were reviewed. No concrete defect was found in the remaining two-file change.

Existing raw evidence was selectively checked: four official accuracy stdout blocks match the PR, all 28 workload exit records are zero, and the 24 serving runs each record 128 measured requests plus 16 warmups. Accuracy concurrency is 8, serving client concurrency is 128, and server maximum sequences is 32. These historical GPU runs used `eb34e115`, not this source commit.

## Pending H20 validation

Upstream commit `c56706e8` (PR #63820), merged after the GPU evaluation revision, enables TMA weight loading for the target H20 shapes. The existing dedicated TMA test selects H200. Current-head H20 validation of this changed execution path remains unrun. The available H20 host failed the read-only preflight because all GPUs were occupied and the Docker filesystem was full; no GPU job was launched.

A focused current-head H20 kernel comparison can exercise the seven tuned configurations and compare TMA with pointer loads, including CUDA Graph execution. This would supplement the historical full-model evidence; it would not turn the old full-model throughput numbers into current-head measurements.
