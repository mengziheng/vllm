# Sources and attribution

This branch records the build and evaluation evidence for vLLM PR #58986. Model weights are not included.

- vLLM source: https://github.com/vllm-project/vllm, fixed commit `eb34e115d041ac685fa38eadf8eae6c51d61dc23` (Apache-2.0). External native-source revisions and build commands are recorded with the build evidence.
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness, package `lm_eval==0.4.13` (MIT).
- GSM8K: OpenAI, https://huggingface.co/datasets/openai/gsm8k, revision `740312add88f781978c0658806c59bc2815b9866`, `main` configuration (MIT). Dataset paper: Cobbe et al., *Training Verifiers to Solve Math Word Problems*, 2021, https://arxiv.org/abs/2110.14168.
- AIME25: Yifan Zhang and Math-AI Team, *American Invitational Mathematics Examination (AIME) 2025*, https://huggingface.co/datasets/math-ai/aime25, revision `563bb8404243c5f09de6ec262f2db674fe5bce9b` (dataset card: Apache-2.0).

Per-sample evaluation logs may contain benchmark questions and reference answers from these datasets. Generated model responses and raw command output are retained as evaluation evidence; they are not independent benchmark inputs.

The GSM8K source repository MIT notice is retained in [third-party-licenses/GSM8K-MIT.txt](third-party-licenses/GSM8K-MIT.txt).
