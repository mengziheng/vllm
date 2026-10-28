"""Load full official lm_eval datasets and prepare two tokenizer inputs on CPU."""
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import logging
import os
from pathlib import Path
import sys

os.environ.update(
    PYTHONPATH="/client:/common/frameworks/v032:/common/site-packages:/work",
    HF_HOME="/work/.cache/unified/huggingface",
    HF_HUB_CACHE="/work/.cache/unified/huggingface/hub",
    HF_DATASETS_CACHE="/work/.cache/unified/huggingface/datasets",
    HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
)
sys.path.insert(0, "/client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
out = Path("/work/results")
out.mkdir(exist_ok=True)
manifest = json.loads(Path("/work/official-data-manifest.json").read_text())
report = {
    "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "python_executable": sys.executable, "python": sys.version,
    "packages": {name: metadata.version(name) for name in ("lm_eval", "datasets", "pyarrow", "huggingface_hub", "transformers", "torch")},
    "env": {key: os.environ.get(key) for key in ("PYTHONPATH", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "OPERATOR_SWEEP_RUN_ID")},
    "model_loading_invoked": False, "model_inference_invoked": False,
    "scoring_invoked": False, "tasks": {},
}
print(json.dumps({"runtime": report}, indent=2), flush=True)
from lm_eval.tasks import TaskManager

manager = TaskManager()
loaded = manager.load(["gsm8k", "aime25"])
tasks = loaded["tasks"]
assert set(tasks) == {"gsm8k", "aime25"}
for name, task in tasks.items():
    source = next(x for x in manifest["datasets"] if x["repo_id"] == task.config.dataset_path)
    assert task.config.dataset_name == source["dataset_name"]
    split_report = {}
    for split_name, dataset in task.dataset.items():
        expected = source["splits"][split_name]
        digest = hashlib.sha256()
        for row in dataset:
            digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
        actual = {"rows": len(dataset), "columns": dataset.column_names,
                  "fingerprint": dataset._fingerprint,
                  "canonical_rows_sha256": digest.hexdigest(), "cache_files": dataset.cache_files}
        assert actual["rows"] == expected["rows"], (name, split_name, actual, expected)
        assert actual["canonical_rows_sha256"] == expected["canonical_rows_sha256"], (name, split_name)
        split_report[split_name] = actual
    entry = manager.task_index[name]
    yaml_path = Path(entry.yaml_path)
    report["tasks"][name] = {
        "dataset_path": task.config.dataset_path,
        "dataset_name": task.config.dataset_name,
        "fixed_repo_revision": source["revision"],
        "official_yaml": str(yaml_path),
        "official_yaml_sha256": hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
        "splits": split_report, "evaluation_rows": len(task.eval_docs),
        "official_num_fewshot": task.config.num_fewshot,
    }
    print(json.dumps({"loaded_task": name, **report["tasks"][name]}, indent=2), flush=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True, trust_remote_code=False)
assert tokenizer.chat_template, "The local tokenizer has no chat template"
tokenizer_files = {}
for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"):
    path = Path("/model") / filename
    if path.is_file():
        tokenizer_files[filename] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
report["tokenizer"] = {
    "path": "/model", "class": type(tokenizer).__name__,
    "vocab_size": tokenizer.vocab_size, "tokenizer_length": len(tokenizer),
    "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    "input_files": tokenizer_files, "samples": {},
    "input_scope": "First evaluation document per task; official task few-shot default, seed 1234, one user message, add_generation_prompt=True. Input preparation only; these are not evaluation results.",
}
for name, task in tasks.items():
    task.set_fewshot_seed(1234)
    context = task.fewshot_context(task.eval_docs[0], num_fewshot=task.config.num_fewshot)
    messages = [{"role": "user", "content": context}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    assert isinstance(rendered, str) and rendered
    assert isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], int)
    retokenized = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    assert token_ids == retokenized
    sample = {"task": name, "evaluation_doc_index": 0, "fewshot_seed": 1234,
              "num_fewshot": task.config.num_fewshot, "input_token_count": len(token_ids),
              "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
              "token_ids": token_ids}
    (out / f"tokenizer-input-{name}.txt").write_text(rendered)
    (out / f"tokenizer-input-{name}.json").write_text(json.dumps(sample, indent=2) + "\n")
    report["tokenizer"]["samples"][name] = {key: value for key, value in sample.items() if key != "token_ids"}

import torch
report["cuda_initialized"] = torch.cuda.is_initialized()
assert report["cuda_initialized"] is False
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
report["status"] = "passed"
(out / "offline-load-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
