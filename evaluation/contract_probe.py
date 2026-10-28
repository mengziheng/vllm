#!/usr/bin/env python3
"""Parse the real 0.4.13 CLI/task YAML and build request bodies without requests or scores."""
import argparse
import inspect
import json
from pathlib import Path
import sys

import compare as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import importlib.metadata as metadata
    import lm_eval
    from lm_eval._cli import HarnessCLI
    from lm_eval.models.api_models import TemplateAPI
    from lm_eval.models.openai_completions import LocalCompletionsAPI, LocalChatCompletion
    import torch
    import yaml
    class TaskLoader(yaml.SafeLoader):
        pass
    TaskLoader.add_multi_constructor('!', lambda loader, tag, node: loader.construct_scalar(node))
    c.require(metadata.version('lm_eval') == '0.4.13', 'probe requires actual lm_eval 0.4.13')
    report = {'lm_eval': metadata.version('lm_eval'), 'lm_eval_path': lm_eval.__file__,
              'default_max_gen_toks': inspect.signature(TemplateAPI.__init__).parameters['max_gen_toks'].default,
              'tasks': {}}
    for job in c.make_workloads(sys.executable, args.model, 'http://127.0.0.1:8000', '/unused')[:2]:
        task = job['expected']['task']
        relative = {'gsm8k': 'gsm8k/gsm8k.yaml', 'aime25': 'aime/aime25.yaml'}[task]
        task_path = Path(lm_eval.__file__).parent / 'tasks' / relative
        config = yaml.load(task_path.read_text(), Loader=TaskLoader)
        c.require(config['task'] == task, 'official task name differs')
        original_argv = sys.argv
        try:
            sys.argv = ['lm_eval', *job['argv'][3:]]
            parsed = HarnessCLI().parse_args()
        finally:
            sys.argv = original_argv
        model_args = parsed.model_args
        generation = parsed.gen_kwargs
        if isinstance(model_args, str):
            model_args = json.loads(model_args)
        if isinstance(generation, str):
            generation = json.loads(generation)
        c.require(model_args['num_concurrent'] == 8 and model_args['seed'] == 20261028, 'CLI model_args mismatch')
        official_generation = config['generation_kwargs']
        c.require(generation['until'] == official_generation['until'], 'official stop sequences changed')
        official_max = official_generation.get('max_gen_toks', report['default_max_gen_toks'])
        c.require(generation['max_gen_toks'] == official_max, 'official/default generation maximum changed')
        c.require(parsed.num_fewshot == c.ACCURACY_PROTOCOL[task]['fewshot'], 'fewshot mismatch')
        backend = LocalCompletionsAPI if task == 'gsm8k' else LocalChatCompletion
        # Explicit tokenizer_backend avoids LocalCompletionsAPI's optional remote auto-detection.
        # The real completion constructor loads only the local tokenizer; chat uses no tokenizer.
        client = backend(**model_args, batch_size=1)
        message = 'Question: CPU contract only\nAnswer:' if task == 'gsm8k' else [{'role': 'user', 'content': 'CPU contract only'}]
        payload = client._create_payload(message, generate=True, gen_kwargs=dict(generation), seed=model_args['seed'], eos=None)
        c.require(payload['max_tokens'] == official_max and payload['seed'] == 20261028 and payload['temperature'] == 0, 'actual API body mismatch')
        c.require(payload['stop'] == official_generation['until'], 'actual API stop mismatch')
        report['tasks'][task] = {'official_yaml': str(task_path), 'yaml_sha256': c.sha256(task_path),
                                'mode': parsed.model, 'fewshot': parsed.num_fewshot,
                                'apply_chat_template': getattr(parsed, 'apply_chat_template', False),
                                'model_args': model_args, 'generation_kwargs': generation, 'actual_payload': payload}
    c.require(not torch.cuda.is_initialized(), 'CPU contract probe initialized CUDA unexpectedly')
    report.update(cuda_initialized=False, requests_sent=0, dataset_loaded=False, scores_generated=False)
    with Path(args.output).open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
