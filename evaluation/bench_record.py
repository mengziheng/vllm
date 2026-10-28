#!/usr/bin/env python3
"""Run the official vLLM CLI and record actual requests after tokenizer alignment."""
import hashlib
import json
from pathlib import Path
import sys
from functools import partial


def request_receipt(requests):
    return [{'index': index,
             'prompt_sha256': hashlib.sha256(json.dumps(request.prompt, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
             'input_len': request.prompt_len, 'output_len': request.expected_output_len}
            for index, request in enumerate(requests)]


async def record_benchmark(original, receipt, *args, serve_module=None, **kwargs):
    # The official main passes requests by keyword, after server tokenizer alignment.
    receipt = Path(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    obj = {'requests': request_receipt(kwargs['input_requests']), 'request_outcomes': []}
    if serve_module is None:
        with receipt.open('x') as stream:
            json.dump(obj, stream, ensure_ascii=False, indent=2)
        return await original(*args, **kwargs)
    backend = kwargs['endpoint_type']
    original_request = serve_module.ASYNC_REQUEST_FUNCS[backend]
    original_wait = serve_module.wait_for_endpoint
    phase = 'warmup'
    non_readiness_calls = 0
    async def recorded_request(*request_args, **request_kwargs):
        nonlocal non_readiness_calls
        this_phase = phase
        if this_phase != 'readiness':
            this_phase = 'warmup' if non_readiness_calls < kwargs['num_warmups'] else 'measured'
            non_readiness_calls += 1
        row = {'phase': this_phase, 'call_index': len(obj['request_outcomes'])}
        obj['request_outcomes'].append(row)
        try:
            output = await original_request(*request_args, **request_kwargs)
            row.update(success=bool(output.success), error=output.error)
            return output
        except BaseException as error:
            row.update(success=False, error=str(error))
            raise
    async def recorded_wait(*wait_args, **wait_kwargs):
        nonlocal phase
        phase = 'readiness'
        try:
            return await original_wait(*wait_args, **wait_kwargs)
        finally:
            phase = 'warmup'
    serve_module.ASYNC_REQUEST_FUNCS[backend] = recorded_request
    serve_module.wait_for_endpoint = recorded_wait
    try:
        result = await original(*args, **kwargs)
    finally:
        serve_module.ASYNC_REQUEST_FUNCS[backend] = original_request
        serve_module.wait_for_endpoint = original_wait
        # Only memory bookkeeping occurs per request; the raw receipt is written after timing.
        with receipt.open('x') as stream:
            json.dump(obj, stream, ensure_ascii=False, indent=2)
    warmups = [row for row in obj['request_outcomes'] if row['phase'] == 'warmup']
    if len(warmups) != kwargs['num_warmups'] or any(not row['success'] or row['error'] for row in warmups):
        raise ValueError('warmup request failure or missing warmup outcome; raw outcomes retained')
    return result


def main():
    if len(sys.argv) < 5 or sys.argv[1] != '--input-receipt' or sys.argv[3] != '--':
        raise ValueError('expected --input-receipt FILE -- bench serve ...')
    receipt = Path(sys.argv[2])
    sys.argv = ['vllm', *sys.argv[4:]]
    import vllm.benchmarks.serve as serve
    from vllm.entrypoints.cli.main import main as official_main
    original = serve.benchmark
    serve.benchmark = partial(record_benchmark, original, receipt, serve_module=serve)
    official_main()


if __name__ == '__main__':
    main()
