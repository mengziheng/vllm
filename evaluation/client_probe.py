#!/usr/bin/env python3
"""Record distribution versions separately from the actual client source imports."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path

import compare as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--native-receipt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    m = c.read(args.manifest)
    source_manifest = {**m, 'arms': {'client': {'repo': m['client_source_repo']}}}
    source = c.verify_source(source_manifest, 'client', dict(os.environ), args.native_receipt)
    import vllm
    import vllm.benchmarks.serve as serve
    obj = {'versions': {key: metadata.version(key) for key in ('lm_eval', 'vllm', 'transformers', 'datasets')},
           'vllm_file': str(Path(vllm.__file__).resolve()),
           'bench_serve_file': str(Path(serve.__file__).resolve()), 'source': source}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
    print(json.dumps(obj, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
