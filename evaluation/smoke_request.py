#!/usr/bin/env python3
"""Send one real model request as a startup check; compute no task score."""
import argparse
import json
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    body = {'model': args.model, 'prompt': 'Question: What is 1+1?\nAnswer:', 'max_tokens': 1,
            'temperature': 0, 'seed': 20261028, 'ignore_eos': True, 'cache_salt': 'pr58986-native-startup'}
    request = Request(args.endpoint.rstrip('/') + '/v1/completions', data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=600) as response:
        result = json.load(response)
    print(json.dumps({'request': body, 'response': result}, ensure_ascii=False), flush=True)
    if len(result.get('choices', [])) != 1 or result.get('usage', {}).get('completion_tokens') != 1:
        raise ValueError('startup request did not complete one real output token; response retained')


if __name__ == '__main__':
    main()
