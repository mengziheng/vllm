"""Capture CPU registrations used by the fixed GLM5Next execution chain."""
import importlib
import json
import os
from pathlib import Path
import sys

root = Path('/work')
sys.path.insert(0, os.environ.get('SUBJECT_ROOT', '/work/package/current'))
import torch
import vllm

result = {'vllm': vllm.__file__, 'modules': {}, 'schemas': {}}
for name in ('_C_stable_libtorch', '_moe_C_stable_libtorch', '_flashmla_C', '_flashmla_extension_C', '_flashkda_C'):
    try:
        module = importlib.import_module('vllm.' + name)
        result['modules'][name] = {'path': module.__file__}
    except Exception as exc:
        result['modules'][name] = {'error': repr(exc)}
expected = {
    '_moe_C::moe_align_block_size': 8, '_moe_C::moe_sum': 4,
    '_moe_C::grouped_topk': 8,
    '_flashkda_C::get_workspace_size': 3, '_flashkda_C::fwd': 16,
    '_flashmla_C::sparse_decode_fwd': 14, '_flashmla_C::dense_decode_fwd': 10,
    '_flashmla_C::sparse_prefill_fwd': 8, '_flashmla_C::dense_prefill_fwd': 13,
    '_C::cooperative_topk': 6, '_C::persistent_topk': 6,
    '_C::top_k_per_row_prefill': 8, '_C::top_k_per_row_decode': 8,
    '_C_cache_ops::concat_and_cache_mla': 7,
    '_C_cache_ops::cp_gather_and_upconvert_fp8_kv_cache': 9,
}
for name in (*expected, '_C::merge_attn_states', '_C::rms_norm', '_C::fused_add_rms_norm',
             '_C::rotary_embedding', '_C::silu_and_mul', '_C::silu_and_mul_with_clamp',
             '_C::fp32_router_gemm', '_C_cache_ops::concat_mla_q',
             '_C_cache_ops::indexer_k_quant_and_cache', '_C_cache_ops::cp_gather_indexer_k_quant_cache',
             '_C_cache_ops::cp_gather_cache'):
    try:
        schema = torch._C._dispatch_find_schema_or_throw(name, '').schema()
        result['schemas'][name] = {'schema': str(schema), 'argument_count': len(schema.arguments)}
        if name in expected:
            result['schemas'][name].update(expected_count=expected[name], count_matches=len(schema.arguments) == expected[name])
    except Exception as exc:
        result['schemas'][name] = {'error': repr(exc)}
result['cuda_initialized'] = torch.cuda.is_initialized()
assert not result['cuda_initialized']
destination = Path(os.environ.get('SCHEMA_RECEIPT', '/work/receipts/native-schemas-before.json'))
destination.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
