"""Check the assembled package through its real wrappers in a fresh process."""
import atexit
import importlib
import json
import os
from pathlib import Path
import sys

repository = Path(os.environ.get('SUBJECT_ROOT', '/work/assembled/tuned'))
os.chdir(repository)
sys.path.insert(0, str(repository))
import torch

result = {'subject': str(repository), 'modules': {}, 'load_library': [],
          'loaded_libraries_before': sorted(torch.ops.loaded_libraries), 'schemas': {}, 'status': 'incomplete'}
destination = Path(os.environ.get('PROBE_RECEIPT', '/work/receipts/assembled-runtime-probe.json'))
atexit.register(lambda: destination.write_text(json.dumps(result, indent=2)))
original_load = torch.ops.load_library
def record_load(path):
    result['load_library'].append(str(Path(path).resolve()))
    return original_load(path)
torch.ops.load_library = record_load

import vllm
result['vllm'] = vllm.__file__
result['vllm_python_version'] = getattr(vllm, '__version__', None)
assert Path(vllm.__file__).is_relative_to(repository)
for name in ('vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch',
             'vllm._flashkda_C', 'vllm._flashmla_extension_C',
             'vllm.third_party.flashmla.flash_mla_interface',
             'vllm.vllm_flash_attn', 'vllm.vllm_flash_attn.flash_attn_interface',
             'vllm._custom_ops'):
    module = importlib.import_module(name)
    result['modules'][name] = str(Path(module.__file__).resolve())
    assert Path(module.__file__).resolve().is_relative_to(repository), name
deep_gemm_utils = importlib.import_module('vllm.utils.deep_gemm')
deep_gemm = deep_gemm_utils._import_deep_gemm()
assert deep_gemm is not None
result['modules']['deep_gemm_lazy_loader'] = str(Path(deep_gemm.__file__).resolve())
assert Path(deep_gemm.__file__).resolve().is_relative_to(repository)
for name in ('vllm._flashmla_C', 'vllm.vllm_flash_attn._vllm_fa3_C',
             deep_gemm.__name__ + '._C'):
    module = importlib.import_module(name)
    result['modules'][name] = str(Path(module.__file__).resolve())
    assert Path(module.__file__).resolve().is_relative_to(repository), name

expected = {'_flashmla_C::sparse_decode_fwd': 14,
            '_C_cache_ops::concat_and_cache_mla': 7,
            '_moe_C::moe_align_block_size': 8, '_moe_C::moe_sum': 4,
            '_moe_C::grouped_topk': 8, '_flashkda_C::fwd': 16,
            '_flashkda_C::get_workspace_size': 3,
            '_flashmla_extension_C::get_mla_decoding_metadata_dense_fp8': 3,
            '_flashmla_extension_C::fwd_kvcache_mla_fp8': 11}
for name in (*expected, '_C::cooperative_topk', '_C::persistent_topk',
             'deep_gemm::fp8_fp4_mqa_logits', 'deep_gemm::fp8_fp4_paged_mqa_logits'):
    schema = torch._C._dispatch_find_schema_or_throw(name, '').schema()
    result['schemas'][name] = {'schema': str(schema), 'argument_count': len(schema.arguments)}
    if name in expected:
        assert len(schema.arguments) == expected[name], name
result['cuda_initialized'] = torch.cuda.is_initialized()
assert not result['cuda_initialized']
result['loaded_libraries_after'] = sorted(torch.ops.loaded_libraries)
result['native_maps'] = [line for line in Path('/proc/self/maps').read_text().splitlines()
                         if any(tag in line for tag in ('_flashmla_', '_flashkda_', '_moe_C_', '_C_stable_', '_C_extension', '_vllm_fa3_'))]
assert result['native_maps']
assert all('/common/frameworks/' not in line for line in result['native_maps'])
result['pins'] = json.loads(Path('/work/receipts/native-receipt.json').read_text())['pins']
result['status'] = 'passed'
destination.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'native_maps'}, indent=2))
