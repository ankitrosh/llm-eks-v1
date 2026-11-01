"""Download only the checkpoint shards needed by one Qwen3-14B rank."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument('--rank', type=int, required=True, choices=[0, 1])
p.add_argument('--split', type=int, default=20)
p.add_argument('--revision', default='main')
p.add_argument('--repo', default='Qwen/Qwen3-14B')
p.add_argument('--model-dir', default='models/qwen3-14b')
a = p.parse_args()

root = Path(a.model_dir)
root.mkdir(parents=True, exist_ok=True)

sha = HfApi().model_info(a.repo, revision=a.revision).sha

metadata = [
    'config.json',
    'model.safetensors.index.json',
    'tokenizer*',
    'vocab.json',
    'merges.txt',
    'generation_config.json',
    'special_tokens_map.json',
]

snapshot_download(
    repo_id=a.repo,
    revision=sha,
    local_dir=str(root),
    allow_patterns=metadata,
)

config = json.loads((root / 'config.json').read_text())
n = config['num_hidden_layers']
assert 0 < a.split < n

start, stop = (
    (0, a.split)
    if a.rank == 0
    else (a.split, n)
)

prefixes = [
    f'model.layers.{i}.'
    for i in range(start, stop)
]

if a.rank == 0:
    prefixes.append('model.embed_tokens.')
else:
    prefixes += ['model.norm.', 'lm_head.']

index = json.loads(
    (root / 'model.safetensors.index.json').read_text()
)

keys = [
    k for k in index['weight_map']
    if any(k.startswith(x) for x in prefixes)
]

shards = sorted({
    index['weight_map'][k]
    for k in keys
})

assert keys and shards

print(
    f'rank={a.rank} layers=[{start},{stop}) '
    f'tensors={len(keys)}',
    flush=True,
)
print('sha=', sha, flush=True)
print('shards=', shards, flush=True)

snapshot_download(
    repo_id=a.repo,
    revision=sha,
    local_dir=str(root),
    allow_patterns=shards,
)

for s in shards:
    assert (root / s).is_file(), f'missing {s}'

(root / 'checkpoint_revision.txt').write_text(sha + '\n')

(root / f'rank{a.rank}_manifest.json').write_text(
    json.dumps({
        'rank': a.rank,
        'start': start,
        'stop': stop,
        'sha': sha,
        'shards': shards,
        'owned_tensor_count': len(keys),
    }, indent=2) + '\n'
)

print('PASS: rank-local shard download complete')
