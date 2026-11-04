"""Download only the checkpoint shards needed by one Qwen3-14B rank."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download

p = argparse.ArgumentParser()
p.add_argument('--rank', type=int, required=True, choices=[0, 1])
p.add_argument('--split', type=int, default=20)
p.add_argument('--revision', default='main')
p.add_argument('--repo', default='Qwen/Qwen3-14B')
p.add_argument('--tokenizer-repo', default=None)
p.add_argument('--tokenizer-revision', default='main')
p.add_argument('--model-dir', default='models/qwen3-14b')
a = p.parse_args()

root = Path(a.model_dir)
root.mkdir(parents=True, exist_ok=True)

sha = HfApi().model_info(a.repo, revision=a.revision).sha

metadata = [
    'config.json',
    'model.safetensors.index.json',
    'tokenizer*',
    'chat_template.jinja',
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

# Resolve the chat template before starting inference.
# Most checkpoints include it themselves. Some quantized
# checkpoints require the template from their base model.
tokenizer_source = a.repo
tokenizer_sha = sha

tokenizer_config_path = root / 'tokenizer_config.json'
if not tokenizer_config_path.is_file():
    raise RuntimeError('Missing tokenizer_config.json')

tokenizer_config = json.loads(tokenizer_config_path.read_text())
has_template = (
    bool(tokenizer_config.get('chat_template'))
    or (root / 'chat_template.jinja').is_file()
)

if not has_template:
    if not a.tokenizer_repo:
        raise RuntimeError(
            'Tokenizer has no chat template. '
            'Specify --tokenizer-repo with a compatible base model.'
        )

    tokenizer_source = a.tokenizer_repo
    tokenizer_sha = HfApi().model_info(
        tokenizer_source,
        revision=a.tokenizer_revision,
    ).sha

    upstream_path = hf_hub_download(
        repo_id=tokenizer_source,
        revision=tokenizer_sha,
        filename='tokenizer_config.json',
    )

    upstream_config = json.loads(
        Path(upstream_path).read_text()
    )
    template = upstream_config.get('chat_template')

    if not isinstance(template, str) or not template.strip():
        raise RuntimeError(
            'Base model tokenizer_config.json '
            'does not contain a usable chat template'
        )

    target = root / 'chat_template.jinja'
    temporary = root / '.chat_template.jinja.tmp'
    temporary.write_text(template)
    temporary.replace(target)

print(
    f'chat_template_source={tokenizer_source} '
    f'revision={tokenizer_sha}',
    flush=True,
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
        'tokenizer_repo': tokenizer_source,
        'tokenizer_sha': tokenizer_sha,
        'shards': shards,
        'owned_tensor_count': len(keys),
    }, indent=2) + '\n'
)

print('PASS: rank-local shard download complete')
