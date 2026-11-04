from pathlib import Path

source = Path("src/gpu/qwen3_serve.py")
original = source.read_text()
code = original

def replace_once(old, new):
    global code
    count = code.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one occurrence, found {count}: {old[:90]!r}"
        )
    code = code.replace(old, new, 1)


replace_once(
    "from qwen3_stage_loader import Qwen3Stage\n",
    "from qwen3_stage_loader import Qwen3Stage\n"
    "from prefix_cache import PrefixCache\n"
    "from prefix_kv import snapshot_prefix, restore_prefix\n",
)

helpers = '''
_PREFIX_MANAGER = None


def _prefix_enabled() -> bool:
    return os.environ.get(
        "LLM_PREFIX_CACHE_ENABLED", "0"
    ).lower() in ("1", "true", "yes")


def _prefix_manager():
    """One cache per process; inference runs on the main thread."""
    global _PREFIX_MANAGER

    if not _prefix_enabled():
        return None

    if _PREFIX_MANAGER is None:
        _PREFIX_MANAGER = PrefixCache(
            max_bytes=int(os.environ.get(
                "LLM_PREFIX_CACHE_MAX_MIB", "256"
            )) * 1024 * 1024,
            max_entries=int(os.environ.get(
                "LLM_PREFIX_CACHE_MAX_ENTRIES", "4"
            )),
            min_reuse_tokens=int(os.environ.get(
                "LLM_PREFIX_CACHE_MIN_TOKENS", "64"
            )),
        )

    return _PREFIX_MANAGER


def _prefix_model_key(stage, rank):
    """Identify the loaded local model partition."""
    cfg = stage.cfg
    return (
        f"{getattr(cfg, '_name_or_path', 'unknown')}:"
        f"{cfg.num_hidden_layers}:{cfg.hidden_size}:"
        f"{stage.start}:{len(stage.layers)}:"
        f"{stage.dtype}:rank{rank}:"
        f"{os.environ.get('LLM_GPTQ_BACKEND', 'tritonv2')}"
    )


def _prepare_request_cache(stage, rank, prompt, prompt_len,
                           manager, empty_cache):
    """Agree on the reusable prefix across both NCCL ranks."""

    # Both ranks must have the feature enabled or disabled.
    flag = torch.tensor(
        [int(manager is not None)],
        device="cuda:0",
        dtype=torch.int64,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)

    if int(flag.item()) not in (0, WORLD_SIZE):
        raise RuntimeError(
            "LLM_PREFIX_CACHE_ENABLED differs between ranks"
        )

    if manager is None:
        return empty_cache, None, None, 0

    # Rank 1 needs the exact token IDs for cache lookup.
    # This transfers token IDs, not KV tensors.
    if rank == 0:
        ids_gpu = prompt.reshape(-1).contiguous()
    else:
        ids_gpu = torch.empty(
            prompt_len, device="cuda:0", dtype=torch.long
        )

    dist.broadcast(ids_gpu, src=0)
    token_ids = tuple(ids_gpu.cpu().tolist())

    model_key = _prefix_model_key(stage, rank)
    namespace = os.environ.get(
        "LLM_PREFIX_CACHE_NAMESPACE", "default"
    )

    hit = manager.lookup(
        token_ids,
        model_key=model_key,
        namespace=namespace,
    )
    local_reuse = hit.matched_tokens if hit else 0

    # The minimum guarantees both ranks have the entire prefix.
    agreed = torch.tensor(
        [local_reuse], device="cuda:0", dtype=torch.int64
    )
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    reuse = int(agreed.item())

    if reuse == 0:
        return empty_cache, token_ids, model_key, 0

    # If restoring fails on either stage, BOTH stages fall back.
    restored = None
    success = 1

    try:
        if hit is None:
            raise RuntimeError("Local prefix unavailable")
        restored = restore_prefix(
            stage.make_cache, hit.payload, reuse
        )
    except Exception as exc:
        success = 0
        LOG.warning(
            "rank=%s KV restore failed: %s", rank, exc
        )
        manager.clear()

    confirmed = torch.tensor(
        [success], device="cuda:0", dtype=torch.int64
    )
    dist.all_reduce(confirmed, op=dist.ReduceOp.MIN)

    if int(confirmed.item()) == 0:
        return empty_cache, token_ids, model_key, 0

    LOG.info("rank=%s reused %d prefix tokens", rank, reuse)

    return restored, token_ids, model_key, reuse


def _save_prompt_prefix(stage, rank, cache, token_ids,
                        model_key, manager):
    """Store prompt-only KV state, never generated-token state."""

    start = stage.start
    end = start + len(stage.layers)
    namespace = os.environ.get(
        "LLM_PREFIX_CACHE_NAMESPACE", "default"
    )

    try:
        # Estimate snapshot memory before cloning GPU tensors.
        estimated = 0
        for index in range(start, end):
            layer = cache.layers[index]
            if type(layer).__name__ != "DynamicLayer":
                return False
            if cache.get_seq_length(index) != len(token_ids):
                return False

            estimated += (
                layer.keys.numel() * layer.keys.element_size()
                + layer.values.numel() * layer.values.element_size()
            )

        if estimated > manager.max_bytes:
            return False

        # Preserve headroom for decoding and temporary allocations.
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < estimated + 512 * 1024 * 1024:
            LOG.info(
                "rank=%s skipping KV snapshot: memory headroom",
                rank,
            )
            return False

        snapshot = snapshot_prefix(
            cache, start, end, len(token_ids)
        )

        return manager.put(
            token_ids,
            snapshot,
            nbytes=snapshot.nbytes,
            model_key=model_key,
            namespace=namespace,
        )

    except Exception as exc:
        # A cache-storage failure must not corrupt active inference.
        LOG.warning(
            "rank=%s prefix snapshot skipped: %s", rank, exc
        )
        manager.clear()
        return False


'''

replace_once(
    "WORLD_SIZE = 2\n",
    "WORLD_SIZE = 2\n\n" + helpers,
)

replace_once(
    '    finish_reason = "length"\n',
    '''    finish_reason = "length"

    prefix_manager = _prefix_manager()
    prefix_stored = False

    cache, prefix_ids, prefix_model_key, prefix_reused = (
        _prepare_request_cache(
            stage, rank, prompt, prompt_len,
            prefix_manager, cache,
        )
    )
''',
)

replace_once(
    "position = 0 if step == 0 else prompt_len + step - 1",
    "position = prefix_reused if step == 0 else prompt_len + step - 1",
)

replace_once(
    "sequence_len = prompt_len if step == 0 else 1",
    "sequence_len = prompt_len - prefix_reused if step == 0 else 1",
)

replace_once(
    "input_ids = prompt if step == 0 else previous_token.view(1, 1)",
    "input_ids = prompt[:, prefix_reused:] if step == 0 else previous_token.view(1, 1)",
)

replace_once(
    "            completion = now_ms()\n",
    '''            # Snapshot immediately after prefill, before any
            # generated token is appended to the active KV cache.
            if step == 0 and prefix_manager is not None:
                prefix_stored = _save_prompt_prefix(
                    stage, rank, cache, prefix_ids,
                    prefix_model_key, prefix_manager,
                )

            completion = now_ms()
''',
)

replace_once(
    '                    "e2e_ms": round(total_ms, 3),\n',
    '''                    "prefix_cache_enabled": prefix_manager is not None,
                    "prefix_cache_hit": prefix_reused > 0,
                    "prefix_cached_tokens": prefix_reused,
                    "prefix_cache_stored": prefix_stored,
                    "prefix_cache_entries": (
                        prefix_manager.size if prefix_manager else 0
                    ),
                    "prefix_cache_used_mib": (
                        round(prefix_manager.used_bytes / 2**20, 2)
                        if prefix_manager else 0
                    ),
                    "prefill_gpu_ms": round(stage_compute_ms[0], 3),
                    "e2e_ms": round(total_ms, 3),
''',
)

# Only write after all expected edits were found.
backup = Path(
    "infrastructure/backups/qwen3_serve_before_prefix.py"
)
if not backup.exists():
    backup.write_text(original)

compile(code, str(source), "exec")
source.write_text(code)

print("PASS: Serving integration patched")
print("PASS: Python syntax validated")
print("Backup:", backup)
print("Updated:", source)
