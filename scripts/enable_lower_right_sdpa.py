from pathlib import Path

path = Path("src/gpu/qwen3_stage_loader.py")
source = path.read_text()

old = '''        if use_unmasked_sdpa:
            mask = None
        else:
            key = torch.arange(
'''

new = '''        prefix_mask = os.environ.get(
            "LLM_SDPA_PREFIX_MASK", "explicit"
        ).strip().lower()

        if prefix_mask not in {"explicit", "lower_right"}:
            raise ValueError(
                f"Unsupported SDPA prefix mask: {prefix_mask}"
            )

        use_lower_right = (
            self.cfg._attn_implementation == "sdpa"
            and start_position > 0
            and q > 1
            and prefix_mask == "lower_right"
        )

        if use_unmasked_sdpa:
            mask = None
        elif use_lower_right:
            from torch.nn.attention.bias import causal_lower_right

            mask = causal_lower_right(
                q, start_position + q
            )
        else:
            key = torch.arange(
'''

if source.count(old) != 1:
    raise RuntimeError(
        "Expected attention-mask block not found exactly once"
    )

backup = Path(
    "infrastructure/backups/"
    "qwen3_stage_loader_before_lower_right.py"
)
backup.parent.mkdir(parents=True, exist_ok=True)

if not backup.exists():
    backup.write_text(source)

updated = source.replace(old, new, 1)
compile(updated, str(path), "exec")
path.write_text(updated)

print("PASS: Optional lower-right SDPA mask added")
print("Default remains explicit masking")
