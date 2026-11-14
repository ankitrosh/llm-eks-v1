# Historical V1–V3 root files

This directory retains files from earlier distributed
inference experiments.

## Organization

- `benchmarks/`: early benchmark clients and requests
- `kubernetes-experiments/`: historical cluster,
  networking, storage and Qwen3 manifests
- `fault-injection/`: early failure and recovery experiments

These are historical reference files, not current
public CLI entry points.

The active framework is in `src/llm_runtime/`.
Public usage is documented in `README.md`,
`docs/CLI_GUIDE.md` and `examples/`.

Historical commands may need path adjustments
before they can be reproduced.

Archiving files does not erase Git history or
sanitize infrastructure identifiers.

The original locations are preserved in Git history.
