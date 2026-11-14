# Repository map

The repository separates the current framework, reference deployment, older custom runtime, and historical experiments.

```text
llm-eks-v1/
├── README.md                   Start here
├── pyproject.toml              Python package and CLI
├── src/
│   ├── llm_runtime/
│   │   ├── cli.py              Configure, plan, render, doctor, test
│   │   ├── inspection/         Model metadata and inspectors
│   │   ├── extensions/         Strategies, registry, policies
│   │   ├── backends/           vLLM planner and external adapters
│   │   ├── coordination/       CPU gateway and recovery logic
│   │   ├── v3_*.py             Custom pipeline experiments
│   │   ├── v3_1_*.py           Partitioning and transport work
│   │   └── v3_2_*.py           Batching and KV-cache experiments
│   └── gpu/                    V1/V2 custom Qwen3 runtime
├── examples/
│   ├── basic/                  Reference configuration
│   ├── model_swap/             Compatible model replacement
│   ├── custom_extensions/      User extension workflow
│   ├── v3_4_1/                Backend and vLLM scheduler examples
│   └── v3_5_cli/              Custom CLI backend example
├── scripts/                   Model, AWS, and manifest utilities
├── deploy/                    Kubernetes resources
├── infrastructure/            AWS and model profiles
├── tests/                     Regression and contract tests
├── benchmarks/                Benchmark data and scripts
├── docs/                      Guides and milestone reports
└── research/archive/          Historical logs and backups
```

## Where to start

**Use the framework**

- `README.md`
- `examples/basic/README.md`
- `docs/CLI_GUIDE.md`

**Change a model**

- `examples/model_swap/README.md`
- `src/llm_runtime/inspection/`

**Customize execution**

- `examples/custom_extensions/README.md`
- `src/llm_runtime/extensions/`
- `src/llm_runtime/backends/`

**Understand the reference deployment**

- `src/llm_runtime/coordination/`
- `src/llm_runtime/backends/vllm/`
- `deploy/v3_3/rendered/`

**Explore earlier optimizations**

- `src/gpu/`
- `src/llm_runtime/v3_2_*.py`
- `docs/V2_MILESTONE_REPORT.md`

**Understand validation**

- [Test guide](TEST_GUIDE.md)
- [Benchmark procedure](BENCHMARK.md)

## Current and historical code

The current reference serves inference through vLLM.

The custom PyTorch/NCCL runtime remains in the repository to preserve the original experiments and allow future development.

The presence of an older runtime implementation does not mean it is used by the active vLLM reference deployment.
