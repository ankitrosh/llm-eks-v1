"""Prepare isolated Qwen3-14B mixed-length and cache-reuse test.

No AWS or Kubernetes resources are created.
The successful throughput benchmark remains unchanged.
"""

from pathlib import Path
import py_compile
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_NAME = "v3-qwen3-14b-benchmark"
NAME = "v3-qwen3-14b-edgecases"
PORTS = {"rendezvous": 29668, "control": 29669}

BASE_RUNNER = ROOT / "scripts/v3_qwen3_14b_benchmark.py"
EDGE_RUNNER = ROOT / "scripts/v3_qwen3_14b_edgecases.py"

BASE_MANIFEST = ROOT / f"deploy/manifests/{BASE_NAME}.yaml"
EDGE_MANIFEST = ROOT / f"deploy/manifests/{NAME}.yaml"
EDGE_CM = Path(f"/tmp/{NAME}-code-configmap.yaml")

source = BASE_RUNNER.read_text()


import re

def replace_once(old, new):
    global source

    if source.count(old) == 1:
        source = source.replace(old, new, 1)
        return

    if old == 'counts = [len(result.tokens) for result in results]':
        matches = list(re.finditer(
            r'counts\s*=\s*\[\s*len\(result\.tokens\)\s+'
            r'for\s+result\s+in\s+results\s*\]',
            source,
        ))
        if len(matches) != 1:
            raise RuntimeError("Expected exactly one counts assignment")

        match = matches[0]
        source = (
            source[:match.start()]
            + match.group(0)
            + " + [len(repeated.tokens)]"
            + source[match.end():]
        )
        return

    if old.startswith('"concurrency": REQUESTS, "input_tokens_per_request":'):
        before = '"concurrency": REQUESTS,'
        if source.count(before) != 1:
            raise RuntimeError("Expected exactly one concurrency metric")

        source = source.replace(
            before,
            before + '\n                    "test_type": "mixed-length-cache-reuse",',
            1,
        )
        return

    if old.startswith('print("PASS: Qwen3-14B 20/20 four-request'):
        before = "PASS: Qwen3-14B 20/20 four-request NCCL benchmark"
        after = "PASS: Qwen3-14B mixed-length and cache-reuse edge cases"

        if source.count(before) != 1:
            raise RuntimeError("Expected exactly one success message")

        source = source.replace(before, after, 1)
        return

    raise RuntimeError(
        f"Expected once ({source.count(old)}): {old!r}"
    )

replace_once(
    "PROMPT_TOKENS = 128",
    "PROMPT_TOKENS = 1  # warm-up prompt length"
)
replace_once("NEW_TOKENS = 64", "NEW_TOKENS = 8")
replace_once("WARMUP_TOKENS = 16", "WARMUP_TOKENS = 2")

# Use a natural passage with varied content.
# Request lengths are 1, 8, 128, and 256 tokens.
# Shared prefixes also exercise independent KV caches.

start = source.index("def make_prompts(tokenizer):")
end = source.index("\ndef main():", start)

source = source[:start] + '''def make_prompts(tokenizer):
    """Mixed prompt lengths with shared prefixes."""
    passage = (
        "A coastal research team deployed environmental sensors after a storm. "
        "They wanted to understand why the northern marsh flooded before the southern bank. "
        "The first station measured sea pressure, wind direction and soil moisture. "
        "At the river mouth, a technician noted that floating debris moved against the expected tide. "
        "A second crew collected water samples beside a recently repaired wooden bridge. "
        "Their field notebook described a faulty radio antenna and an unusually warm current. "
        "During the afternoon, the crew changed the sampling interval to capture brief surges. "
        "A weather balloon recorded a sharp temperature inversion above the harbor. "
        "The scientists compared those readings against an earlier survey from dry weather. "
        "They discovered that a narrow drainage channel was blocked by a fallen tree. "
        "A local engineer suggested opening the channel only after confirming the downstream risk. "
        "Meanwhile, a graduate student wrote a simulation to estimate possible changes in flow. "
        "The model treated the bridges as barriers and the wetlands as temporary storage. "
        "Early results were surprising because the lowest field did not flood first. "
        "The team then inspected sediment buildup and found a hidden ridge beside the road. "
        "A drone flight documented the ridge without disturbing nearby nesting birds. "
        "After sunset, staff moved the instruments to higher ground and archived all measurements. "
        "The next morning they discussed which observations would be most useful for validation. "
        "Their report separated directly measured facts from uncertain causal explanations. "
        "One member proposed a second experiment using deliberately varied sensor placements. "
        "Another questioned whether yesterday's wind had biased their instruments. "
        "A third calculated how the sampling frequency would change energy consumption. "
        "The final plan included redundant logging, calibration checks and an independent audit. "
        "They published a summary explaining both the successes and limitations of the study. "
    )

    tokens = tokenizer.encode(
        passage,
        add_special_tokens=False,
    )

    if len(tokens) < 256:
        raise RuntimeError(
            f"Passage has {len(tokens)} tokens; need 256"
        )

    lengths = (1, 8, 128, 256)
    print(f"EDGE_PROMPT_LENGTHS={lengths}", flush=True)

    return [tokens[:n] for n in lengths]


''' + source[end:]

# 2 warm-up steps + 4*8 concurrent steps + 8 repeated steps.
old = "REQUESTS * NEW_TOKENS + WARMUP_TOKENS"

if source.count(old) < 2:
    raise RuntimeError(
        "Unexpected forward-count expression count"
    )

source = source.replace(
    old,
    "(REQUESTS + 1) * NEW_TOKENS + WARMUP_TOKENS"
)

replace_once(
    'assert stats["releases"] == REQUESTS + 1, stats',
    'assert stats["releases"] == REQUESTS + 2, stats'
)

# After the first four requests complete, wait for cache releases,
# then resubmit an identical prompt to verify deterministic output.

anchor = "                elapsed = time.perf_counter() - t0\n"

insert = '''                release_deadline = time.monotonic() + 30

                while link.stats()["releases"] < REQUESTS + 1:
                    if time.monotonic() > release_deadline:
                        raise RuntimeError(
                            "First-wave cache release timeout"
                        )
                    time.sleep(0.05)

                repeated = scheduler.submit(
                    torch.tensor(
                        [prompts[2]],
                        device=device,
                        dtype=torch.long,
                    ),
                    prompt_length=len(prompts[2]),
                    max_new_tokens=NEW_TOKENS,
                    timeout_s=600,
                ).result(timeout=660)

                assert repeated.tokens == results[2].tokens, (
                    "Cache contamination or nondeterministic decode",
                    repeated.tokens,
                    results[2].tokens,
                )

                print(
                    "PASS: repeated prompt matches after cache release",
                    flush=True,
                )
'''

replace_once(anchor, insert + anchor)

replace_once(
    'counts = [len(result.tokens) for result in results]',
    'counts = [len(result.tokens) for result in results] '
    '+ [len(repeated.tokens)]'
)

replace_once(
    'assert counts == [NEW_TOKENS] * REQUESTS, counts',
    'assert counts == [NEW_TOKENS] * (REQUESTS + 1), counts'
)

replace_once(
    'assert metrics["completed_requests"] == REQUESTS, metrics',
    'assert metrics["completed_requests"] == REQUESTS + 1, metrics'
)

replace_once(
    'assert metrics["generated_tokens"] == REQUESTS * NEW_TOKENS, metrics',
    'assert metrics["generated_tokens"] == '
    '(REQUESTS + 1) * NEW_TOKENS, metrics'
)

replace_once(
    '"input_tokens_per_request": PROMPT_TOKENS,',
    '"input_tokens_per_request": [len(p) for p in prompts],'
)

replace_once(
    '"concurrency": REQUESTS, "input_tokens_per_request":',
    '"concurrency": REQUESTS, '
    '"test_type": "mixed-length-cache-reuse", '
    '"input_tokens_per_request":'
)

replace_once(
    'print("PASS: Qwen3-14B 20/20 four-request NCCL benchmark", flush=True)',
    'print("PASS: Qwen3-14B mixed-length and cache-reuse edge cases", flush=True)'
)

EDGE_RUNNER.write_text(source)
py_compile.compile(str(EDGE_RUNNER), doraise=True)

# Generate a distinct ConfigMap and deployment.
# Original files are only read, never modified.

docs = list(yaml.safe_load_all(BASE_MANIFEST.read_text()))
assert len(docs) == 3

base_pod = docs[1]["spec"]["template"]["spec"]
base_code = next(
    v for v in base_pod["volumes"]
    if v["name"] == "test-code"
)

items = base_code["projected"]["sources"][0]["configMap"]["items"]

data = {}
for item in items:
    path = (
        EDGE_RUNNER
        if item["key"] == "runner"
        else ROOT / item["path"]
    )
    data[item["key"]] = path.read_text()

assert sum(len(x.encode("utf-8")) for x in data.values()) < 800_000

cm = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {
        "name": NAME + "-code",
        "namespace": "llm",
    },
    "data": data,
}

EDGE_CM.write_text(yaml.safe_dump(cm, sort_keys=False))

for doc in docs:
    doc["metadata"]["name"] = doc["metadata"]["name"].replace(
        BASE_NAME, NAME
    )

    if doc["kind"] == "Service":
        doc["spec"]["selector"]["app"] = NAME

        for port in doc["spec"]["ports"]:
            port["port"] = PORTS[port["name"]]
            port["targetPort"] = PORTS[port["name"]]

        continue

    assert doc["kind"] == "Job"

    template = doc["spec"]["template"]
    template["metadata"]["labels"]["app"] = NAME

    pod = template["spec"]
    pod["hostname"] = pod["hostname"].replace(BASE_NAME, NAME)
    pod["subdomain"] = pod["subdomain"].replace(BASE_NAME, NAME)

    selectors = pod["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]

    selectors[0]["labelSelector"]["matchLabels"]["app"] = NAME

    container = pod["containers"][0]
    container["command"][-1] = (
        "/workspace/v3_qwen3_14b_edgecases.py"
    )

    for env in container["env"]:
        if env["name"] == "MASTER_ADDR":
            env["value"] = env["value"].replace(BASE_NAME, NAME)
        elif env["name"] == "MASTER_PORT":
            env["value"] = str(PORTS["rendezvous"])
        elif env["name"] == "V3_CTRL_PORT":
            env["value"] = str(PORTS["control"])

    code = next(
        v for v in pod["volumes"]
        if v["name"] == "test-code"
    )

    projected = code["projected"]["sources"][0]["configMap"]
    projected["name"] = NAME + "-code"

    for item in projected["items"]:
        if item["key"] == "runner":
            item["path"] = "v3_qwen3_14b_edgecases.py"

EDGE_MANIFEST.write_text(
    yaml.safe_dump_all(docs, sort_keys=False)
)

print("PASS: Edge-case runner compiled")
print("PASS: Two rank-specific Jobs prepared")
print("PASS: Original benchmark files preserved")
print("Manifest:", EDGE_MANIFEST)
print("ConfigMap:", EDGE_CM)
