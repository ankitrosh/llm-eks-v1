"""Create an isolated vLLM recovery controller from V1."""

import ast
import json
from pathlib import Path

src = Path(
    "infrastructure/backups/v3-3-cpu/recovery_controller.py"
)
assert src.is_file(), "Missing original recovery backup"
s = src.read_text()

old_header = (
    '"""Durable V1 recovery controller '
    'for the two-rank NCCL pipeline."""'
)
assert s.count(old_header) == 1
s = s.replace(
    old_header,
    '"""Durable V3.3 recovery controller for vLLM PP=2."""',
)

old_state = "configmaps/qwen3-recovery-state"
assert s.count(old_state) == 1
s = s.replace(old_state, "configmaps/v33-recovery-state")

for old in (
    'NAME = "qwen3"\n',
    'SCALE = f"/apis/apps/v1/namespaces/{NS}/'
    'statefulsets/{NAME}/scale"\n',
):
    assert s.count(old) == 1, f"Expected: {old!r}"
    s = s.replace(old, "")

begin = s.index("URLS = [\n")
end = s.index("]\n", begin) + len("]\n")
assert "qwen3-headless" in s[begin:end]
s = s[:begin] + s[end:]

anchor = "from urllib.request import Request, urlopen\n"
assert s.count(anchor) == 1
s = s.replace(
    anchor,
    anchor
    + """
# Both source files must be mounted into
# the same directory in the V3.3 pod.
from vllm_engine_controller import KubernetesVLLMEngineController
""",
)

start = s.index("def scale_get():")
end = s.index("class Watchdog:", start)

old_helpers = s[start:end]
for name in ("scale_set", "pod", "inspect"):
    assert f"def {name}(" in old_helpers

replacement = '''def vllm_api_healthy():
    url = os.getenv(
        "V33_VLLM_HEALTH_URL",
        "http://v33-vllm-rank0.llm.svc.cluster.local:8000/health",
    )

    try:
        with urlopen(url, timeout=2) as response:
            return response.status == 200

    except (HTTPError, URLError, TimeoutError, OSError):
        return False


engine = KubernetesVLLMEngineController(
    kube=kube,
    api_healthy=vllm_api_healthy,
    namespace=NS,
    name="v33-vllm",
)


def scale_get():
    return engine.scale_get()


def scale_set(replicas):
    return engine.scale_set(replicas)


def inspect():
    return engine.inspect()


'''

s = s[:start] + replacement + s[end:]

old_wait = (
    'if not pod(0)["exists"] '
    'and not pod(1)["exists"]:'
)
assert s.count(old_wait) == 1
s = s.replace(old_wait, "if engine.pods_absent():")

s = s.replace("NCCL group", "vLLM engine")

assert (
    'AUTO_RECOVER = os.getenv("AUTO_RECOVER", "false")'
    in s
)
assert "statefulsets/{NAME}" not in s
assert "qwen3-headless" not in s
assert "qwen3-recovery-state" not in s

ast.parse(s)

output = Path(
    "src/llm_runtime/coordination/v33_recovery_controller.py"
)
output.write_text(s)

state = {
    "mode": "PAUSED",
    "phase": "PAUSED",
    "generation": 0,
    "recoveries": 0,
    "attempt": 0,
    "baseline": None,
    "started_at": None,
    "last_error": None,
}

manifest = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {
        "name": "v33-recovery-state",
        "namespace": "llm",
    },
    "data": {
        "state": json.dumps(state, sort_keys=True),
    },
}

path = Path("deploy/v3_3/v33-recovery-state.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2) + "\n")

print("PASS: Generated isolated V3.3 recovery controller")
print("PASS: Created separate PAUSED recovery state")
print("PASS: Old StatefulSet references removed")
print("NO LIVE RESOURCES MODIFIED")
