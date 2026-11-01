#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:---check}"
NS="llm"

if [[ "$MODE" != "--check" && "$MODE" != "--apply" ]]; then
  echo "Usage: scripts/deploy.sh [--check|--apply]"
  exit 2
fi

required_files=(
  infrastructure/storageclass.yaml
  infrastructure/model-image.txt
  kubernetes/namespace.yaml
  kubernetes/qwen3-services.yaml
  kubernetes/qwen3-statefulset.yaml
  kubernetes/recovery-rbac.yaml
  kubernetes/recovery-state.yaml
  kubernetes/recovery.yaml
  kubernetes/coordinator.yaml
  src/coordinator.py
  src/recovery_controller.py
)

for file in "${required_files[@]}"; do
  if [[ ! -s "$file" ]]; then
    echo "ERROR: Missing or empty file: $file" >&2
    exit 1
  fi
done

echo "=== DEPLOYMENT MODE: $MODE ==="
echo "Kubernetes context:"
kubectl config current-context

echo
echo "=== Checking cluster prerequisites ==="

kubectl get csidriver ebs.csi.aws.com -o name

# Check node readiness and advertised GPU resources.
python3 - <<'PY'
import json
import subprocess

data = json.loads(subprocess.check_output(
    ["kubectl", "get", "nodes", "-o", "json"],
    text=True
))

cpu = 0
gpu = 0

for node in data["items"]:
    labels = node["metadata"].get("labels", {})
    ready = any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in node["status"].get("conditions", [])
    )

    if not ready:
        continue

    if labels.get("pool") == "cpu":
        cpu += 1

    if labels.get("pool") == "gpu":
        allocatable = node["status"].get("allocatable", {})
        if int(allocatable.get("nvidia.com/gpu", "0")) >= 1:
            gpu += 1

print(f"Ready CPU nodes: {cpu}")
print(f"Ready GPU nodes with allocatable GPUs: {gpu}")

if cpu < 1 or gpu < 2:
    raise SystemExit(
        "Insufficient Ready CPU/GPU nodes. "
        "Check GPU device plugin and node groups."
    )
PY

echo
echo "=== Validating Kubernetes manifests ==="

kubectl apply --dry-run=client \
  -f kubernetes/ \
  -f infrastructure/storageclass.yaml

kubectl create configmap qwen3-coordinator-source \
  -n "$NS" \
  --from-file=coordinator.py=src/coordinator.py \
  --dry-run=client -o yaml >/dev/null

kubectl create configmap qwen3-recovery-source \
  -n "$NS" \
  --from-file=recovery_controller.py=src/recovery_controller.py \
  --dry-run=client -o yaml >/dev/null

if [[ "$MODE" == "--check" ]]; then
  echo
  echo "CHECK PASSED: No resources were changed."
  echo "Use --apply only on a fresh application deployment."
  exit 0
fi

echo
echo "=== Refusing to overwrite an existing deployment ==="

for resource in \
  statefulset/qwen3 \
  deployment/qwen3-coordinator \
  deployment/qwen3-recovery \
  configmap/qwen3-recovery-state
do
  if kubectl get "$resource" -n "$NS" \
      -o name >/dev/null 2>&1; then
    echo "ERROR: Existing $resource detected."
    echo "This bootstrap script does not upgrade running deployments."
    exit 1
  fi
done

echo
echo "=== 1. Namespace and storage ==="

kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f infrastructure/storageclass.yaml

echo
echo "=== 2. GPU Services ==="

kubectl apply -f kubernetes/qwen3-services.yaml

echo
echo "=== 3. Application source ConfigMaps ==="

kubectl create configmap qwen3-coordinator-source \
  -n "$NS" \
  --from-file=coordinator.py=src/coordinator.py \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap qwen3-recovery-source \
  -n "$NS" \
  --from-file=recovery_controller.py=src/recovery_controller.py \
  --dry-run=client -o yaml | kubectl apply -f -

echo
echo "=== 4. Recovery permissions and initial state ==="

kubectl apply -f kubernetes/recovery-rbac.yaml

# Initial state is created exactly once during bootstrap.
# Never use this step to upgrade an existing deployment.
kubectl apply -f kubernetes/recovery-state.yaml

echo
echo "=== 5. Deploy inference workers ==="

kubectl apply -f kubernetes/qwen3-statefulset.yaml

echo "Waiting for both GPU workers."
echo "First-time model downloads may take significantly longer."

kubectl rollout status statefulset/qwen3 \
  -n "$NS" --timeout=3600s

kubectl wait \
  -n "$NS" \
  --for=condition=Ready \
  pod/qwen3-0 pod/qwen3-1 \
  --timeout=300s

echo
echo "=== 6. Start recovery controller ==="

kubectl apply -f kubernetes/recovery.yaml

kubectl rollout status deployment/qwen3-recovery \
  -n "$NS" --timeout=300s

controller_ready=false

for i in $(seq 1 60); do
  if kubectl exec -n "$NS" \
    deployment/qwen3-recovery -- python -c '
import json
import urllib.request

s = json.load(urllib.request.urlopen(
    "http://127.0.0.1:8081/status", timeout=5
))

assert s["phase"] == "SERVING", s
assert s["mode"] == "RUNNING", s
assert s["auto_recover"] is True, s
' >/dev/null 2>&1; then
    controller_ready=true
    break
  fi

  sleep 5
done

if [[ "$controller_ready" != "true" ]]; then
  echo "ERROR: Recovery controller did not reach SERVING."
  kubectl logs -n "$NS" deployment/qwen3-recovery --tail=100
  exit 1
fi

echo "Recovery controller: SERVING"

echo
echo "=== 7. Start CPU gateway ==="

kubectl apply -f kubernetes/coordinator.yaml

kubectl rollout status deployment/qwen3-coordinator \
  -n "$NS" --timeout=300s

echo
echo "=== 8. End-to-end smoke test ==="

kubectl exec -i -n "$NS" \
  deployment/qwen3-coordinator -- python - <<'PY'
import json
import urllib.request

payload = json.dumps({
    "prompt": "What is the capital of France?",
    "max_new_tokens": 24,
}).encode()

request = urllib.request.Request(
    "http://127.0.0.1:8080/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=240) as response:
    result = json.load(response)

assert result["generated_tokens"] > 0
assert "Paris" in result["text"]

print("Inference response:", result["text"])
print("E2E latency:", result["metrics"]["e2e_ms"], "ms")
print("PASS: End-to-end inference")
PY

echo
echo "=== DEPLOYMENT COMPLETE ==="

kubectl get pods -n "$NS" -o wide
kubectl get pvc -n "$NS"
