#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

NS=llm
REGION=us-east-2
CLUSTER=llm-pipeline-v1

V3_IMAGE="657173565477.dkr.ecr.us-east-2.amazonaws.com/qwen3-14b-pipeline@sha256:514663c208715815ed3516b2dbce2e94158207ffd7adf11348449d95441fd747"

OUT="infrastructure/profiles/v3-regression-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
echo "Results directory: $OUT"

cleanup() {
  status=$?
  trap - EXIT
  echo "========== AUTOMATIC SHUTDOWN =========="
  bash scripts/v3_regression_shutdown.sh || status=1
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "========== PREFLIGHT =========="

test "$(kubectl config current-context)" = \
  "arn:aws:eks:us-east-2:657173565477:cluster/llm-pipeline-v1"

test "$(kubectl -n "$NS" get sts qwen3 \
  -o jsonpath='{.spec.replicas}')" = "0"

test "$(kubectl -n "$NS" get sts qwen3 \
  -o jsonpath='{.spec.podManagementPolicy}')" = "Parallel"

CURRENT_IMAGE=$(kubectl -n "$NS" get sts qwen3 \
  -o jsonpath='{.spec.template.spec.containers[?(@.name=="runtime")].image}')

test "$CURRENT_IMAGE" = "$V3_IMAGE"

for NG in cpu-system gpu-a10g; do
  DESIRED=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NG" \
    --region "$REGION" \
    --query 'nodegroup.scalingConfig.desiredSize' \
    --output text)

  test "$DESIRED" = "0"
done

for DEP in qwen3-coordinator qwen3-recovery \
           qwen3-gpu-observer qwen3-maintenance-supervisor; do
  test "$(kubectl -n "$NS" get deploy "$DEP" \
    -o jsonpath='{.spec.replicas}')" = "0"
done

echo "PASS: Isolated regression preflight"

wait_pool() {
  local pool="$1"
  local required="$2"

  for i in $(seq 1 120); do
    count=$(kubectl get nodes -l "pool=$pool" -o json |
      python3 -c '
import json,sys
data=json.load(sys.stdin)
pool=sys.argv[1]
count=0
for n in data["items"]:
    ready=any(
        c["type"]=="Ready" and c["status"]=="True"
        for c in n["status"].get("conditions",[])
    )
    gpu=int(n["status"].get("allocatable",{}).get("nvidia.com/gpu","0"))
    if ready and (pool!="gpu" or gpu>=1):
        count+=1
print(count)
' "$pool")

    if [ "$count" -ge "$required" ]; then
      echo "PASS: $pool Ready nodes = $count"
      return 0
    fi

    sleep 10
  done

  echo "ERROR: $pool capacity not Ready"
  return 1
}

stop_ranks() {
  kubectl -n "$NS" scale sts/qwen3 --replicas=0

  for i in $(seq 1 120); do
    pods=$(kubectl -n "$NS" get pods \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' |
      grep -E '^qwen3-[01]$' || true)

    if [ -z "$pods" ]; then
      return 0
    fi

    sleep 5
  done

  return 1
}

start_ranks() {
  kubectl -n "$NS" scale sts/qwen3 --replicas=2

  echo "========== WAIT FOR BOTH RANKS =========="

  for i in $(seq 1 90); do
    READY=$(kubectl -n "$NS" get sts qwen3 \
      -o jsonpath='{.status.readyReplicas}')

    if [[ "${READY:-0}" == "2" ]]; then
      echo "PASS: Both Qwen3 ranks Ready"
      kubectl -n "$NS" get pods qwen3-0 qwen3-1 -o wide
      return 0
    fi

    sleep 10
  done

  echo "ERROR: Qwen3 ranks not Ready"
  kubectl -n "$NS" get pods -o wide
  kubectl -n "$NS" describe pods qwen3-0 qwen3-1 || true
  return 1
}

probe() {
  local output="$1"

  kubectl -n "$NS" exec -i qwen3-0 -c runtime \
    -- python - \
    < scripts/v3_regression_probe.py \
    > "$output"

  python3 -m json.tool "$output" >/dev/null
  echo "PASS: Inference saved to $output"
}

echo "========== START CPU NODE =========="

aws eks update-nodegroup-config \
  --cluster-name "$CLUSTER" \
  --nodegroup-name cpu-system \
  --region "$REGION" \
  --scaling-config minSize=0,maxSize=1,desiredSize=1 \
  --query 'update.id' --output text

wait_pool cpu 1

echo "========== START TWO A10Gs =========="

aws eks update-nodegroup-config \
  --cluster-name "$CLUSTER" \
  --nodegroup-name gpu-a10g \
  --region "$REGION" \
  --scaling-config minSize=0,maxSize=2,desiredSize=2 \
  --query 'update.id' --output text

wait_pool gpu 2

echo "========== RUN A: LEGACY =========="

start_ranks
probe "$OUT/legacy.json"

echo "========== RUN B: V3 ADAPTER =========="

stop_ranks

kubectl -n "$NS" set env sts/qwen3 \
  --containers=runtime \
  LLM_RUNTIME_PLAN=/app/v3/plans/qwen3-v3-no-cache.json \
  LLM_STAGE_IMPL=v3_adapter

start_ranks

for rank in 0 1; do
  kubectl -n "$NS" logs "qwen3-$rank" -c runtime |
    grep 'V3 runtime plan accepted' |
    tail -1
done

probe "$OUT/v3-no-cache.json"

echo "========== COMPARE TOKEN IDS =========="

python3 - "$OUT/legacy.json" "$OUT/v3-no-cache.json" <<'PY'
import json,sys

a=json.load(open(sys.argv[1]))
b=json.load(open(sys.argv[2]))

if a["generated_ids"] != b["generated_ids"]:
    print("FAIL: Token IDs differ")
    print("Legacy:", a["generated_ids"])
    print("V3:", b["generated_ids"])
    raise SystemExit(1)

print("PASS: Exact generated token IDs match")
print("Generated tokens:", len(a["generated_ids"]))
print("Legacy metrics:", json.dumps(a.get("metrics",{})))
print("V3 metrics:", json.dumps(b.get("metrics",{})))
PY

echo "========== RUN C: V3 PREFIX CACHE =========="

stop_ranks

kubectl -n "$NS" set env sts/qwen3 \
  --containers=runtime \
  LLM_RUNTIME_PLAN=/app/v3/plans/qwen3-v3-runtime.json \
  LLM_STAGE_IMPL=v3_adapter

start_ranks

probe "$OUT/v3-prefix-cold.json"
probe "$OUT/v3-prefix-warm.json"

python3 - "$OUT/v3-prefix-cold.json" "$OUT/v3-prefix-warm.json" <<'PY'
import json,sys

cold=json.load(open(sys.argv[1]))
warm=json.load(open(sys.argv[2]))

a=cold.get("metrics",{})
b=warm.get("metrics",{})

print("Cold TTFT:", a.get("ttft_ms"))
print("Warm TTFT:", b.get("ttft_ms"))
print("Cold cache hit:", a.get("prefix_cache_hit"))
print("Warm cache hit:", b.get("prefix_cache_hit"))
print("Reused tokens:", b.get("prefix_cached_tokens"))

assert a.get("prefix_cache_hit") is False
assert b.get("prefix_cache_hit") is True
assert b.get("prefix_cached_tokens",0) > 0
assert cold["generated_ids"] == warm["generated_ids"]

print("PASS: Prefix cache reused tokens and preserved output")
PY

echo "========== REGRESSION COMPLETED =========="
echo "Results: $OUT"
echo "Shutdown runs automatically on exit."
