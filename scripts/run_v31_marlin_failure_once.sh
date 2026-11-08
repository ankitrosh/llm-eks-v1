#!/usr/bin/env bash
set -euo pipefail

REGION=us-east-2
CLUSTER=llm-pipeline-v1
NODEGROUP=gpu-a10g
ASG=eks-gpu-a10g-32d0c4bb-a5e4-bea9-6af3-ec95a70be2f2
REPOSITORY=qwen3-14b-pipeline
TAG=v3.1-gptq-marlin-r1
JOB=v31-marlin-failure
SERVICE=v31-marlin-failure
CM=v31-marlin-failure-scripts
LOG_DIR=test-results/v3_1/gate-c-failure-raise
LOG="$LOG_DIR/rank0.log"
LOG_R1="$LOG_DIR/rank1.log"

GPU_STARTED=0
CM_CREATED=0
SERVICE_CREATED=0
JOB_CREATED=0

cleanup() {
  rc=$?
  trap - EXIT INT TERM

  echo "========== COLLECT RESULTS =========="
  mkdir -p "$LOG_DIR"

  if [[ "$JOB_CREATED" == 1 ]]; then
    kubectl -n llm get pods \
      -l "batch.kubernetes.io/job-name=$JOB" \
      -o wide > "$LOG_DIR/pods.txt" 2>&1 || true

    kubectl -n llm describe job "$JOB" \
      > "$LOG_DIR/job-describe.txt" 2>&1 || true

    for rank in 0 1; do
      pod=$(kubectl -n llm get pods \
        -l "batch.kubernetes.io/job-name=$JOB" \
        -o json 2>/dev/null | python3 -c '
import json,sys
rank=sys.argv[1]
pods=json.load(sys.stdin)["items"]
for pod in pods:
    annotation=pod["metadata"].get("annotations",{})
    if annotation.get("batch.kubernetes.io/job-completion-index")==rank:
        print(pod["metadata"]["name"])
        break
' "$rank" 2>/dev/null || true)

      if [[ -n "$pod" ]]; then
        kubectl -n llm logs "pod/$pod" \
          --tail=-1 > "$LOG_DIR/rank${rank}.log" 2>&1 || true
        kubectl -n llm describe pod "$pod" \
          > "$LOG_DIR/pod-${rank}-describe.txt" 2>&1 || true
      else
        echo "No pod found for rank $rank" \
          > "$LOG_DIR/rank${rank}.log"
      fi
    done
    if [[ "$rc" == 0 ]]; then
      echo "========== VERIFY EXPECTED FAILURE =========="

      if grep -Fq 'PASS: Rank 0 received remote FATAL;' "$LOG" &&
         grep -Fq 'V31_INJECTED_EXECUTION_ERROR' "$LOG" &&
         grep -Fq 'PASS: Gate C.4a rank 0 completed' "$LOG" &&
         grep -Fq 'PASS: Rank 1 raised the injected error' "$LOG_R1" &&
         grep -Fq 'PASS: Gate C.4a rank 1 completed' "$LOG_R1"; then

        echo "PASS: Both ranks verified the injected failure"

      else
        echo "ERROR: Missing expected failure evidence"
        rc=1
      fi
    fi

    kubectl -n llm delete job "$JOB" \
      --ignore-not-found --wait=true \
      --timeout=120s || true
  fi

  if [[ "$CM_CREATED" == 1 ]]; then
    kubectl -n llm delete configmap "$CM" \
      --ignore-not-found || true
  fi

  if [[ "$SERVICE_CREATED" == 1 ]]; then
    kubectl -n llm delete service "$SERVICE" \
      --ignore-not-found || true
  fi

  if [[ "$GPU_STARTED" == 1 ]]; then
    echo "========== SCALE GPU TO ZERO =========="
    stopped=0

    for _ in $(seq 1 36); do
      status=$(aws eks describe-nodegroup \
        --cluster-name "$CLUSTER" \
        --nodegroup-name "$NODEGROUP" \
        --region "$REGION" \
        --query nodegroup.status \
        --output text 2>/dev/null || true)

      desired=$(aws eks describe-nodegroup \
        --cluster-name "$CLUSTER" \
        --nodegroup-name "$NODEGROUP" \
        --region "$REGION" \
        --query nodegroup.scalingConfig.desiredSize \
        --output text 2>/dev/null || true)

      if [[ "$status" == ACTIVE && "$desired" == 0 ]]; then
        stopped=1
        break
      fi

      if [[ "$status" == ACTIVE ]]; then
        if aws eks update-nodegroup-config \
          --cluster-name "$CLUSTER" \
          --nodegroup-name "$NODEGROUP" \
          --region "$REGION" \
          --scaling-config \
          minSize=0,maxSize=2,desiredSize=0 \
          >/dev/null; then
          stopped=1
          break
        fi
      fi

      sleep 10
    done

    if [[ "$stopped" != 1 ]]; then
      echo "WARNING: Could not confirm scale-down request."
      rc=1
    fi

    echo "========== TERMINATION LIFECYCLE =========="

    hooks=$(aws autoscaling describe-lifecycle-hooks \
      --auto-scaling-group-name "$ASG" \
      --region "$REGION" \
      --query \
      "LifecycleHooks[?LifecycleTransition=='autoscaling:EC2_INSTANCE_TERMINATING'].LifecycleHookName" \
      --output text 2>/dev/null || true)

    for _ in $(seq 1 60); do
      state=$(aws autoscaling describe-auto-scaling-groups \
        --auto-scaling-group-names "$ASG" \
        --region "$REGION" \
        --query \
        'AutoScalingGroups[0].{desired:DesiredCapacity,instances:Instances[].{id:InstanceId,state:LifecycleState}}' \
        --output json 2>/dev/null || true)

      if [[ -z "$state" ]]; then
        sleep 10
        continue
      fi

      asg_desired=$(python3 -c \
        'import json,sys; print(json.load(sys.stdin)["desired"])' \
        <<< "$state")

      count=$(python3 -c \
        'import json,sys; print(len(json.load(sys.stdin)["instances"]))' \
        <<< "$state")

      if [[ "$asg_desired" == 0 && "$count" == 0 ]]; then
        echo "PASS: ASG desired=0, instances=[]"
        break
      fi

      if [[ "$asg_desired" == 0 && $(wc -w <<< "$hooks") == 1 ]]; then
        while read -r instance; do
          [[ -z "$instance" ]] && continue

          # Refuse lifecycle completion while application
          # Pods remain active in the cluster.
          active=$(kubectl get pods -A -o json |
            python3 -c '
import json,sys
pods=json.load(sys.stdin)["items"]
count=0
for p in pods:
    if p["metadata"]["namespace"] == "kube-system":
        continue
    if p.get("status",{}).get("phase") in ("Succeeded","Failed"):
        continue
    if not p.get("spec",{}).get("nodeName"):
        continue
    count += 1
print(count)
' 2>/dev/null || echo 999)

          if [[ "$active" == 0 ]]; then
            echo "Completing lifecycle action: $instance"

            aws autoscaling complete-lifecycle-action \
              --auto-scaling-group-name "$ASG" \
              --lifecycle-hook-name "$hooks" \
              --instance-id "$instance" \
              --lifecycle-action-result CONTINUE \
              --region "$REGION" >/dev/null || true
          else
            echo "WARNING: Active Pods remain; hook retained."
          fi
        done < <(
          python3 -c '
import json,sys
for i in json.load(sys.stdin)["instances"]:
    if i["state"] == "Terminating:Wait":
        print(i["id"])
' <<< "$state"
        )
      fi

      sleep 10
    done

    echo "========== FINAL AUTOSCALING STATE =========="
    final=$(aws autoscaling describe-auto-scaling-groups \
      --auto-scaling-group-names "$ASG" \
      --region "$REGION" \
      --query \
      'AutoScalingGroups[0].{desired:DesiredCapacity,instances:Instances[].{id:InstanceId,state:LifecycleState}}' \
      --output json)

    echo "$final"

    if ! python3 -c '
import json,sys
d=json.load(sys.stdin)
assert d["desired"] == 0 and d["instances"] == []
' <<< "$final"; then
      echo "WARNING: Verify remaining ASG instances."
      rc=1
    fi
  fi

  echo "Rank 0 results: $LOG"
  echo "Rank 1 results: $LOG_R1"
  [[ ! -s "$LOG" ]] || tail -n 60 "$LOG"
  [[ ! -s "$LOG_R1" ]] || tail -n 30 "$LOG_R1"

  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


echo "========== GATE C PREFLIGHT =========="

test -f scripts/v3_1_qwen3_distributed_marlin.py
test -f scripts/v3_1_marlin_failure_injection.py
test -f scripts/v3_1_qwen3_independent_reference.py
test -f src/llm_runtime/v3_1_nrank_transport.py
test -f deploy/manifests/v3_1_marlin_failure_injection.yaml

python3 -m py_compile \
  scripts/v3_1_qwen3_distributed_marlin.py \
  scripts/v3_1_qwen3_independent_reference.py \
  scripts/v3_1_marlin_failure_injection.py

sha256sum -c docs/V3_FREEZE_SHA256_2026-11-07.txt

NG_STATUS=$(aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --region "$REGION" \
  --query nodegroup.status --output text)

NG_DESIRED=$(aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --region "$REGION" \
  --query nodegroup.scalingConfig.desiredSize \
  --output text)

INSTANCE_COUNT=$(aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names "$ASG" \
  --region "$REGION" \
  --query 'length(AutoScalingGroups[0].Instances)' \
  --output text)

[[ "$NG_STATUS" == ACTIVE ]] || {
  echo "STOP: Node group is not ACTIVE"
  exit 1
}

[[ "$NG_DESIRED" == 0 && "$INSTANCE_COUNT" == 0 ]] || {
  echo "STOP: GPU instances already exist"
  exit 1
}

for kind in job service configmap; do
  if kubectl -n llm get "$kind" \
    "$(if [[ "$kind" == configmap ]]; then echo "$CM"; else echo "$JOB"; fi)" \
    >/dev/null 2>&1; then
    echo "STOP: Gate C $kind already exists"
    exit 1
  fi
done

DIGEST=$(aws ecr describe-images \
  --repository-name "$REPOSITORY" \
  --image-ids imageTag="$TAG" \
  --region "$REGION" \
  --query 'imageDetails[0].imageDigest' \
  --output text)

[[ "$DIGEST" == sha256:* ]] || {
  echo "STOP: Invalid image digest"
  exit 1
}

IMAGE="657173565477.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY}@${DIGEST}"

echo "Pinned image: $IMAGE"

echo "========== PREPARE SOURCE OVERLAY =========="

cat > /tmp/v31-marlin-init.py <<'PYINIT'
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
PYINIT

kubectl -n llm create configmap "$CM" \
  --from-file=v3_1_qwen3_distributed_marlin.py=scripts/v3_1_qwen3_distributed_marlin.py \
  --from-file=v3_1_qwen3_independent_reference.py=scripts/v3_1_qwen3_independent_reference.py \
  --from-file=v3_1_marlin_failure_injection.py=scripts/v3_1_marlin_failure_injection.py \
  --from-file=v3_1_nrank_transport.py=src/llm_runtime/v3_1_nrank_transport.py \
  --from-file=__init__.py=/tmp/v31-marlin-init.py \
  --dry-run=client -o yaml | kubectl apply -f -

CM_CREATED=1

echo "========== PIN MANIFEST IMAGE =========="

python3 - "$IMAGE" <<'PYIMAGE'
from pathlib import Path
import sys

source = Path(
    "deploy/manifests/v3_1_marlin_failure_injection.yaml"
).read_text()

old = (
    "657173565477.dkr.ecr.us-east-2.amazonaws.com/"
    "qwen3-14b-pipeline:v3.1-gptq-marlin-r1"
)

assert source.count(old) == 1

target = Path("/tmp/v31-marlin-failure-pinned.yaml")
target.write_text(source.replace(old, sys.argv[1]))
print("PASS: Immutable image digest pinned")
PYIMAGE

kubectl apply --dry-run=client \
  -f /tmp/v31-marlin-failure-pinned.yaml >/dev/null

echo "========== CREATE HEADLESS SERVICE =========="

awk '/^---$/{exit} {print}' \
  /tmp/v31-marlin-failure-pinned.yaml \
  > /tmp/v31-marlin-failure-service.yaml

SERVICE_CREATED=1
kubectl apply -f /tmp/v31-marlin-failure-service.yaml

echo "========== START TWO A10Gs =========="

GPU_STARTED=1

aws eks update-nodegroup-config \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --region "$REGION" \
  --scaling-config minSize=0,maxSize=2,desiredSize=2 \
  >/dev/null

echo "========== WAIT FOR BOTH GPU NODES =========="

ready=0

for _ in $(seq 1 90); do
  nodes=$(kubectl get nodes \
    -l "eks.amazonaws.com/nodegroup=$NODEGROUP" \
    -o json 2>/dev/null | python3 -c '
import json,sys
data=json.load(sys.stdin)
ready=0
for node in data["items"]:
    conditions=node["status"].get("conditions",[])
    is_ready=any(
        c["type"]=="Ready" and c["status"]=="True"
        for c in conditions
    )
    gpu=node["status"].get("allocatable",{}).get(
        "nvidia.com/gpu"
    )
    if is_ready and gpu=="1" and not node["spec"].get(
        "unschedulable",False
    ):
        ready+=1
print(ready)
' 2>/dev/null || echo 0)

  inservice=$(aws autoscaling describe-auto-scaling-groups \
    --auto-scaling-group-names "$ASG" \
    --region "$REGION" \
    --query \
    "length(AutoScalingGroups[0].Instances[?LifecycleState=='InService'])" \
    --output text 2>/dev/null || echo 0)

  if [[ "$nodes" == 2 && "$inservice" == 2 ]]; then
    echo "PASS: Two Ready A10G nodes"
    ready=1
    break
  fi

  sleep 10
done

[[ "$ready" == 1 ]] || {
  echo "ERROR: Both A10Gs did not become ready"
  kubectl get nodes -o wide || true
  exit 1
}

echo "========== RUN DISTRIBUTED MARLIN JOB =========="

JOB_CREATED=1
kubectl apply -f /tmp/v31-marlin-failure-pinned.yaml

completed=0

for _ in $(seq 1 195); do
  succeeded=$(kubectl -n llm get job "$JOB" \
    -o jsonpath='{.status.succeeded}' \
    2>/dev/null || true)

  failed=$(kubectl -n llm get job "$JOB" \
    -o jsonpath='{.status.failed}' \
    2>/dev/null || true)

  if [[ "${succeeded:-0}" == 2 ]]; then
    completed=1
    echo "PASS: Both distributed rank pods succeeded"
    break
  fi

  if [[ "${failed:-0}" -gt 0 ]]; then
    echo "ERROR: A distributed rank failed"
    exit 1
  fi

  sleep 10
done

[[ "$completed" == 1 ]] || {
  echo "ERROR: Distributed test did not complete"
  exit 1
}

echo "PASS: Gate C.4a Job completed; verifying rank logs"
