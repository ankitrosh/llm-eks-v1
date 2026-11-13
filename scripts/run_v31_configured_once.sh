#!/usr/bin/env bash
set -euo pipefail

REGION=us-east-2
CLUSTER=llm-pipeline-v1
NODEGROUP=gpu-a10g
ASG=eks-gpu-a10g-32d0c4bb-a5e4-bea9-6af3-ec95a70be2f2
REPOSITORY=qwen3-14b-pipeline
TAG=v3.1-gptq-marlin-r1
JOB=v31-configured
SERVICE=v31-configured
CM=v31-configured-scripts
LOG_DIR=test-results/v3_1/configured-nccl
LOG="$LOG_DIR/rank0.log"
LOG_R1="$LOG_DIR/rank1.log"

GPU_STARTED=0
CM_CREATED=0
MODEL_CM=v31-configured-model
MODEL_CM_CREATED=0
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
    kubectl -n llm delete job "$JOB" \
      --ignore-not-found --wait=true \
      --timeout=120s || true
  fi

  if [[ "$MODEL_CM_CREATED" == 1 ]]; then
    kubectl -n llm delete configmap "$MODEL_CM" \
      --ignore-not-found || true
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


cd "$HOME/llm-eks-v1"

echo "========== CONFIGURED V3.1 PREFLIGHT =========="

sha256sum -c docs/V3_FREEZE_SHA256_2026-11-07.txt

python3 -m py_compile scripts/v3_1_configured_nccl.py

python3 - <<'PYCHECK'
import json
from pathlib import Path
import yaml

config = json.loads(Path(
    "configs/v3_1/qwen3_0_6b_gptq_nccl_n2.json"
).read_text())

docs = list(yaml.safe_load_all(Path(
    "deploy/templates/v3_1_indexed_nccl_job.yaml"
).read_text()))

assert len(docs) == 2
service, job = docs
assert service["kind"] == "Service"
assert job["kind"] == "Job"

assert service["metadata"]["name"] == "v31-configured"
assert job["metadata"]["name"] == "v31-configured"

ranks = config["pipeline"]["num_ranks"]
assert ranks == 2, "This AWS lifecycle runner supports 2 A10Gs"

spec = job["spec"]
assert spec["completionMode"] == "Indexed"
assert spec["completions"] == ranks
assert spec["parallelism"] == ranks
assert spec["backoffLimit"] == 0

pod = spec["template"]["spec"]
worker = pod["containers"][0]

env = {
    entry["name"]: entry.get("value")
    for entry in worker["env"]
}

assert int(env["WORLD_SIZE"]) == ranks
assert env["NCCL_IB_DISABLE"] == "1"
assert env["NCCL_NET"] == "Socket"

assert config["communication"] == {
    "backend": "nccl",
    "codec": "identity",
}

assert config["model"]["checkpoint_dir"] == "/models/model"
assert config["pipeline"]["layer_counts"] == [14, 14]

images = [
    pod["initContainers"][0]["image"],
    worker["image"],
]
assert len(set(images)) == 1
assert "@sha256:" in images[0]

source = pod["initContainers"][0]["env"]
model_id = next(
    item["value"]
    for item in source
    if item["name"] == "HF_MODEL_ID"
)
assert model_id == "Qwen/Qwen3-0.6B-GPTQ-Int8"

print("PASS: Configuration and Kubernetes topology agree")
print("PASS: NCCL Socket with identity codec")
print("PASS: Model checkpoint and image identity validated")
PYCHECK

kubectl apply --dry-run=server \
  -f deploy/templates/v3_1_indexed_nccl_job.yaml

echo "========== VERIFY GPU CAPACITY =========="

state=$(aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names "$ASG" \
  --region "$REGION" \
  --query \
    'AutoScalingGroups[0].{desired:DesiredCapacity,instances:Instances[].InstanceId}' \
  --output json)

python3 -c '
import json,sys
state=json.loads(sys.argv[1])
assert state["desired"] == 0, state
assert state["instances"] == [], state
print("PASS: No GPU instances running")
' "$state"

echo "========== CHECK RESOURCE COLLISIONS =========="

for resource in \
  "job/$JOB" \
  "service/$SERVICE" \
  "configmap/$CM" \
  "configmap/$MODEL_CM"
do
  if kubectl -n llm get "$resource" >/dev/null 2>&1; then
    echo "STOP: Existing resource: $resource"
    exit 1
  fi
done

echo "========== VERIFY PINNED ECR IMAGE =========="

DIGEST=$(aws ecr describe-images \
  --repository-name "$REPOSITORY" \
  --image-ids imageTag="$TAG" \
  --region "$REGION" \
  --query 'imageDetails[0].imageDigest' \
  --output text)

[[ "$DIGEST" == sha256:* ]] || {
  echo "STOP: Cannot resolve immutable ECR digest"
  exit 1
}

python3 - "$DIGEST" <<'PYIMAGE'
import sys
from pathlib import Path
import yaml

documents = list(yaml.safe_load_all(Path(
    "deploy/templates/v3_1_indexed_nccl_job.yaml"
).read_text()))

pod = documents[1]["spec"]["template"]["spec"]
images = [
    pod["initContainers"][0]["image"],
    pod["containers"][0]["image"],
]

assert all(
    image.endswith("@" + sys.argv[1])
    for image in images
), "Template image differs from tested ECR digest"

print("PASS: Template uses the verified ECR image digest")
PYIMAGE

echo "========== PREPARE SOURCE OVERLAY =========="

cat > /tmp/v31-configured-init.py <<'PYINIT'
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
PYINIT

echo "========== CREATE MODEL CONFIGMAP =========="

MODEL_CM_CREATED=1
kubectl -n llm create configmap "$MODEL_CM" \
  --from-file=model.json=configs/v3_1/qwen3_0_6b_gptq_nccl_n2.json \
  --dry-run=client -o yaml | kubectl apply -f -

echo "========== CREATE RUNTIME CONFIGMAP =========="

CM_CREATED=1
kubectl -n llm create configmap "$CM" \
  --from-file=v3_1_configured_nccl.py=scripts/v3_1_configured_nccl.py \
  --from-file=v3_1_nrank_transport.py=src/llm_runtime/v3_1_nrank_transport.py \
  --from-file=__init__.py=/tmp/v31-configured-init.py \
  --dry-run=client -o yaml | kubectl apply -f -

echo "========== CREATE HEADLESS SERVICE =========="

awk '/^---$/{exit} {print}' \
  deploy/templates/v3_1_indexed_nccl_job.yaml \
  > /tmp/v31-configured-service.yaml

SERVICE_CREATED=1

kubectl apply -f /tmp/v31-configured-service.yaml

echo "========== START TWO A10Gs =========="

GPU_STARTED=1

aws eks update-nodegroup-config \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --region "$REGION" \
  --scaling-config minSize=0,maxSize=2,desiredSize=2 \
  >/dev/null

echo "========== WAIT FOR BOTH READY A10Gs =========="

ready=0

for _ in $(seq 1 90); do
  nodes=$(kubectl get nodes \
    -l "eks.amazonaws.com/nodegroup=$NODEGROUP" \
    -o json 2>/dev/null | python3 -c '
import json,sys
data=json.load(sys.stdin)
count=0
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
        count+=1
print(count)
' 2>/dev/null || echo 0)

  inservice=$(aws autoscaling describe-auto-scaling-groups \
    --auto-scaling-group-names "$ASG" \
    --region "$REGION" \
    --query \
      "length(AutoScalingGroups[0].Instances[?LifecycleState=='InService'])" \
    --output text 2>/dev/null || echo 0)

  if [[ "$nodes" == 2 && "$inservice" == 2 ]]; then
    ready=1
    break
  fi

  sleep 10
done

[[ "$ready" == 1 ]] || {
  echo "ERROR: Two Ready GPU nodes not available"
  kubectl get nodes -o wide || true
  exit 1
}

echo "PASS: Two Ready A10G nodes"

echo "========== START CONFIGURED NCCL JOB =========="

JOB_CREATED=1

kubectl apply \
  -f deploy/templates/v3_1_indexed_nccl_job.yaml

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
    break
  fi

  if [[ "${failed:-0}" -gt 0 ]]; then
    echo "ERROR: Configured NCCL rank failed"
    exit 1
  fi

  sleep 10
done

[[ "$completed" == 1 ]] || {
  echo "ERROR: Configured NCCL Job did not complete"
  exit 1
}

echo "========== VERIFY REAL GENERATION =========="

mkdir -p "$LOG_DIR"

kubectl -n llm get pods \
  -l "batch.kubernetes.io/job-name=$JOB" \
  -o json > "$LOG_DIR/pods.json"

python3 - "$LOG_DIR" <<'PYVERIFY'
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
pods = json.loads((root / "pods.json").read_text())

ranks = {}

for pod in pods["items"]:
    index = pod["metadata"].get(
        "annotations", {}
    ).get("batch.kubernetes.io/job-completion-index")

    if index in ("0", "1"):
        assert index not in ranks, "Duplicate rank pod"
        ranks[index] = pod

assert set(ranks) == {"0", "1"}, ranks.keys()

for index, pod in ranks.items():
    name = pod["metadata"]["name"]

    output = subprocess.run(
        [
            "kubectl", "-n", "llm", "logs",
            f"pod/{name}", "-c", "inference",
            "--tail=-1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    (root / f"rank{index}-acceptance.log").write_text(
        output.stdout
    )

    statuses = pod.get(
        "status", {}
    ).get("containerStatuses", [])

    inference = next(
        status for status in statuses
        if status["name"] == "inference"
    )

    code = inference["state"]["terminated"]["exitCode"]
    assert code == 0, (index, code)

    assert (
        f"PASS: V3.1 configured rank {index} complete"
        in output.stdout
    ), f"Rank {index} completion marker missing"

rank0 = (
    root / "rank0-acceptance.log"
).read_text()

assert "PASS: V3.1 configured NCCL generation" in rank0

records = []
for line in rank0.splitlines():
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue

    if isinstance(value, dict) and "generated_tokens" in value:
        records.append(value)

assert len(records) == 1, records

result = records[0]

assert result["world_size"] == 2
assert result["prompt_tokens"] == [2, 5, 8, 11]
assert len(result["generated_tokens"]) == 4
assert all(
    isinstance(token, int) and token >= 0
    for token in result["generated_tokens"]
)

(root / "generation.json").write_text(
    json.dumps(result, indent=2) + "\n"
)

print("PASS: Both NCCL ranks exited successfully")
print("PASS: Four autoregressive tokens generated")
print("Generated tokens:", result["generated_tokens"])
PYVERIFY

echo "PASS: V3.1 configuration-driven EKS generation verified"
