#!/usr/bin/env bash
set -euo pipefail

REGION=us-east-2
CLUSTER=llm-pipeline-v1
NODEGROUP=gpu-a10g
ASG=eks-gpu-a10g-32d0c4bb-a5e4-bea9-6af3-ec95a70be2f2
REPOSITORY=qwen3-14b-pipeline
TAG=v3.1-gptq-marlin-r1
JOB=v31-qwen3-full-marlin
CM=v31-qwen3-full-marlin-script
LOG=test-results/v3_1/qwen3-full-marlin-gpu.log

GPU_STARTED=0
CM_CREATED=0
JOB_CREATED=0

cleanup() {
  rc=$?
  trap - EXIT INT TERM

  echo "========== COLLECT RESULTS =========="
  mkdir -p test-results/v3_1

  if [[ "$JOB_CREATED" == 1 ]]; then
    kubectl -n llm logs "job/$JOB" \
      --tail=-1 > "$LOG" 2>&1 || true
    kubectl -n llm delete job "$JOB" \
      --ignore-not-found --wait=true \
      --timeout=120s || true
  fi

  if [[ "$CM_CREATED" == 1 ]]; then
    kubectl -n llm delete configmap "$CM" \
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

  echo "Results: $LOG"
  [[ ! -s "$LOG" ]] || tail -n 75 "$LOG"

  exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "========== PREFLIGHT =========="

test -f scripts/v3_1_qwen3_full_marlin_smoke.py

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

[[ "$NG_STATUS" == ACTIVE ]] || exit 1
[[ "$NG_DESIRED" == 0 ]] || exit 1
[[ "$INSTANCE_COUNT" == 0 ]] || exit 1

if kubectl -n llm get job "$JOB" >/dev/null 2>&1; then
  echo "ERROR: Existing test Job"
  exit 1
fi

if kubectl -n llm get cm "$CM" >/dev/null 2>&1; then
  echo "ERROR: Existing test ConfigMap"
  exit 1
fi

DIGEST=$(aws ecr describe-images \
  --repository-name "$REPOSITORY" \
  --image-ids imageTag="$TAG" \
  --region "$REGION" \
  --query 'imageDetails[0].imageDigest' \
  --output text)

[[ "$DIGEST" == sha256:* ]] || exit 1

IMAGE="657173565477.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY}@${DIGEST}"

echo "========== PREPARE TEST =========="

kubectl -n llm create configmap "$CM" \
  --from-file=scripts/v3_1_qwen3_full_marlin_smoke.py \
  --dry-run=client -o yaml | kubectl apply -f -

CM_CREATED=1

cat > /tmp/v31-full-gpu.yaml <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB}
  namespace: llm
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 1500
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        eks.amazonaws.com/nodegroup: ${NODEGROUP}
      volumes:
        - name: smoke
          configMap:
            name: ${CM}
      containers:
        - name: full-marlin
          image: ${IMAGE}
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -euo pipefail
              export PYTHONPATH=/app:/opt/marlin-output
              export HF_HOME=/tmp/huggingface
              export PYTHONUNBUFFERED=1

              nvidia-smi

              python - <<'PY'
              from huggingface_hub import snapshot_download
              snapshot_download(
                  repo_id="Qwen/Qwen3-0.6B-GPTQ-Int8",
                  local_dir="/tmp/qwen3-0.6b-gptq",
                  allow_patterns=[
                      "config.json",
                      "*.safetensors",
                      "*.safetensors.index.json"
                  ],
              )
              PY

              python /opt/v31/v3_1_qwen3_full_marlin_smoke.py \
                --checkpoint-dir /tmp/qwen3-0.6b-gptq
          volumeMounts:
            - name: smoke
              mountPath: /opt/v31
          resources:
            requests:
              cpu: "1"
              memory: 6Gi
              nvidia.com/gpu: "1"
            limits:
              memory: 12Gi
              nvidia.com/gpu: "1"
EOF

kubectl apply --dry-run=client \
  -f /tmp/v31-full-gpu.yaml >/dev/null

echo "========== START A10G =========="

GPU_STARTED=1

aws eks update-nodegroup-config \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --region "$REGION" \
  --scaling-config minSize=0,maxSize=2,desiredSize=1 \
  >/dev/null

echo "========== CHECK NODE READINESS =========="

ready=0

for _ in $(seq 1 60); do
  node=$(kubectl get nodes \
    -l eks.amazonaws.com/nodegroup="$NODEGROUP" \
    -o jsonpath='{.items[0].metadata.name}' \
    2>/dev/null || true)

  if [[ -n "$node" ]]; then
    status=$(kubectl get node "$node" \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' \
      2>/dev/null || true)

    unschedulable=$(kubectl get node "$node" \
      -o jsonpath='{.spec.unschedulable}' \
      2>/dev/null || true)

    gpu=$(kubectl get node "$node" \
      -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' \
      2>/dev/null || true)

    inservice=$(aws autoscaling describe-auto-scaling-groups \
      --auto-scaling-group-names "$ASG" \
      --region "$REGION" \
      --query \
      "length(AutoScalingGroups[0].Instances[?LifecycleState=='InService'])" \
      --output text 2>/dev/null || true)

    if [[ "$status" == True &&
          "$unschedulable" != true &&
          "$gpu" == 1 &&
          "$inservice" == 1 ]]; then
      echo "Ready: $node"
      ready=1
      break
    fi
  fi

  sleep 10
done

[[ "$ready" == 1 ]] || {
  echo "GPU node did not become ready"
  exit 1
}

echo "========== RUN FULL QWEN3 TEST =========="

JOB_CREATED=1
kubectl apply -f /tmp/v31-full-gpu.yaml

for _ in $(seq 1 150); do
  succeeded=$(kubectl -n llm get job "$JOB" \
    -o jsonpath='{.status.succeeded}' \
    2>/dev/null || true)

  failed=$(kubectl -n llm get job "$JOB" \
    -o jsonpath='{.status.failed}' \
    2>/dev/null || true)

  if [[ "$succeeded" == 1 ]]; then
    echo "PASS: Full-model GPU Job succeeded"
    exit 0
  fi

  if [[ "${failed:-0}" -gt 0 ]]; then
    kubectl -n llm logs "job/$JOB" \
      --tail=120 || true
    exit 1
  fi

  sleep 10
done

echo "ERROR: GPU Job did not complete"
exit 1
