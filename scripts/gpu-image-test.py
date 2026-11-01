#!/usr/bin/env python3
"""Test a candidate Qwen3 image on the existing 2 GPUs, then restore V1."""

import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NS = "llm"
EXPECTED_CONTEXT = (
    "arn:aws:eks:us-east-2:657173565477:"
    "cluster/llm-pipeline-v1"
)

CHECKPOINT = ROOT / "test-results/gpu-candidate-test.json"
SNAPSHOT = ROOT / "test-results/snapshots/gpu-candidate-preflight.json"
LOCK = ROOT / "test-results/.gpu-image-test.lock"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def note(message):
    print(f"[{stamp()}] {message}", flush=True)


def execute(*args, input_text=None, timeout=300):
    p = subprocess.run(
        args,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if p.returncode:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(args)}\n"
            f"{p.stdout}\n{p.stderr}"
        )
    return p.stdout.strip()


def kube(*args, input_text=None, timeout=300):
    return execute(
        "kubectl", *args,
        input_text=input_text,
        timeout=timeout,
    )


def resource(kind, name=None, namespace=NS):
    args = ["get", kind]
    if name:
        args.append(name)
    if namespace:
        args += ["-n", namespace]
    return json.loads(kube(*args, "-o", "json"))


def assert_context():
    actual = execute("kubectl", "config", "current-context")
    if actual != EXPECTED_CONTEXT:
        raise RuntimeError(f"Wrong Kubernetes context: {actual}")


def update_checkpoint(**changes):
    data = json.loads(CHECKPOINT.read_text())
    data.update(changes)
    data["updated_at"] = stamp()

    temporary = CHECKPOINT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(CHECKPOINT)


def check_inference_idle():
    script = r'''
import json
import urllib.request

with urllib.request.urlopen(
    "http://127.0.0.1:8000/metrics", timeout=10
) as response:
    metrics = json.load(response)

assert metrics.get("active") is False, metrics
print("idle")
'''

    result = kube(
        "exec", "-i", "-n", NS, "qwen3-0",
        "-c", "runtime", "--", "python", "-",
        input_text=script,
        timeout=45,
    )

    if result != "idle":
        raise RuntimeError(f"Inference idle check failed: {result}")


def scale(kind, name, replicas):
    note(f"Scaling {kind}/{name} to {replicas}")
    kube(
        "scale", f"{kind}/{name}", "-n", NS,
        f"--replicas={replicas}",
    )


def deployment_pods(name):
    deployment = resource("deployment", name)
    selector = deployment["spec"]["selector"]["matchLabels"]

    return [
        pod for pod in resource("pods")["items"]
        if all(
            pod["metadata"].get("labels", {}).get(k) == v
            for k, v in selector.items()
        )
    ]


def wait_deployment_zero(name, timeout=180):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        deployment = resource("deployment", name)

        if (
            deployment["spec"].get("replicas", 1) == 0
            and not deployment_pods(name)
            and deployment.get("status", {}).get("replicas", 0) == 0
        ):
            return

        time.sleep(5)

    raise TimeoutError(f"{name} pods did not terminate")


def wait_workers_zero(timeout=300):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        names = {
            pod["metadata"]["name"]
            for pod in resource("pods")["items"]
        }

        if not {"qwen3-0", "qwen3-1"} & names:
            return

        time.sleep(5)

    raise TimeoutError("GPU worker pods did not terminate")


def wait_workers(image, timeout=1200):
    note("Waiting for both GPU workers to become Ready")

    deadline = time.monotonic() + timeout
    last_report = 0

    while time.monotonic() < deadline:
        pods = {
            pod["metadata"]["name"]: pod
            for pod in resource("pods")["items"]
        }

        ready = []

        for rank in (0, 1):
            pod = pods.get(f"qwen3-{rank}")
            if not pod:
                continue

            runtime = next(
                (
                    c for c in pod["spec"]["containers"]
                    if c["name"] == "runtime"
                ),
                None,
            )

            if runtime is None or runtime["image"] != image:
                raise RuntimeError(
                    f"qwen3-{rank} has unexpected image"
                )

            if any(
                c["type"] == "Ready" and c["status"] == "True"
                for c in pod.get("status", {}).get("conditions", [])
            ):
                ready.append(pod)

        if len(ready) == 2:
            if (
                ready[0]["spec"]["nodeName"]
                == ready[1]["spec"]["nodeName"]
            ):
                raise RuntimeError("GPU ranks share the same node")

            note("Both workers Ready on separate GPU nodes")
            return

        if time.monotonic() - last_report > 30:
            note(f"GPU workers Ready: {len(ready)}/2")

            if image == (
                ROOT / "infrastructure/rebuilt-image.txt"
            ).read_text().strip():
                try:
                    subprocess.run(
                        [
                            "bash",
                            str(
                                ROOT /
                                "scripts/capture-gpu-diagnostics.sh"
                            ),
                        ],
                        timeout=90,
                        check=False,
                    )
                except Exception as exc:
                    note(f"Diagnostic capture warning: {exc}")

            last_report = time.monotonic()

        time.sleep(5)

    raise TimeoutError("GPU workers did not become Ready")


def wait_deployment_ready(name, timeout=300):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        deployment = resource("deployment", name)

        if deployment.get("status", {}).get(
            "availableReplicas", 0
        ) >= 1:
            return

        time.sleep(5)

    raise TimeoutError(f"{name} did not become Available")


def recovery_state():
    cm = resource("configmap", "qwen3-recovery-state")
    return json.loads(cm["data"]["state"])


def patch_state(**fields):
    state = recovery_state()
    state.update(fields)

    kube(
        "patch", "configmap/qwen3-recovery-state",
        "-n", NS, "--type=merge",
        "-p", json.dumps({
            "data": {"state": json.dumps(state)}
        }),
    )


def wait_serving(timeout=240):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        state = recovery_state()

        if (
            state.get("mode") == "RUNNING"
            and state.get("phase") == "SERVING"
        ):
            note("Recovery controller: RUNNING / SERVING")
            return

        if state.get("phase") == "FAILED":
            raise RuntimeError(
                f"Recovery controller entered FAILED: {state}"
            )

        time.sleep(5)

    raise TimeoutError("Recovery controller did not reach SERVING")


def check_volumes(expected):
    for claim, info in expected.items():
        pvc = resource("pvc", claim)

        if pvc["status"]["phase"] != "Bound":
            raise RuntimeError(f"PVC {claim} not Bound")

        pv_name = pvc["spec"]["volumeName"]
        pv = resource("pv", pv_name, namespace=None)
        volume_id = pv["spec"]["csi"]["volumeHandle"]

        if (
            pv_name != info["pv"]
            or volume_id != info["ebs_volume"]
        ):
            raise RuntimeError(
                f"Persistent storage identity changed: {claim}"
            )

    note("Original EBS volumes and PVC bindings preserved")


def set_image(image):
    note(f"Setting Qwen3 runtime image: {image}")

    kube(
        "set", "image", "statefulset/qwen3",
        "-n", NS, f"runtime={image}",
    )

    sts = resource("statefulset", "qwen3")
    runtime = next(
        c for c in sts["spec"]["template"]["spec"]["containers"]
        if c["name"] == "runtime"
    )

    if runtime["image"] != image:
        raise RuntimeError("StatefulSet image update failed")


def test_candidate():
    gpu_script = r'''
import json
import torch

assert torch.__version__ == "2.14.1+cu130"
assert torch.cuda.is_available()
assert torch.distributed.is_nccl_available()

tensor = torch.ones((8,), device="cuda")
torch.cuda.synchronize()
assert tensor.sum().item() == 8.0

print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "nccl": list(torch.cuda.nccl.version()),
    "gpu": torch.cuda.get_device_name(0),
}))
'''

    gpu_results = {}

    for rank in (0, 1):
        result = kube(
            "exec", "-i", "-n", NS, f"qwen3-{rank}",
            "-c", "runtime", "--", "python", "-",
            input_text=gpu_script,
            timeout=120,
        )

        gpu_results[f"qwen3-{rank}"] = json.loads(
            result.splitlines()[-1]
        )

        note(f"Rank {rank}: CUDA and NCCL libraries verified")

    inference_script = r'''
import json
import urllib.request

payload = json.dumps({
    "prompt": "What is the capital of France?",
    "max_new_tokens": 32,
}).encode()

request = urllib.request.Request(
    "http://127.0.0.1:8000/generate",
    data=payload,
    method="POST",
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(
    request, timeout=180
) as response:
    result = json.load(response)

assert result.get("generated_tokens", 0) > 0, result
assert "paris" in result.get("text", "").lower(), result

print(json.dumps({
    "text": result["text"],
    "generated_tokens": result["generated_tokens"],
    "metrics": result.get("metrics", {}),
}))
'''

    output = kube(
        "exec", "-i", "-n", NS, "qwen3-0",
        "-c", "runtime", "--", "python", "-",
        input_text=inference_script,
        timeout=240,
    )

    inference = json.loads(output.splitlines()[-1])

    note("Candidate two-stage inference passed")

    return {
        "gpu": gpu_results,
        "inference": inference,
    }


def restore():
    assert_context()

    if not CHECKPOINT.exists():
        raise RuntimeError("No GPU-test checkpoint found")

    data = json.loads(CHECKPOINT.read_text())
    original = data["original_image"]

    pinned = (
        ROOT / "infrastructure/model-image.txt"
    ).read_text().strip()

    if original != pinned:
        raise RuntimeError(
            "Original image differs from pinned image"
        )

    volumes = data["storage"]

    note("RESTORE: stopping CPU services")

    scale("deployment", "qwen3-coordinator", 0)
    wait_deployment_zero("qwen3-coordinator")

    scale("deployment", "qwen3-recovery", 0)
    wait_deployment_zero("qwen3-recovery")

    patch_state(
        mode="PAUSED",
        phase="STARTING",
        baseline=None,
    )

    note("RESTORE: removing candidate workers")

    scale("statefulset", "qwen3", 0)
    wait_workers_zero()

    check_volumes(volumes)
    set_image(original)

    note("RESTORE: starting original workers")

    scale("statefulset", "qwen3", 2)
    wait_workers(original)

    check_volumes(volumes)

    state = recovery_state()

    patch_state(
        mode="RUNNING",
        phase="STARTING",
        generation=int(state.get("generation", 0)) + 1,
        attempt=0,
        baseline=None,
        started_at=None,
        last_error=None,
    )

    scale("deployment", "qwen3-recovery", 1)
    wait_deployment_ready("qwen3-recovery")
    wait_serving()

    scale("deployment", "qwen3-coordinator", 1)
    wait_deployment_ready("qwen3-coordinator")

    note("Running original end-to-end verification")

    output = execute(
        "bash", str(ROOT / "scripts/verify.sh"),
        timeout=360,
    )

    print(output, flush=True)

    update_checkpoint(
        restored=True,
        restored_at=stamp(),
        stage="restored",
    )

    note("RESTORE COMPLETE: original V1 verified")


def run_test():
    assert_context()

    if CHECKPOINT.exists():
        previous = json.loads(CHECKPOINT.read_text())

        if not previous.get("restored", False):
            raise RuntimeError(
                "Unfinished GPU test exists. Run restore first."
            )

    note("Running fresh preflight")

    execute(
        "python3",
        str(ROOT / "scripts/gpu-test-preflight.py"),
        timeout=150,
    )

    check_inference_idle()

    snapshot = json.loads(SNAPSHOT.read_text())

    CHECKPOINT.write_text(json.dumps({
        "created_at": stamp(),
        "stage": "starting",
        "restored": False,
        "candidate_passed": False,
        "original_image": snapshot["original_image"],
        "candidate_image": snapshot["candidate_image"],
        "storage": snapshot["storage"],
    }, indent=2) + "\n")

    candidate_ok = False
    error = None

    try:
        note("Starting planned inference downtime")

        scale("deployment", "qwen3-coordinator", 0)
        wait_deployment_zero("qwen3-coordinator")

        # No new gateway traffic; verify no worker request is active.
        check_inference_idle()

        scale("deployment", "qwen3-recovery", 0)
        wait_deployment_zero("qwen3-recovery")

        patch_state(
            mode="PAUSED",
            phase="STARTING",
            baseline=None,
        )

        update_checkpoint(stage="candidate-starting")

        scale("statefulset", "qwen3", 0)
        wait_workers_zero()

        check_volumes(snapshot["storage"])

        set_image(snapshot["candidate_image"])

        scale("statefulset", "qwen3", 2)
        wait_workers(snapshot["candidate_image"], timeout=1800)

        check_volumes(snapshot["storage"])

        result = test_candidate()

        update_checkpoint(
            candidate_passed=True,
            candidate_result=result,
            stage="candidate-tested",
        )

        candidate_ok = True

    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        note(f"CANDIDATE TEST FAILED OR INTERRUPTED: {error}")
        update_checkpoint(candidate_error=error)

    finally:
        note("Restoring original V1")

        try:
            restore()
        except BaseException as restore_error:
            note(
                "URGENT: automatic restoration incomplete: "
                f"{restore_error}"
            )
            note(
                "Run: python3 scripts/gpu-image-test.py restore"
            )
            raise

    if not candidate_ok:
        raise RuntimeError(
            f"Candidate test failed, but V1 was restored. {error}"
        )

    note("SUCCESS: candidate tested and original V1 restored")


def main():
    if (
        len(sys.argv) != 2
        or sys.argv[1] not in ("check", "run", "restore")
    ):
        raise SystemExit(
            "Usage: python3 scripts/gpu-image-test.py "
            "[check|run|restore]"
        )

    LOCK.parent.mkdir(parents=True, exist_ok=True)

    with LOCK.open("w") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            raise SystemExit(
                "Another GPU test or restoration is running"
            )

        assert_context()

        if sys.argv[1] == "check":
            execute(
                "python3",
                str(ROOT / "scripts/gpu-test-preflight.py"),
            )
            print(
                "PASS: Read-only check; no cluster changes",
                flush=True,
            )

        elif sys.argv[1] == "run":
            run_test()

        else:
            restore()


if __name__ == "__main__":
    main()
