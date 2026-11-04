#!/usr/bin/env python3
import json
import subprocess
import time

REGION = "us-east-2"
CLUSTER = "llm-pipeline-v1"
GROUPS = ("gpu-a10g", "cpu-system")
HOOK = "Terminate-LC-Hook"


def command(*args):
    return subprocess.check_output(
        list(args), text=True
    ).strip()


def aws(*args):
    return json.loads(command(
        "aws", *args,
        "--region", REGION,
        "--output", "json",
    ))


def kube(*args):
    return json.loads(command(
        "kubectl", *args, "-o", "json"
    ))


print("========== VERIFY INFERENCE STOPPED ==========")

sts = kube("-n", "llm", "get", "sts", "qwen3")
assert sts["spec"]["replicas"] == 0, (
    "STOP: Qwen pipeline still running"
)

print("PASS: Qwen replicas = 0")


print("\n========== VERIFY NODEGROUP CAPACITY ==========")

groups = []
instances = []

for group in GROUPS:
    node = aws(
        "eks", "describe-nodegroup",
        "--cluster-name", CLUSTER,
        "--nodegroup-name", group,
    )["nodegroup"]

    assert node["scalingConfig"]["desiredSize"] == 0, (
        f"STOP: {group} desired capacity is not zero"
    )

    asg_name = node["resources"]["autoScalingGroups"][0]["name"]

    asg = aws(
        "autoscaling", "describe-auto-scaling-groups",
        "--auto-scaling-group-names", asg_name,
    )["AutoScalingGroups"][0]

    assert asg["DesiredCapacity"] == 0, (
        f"STOP: {group} Auto Scaling desired is not zero"
    )

    groups.append((group, asg_name))

    for instance in asg["Instances"]:
        instances.append({
            "group": group,
            "asg": asg_name,
            "id": instance["InstanceId"],
            "state": instance["LifecycleState"],
        })

    print(f"PASS: {group} desired=0")


print("\n========== CHECK TERMINATING INSTANCES ==========")

for instance in instances:
    print(
        instance["group"],
        instance["id"],
        instance["state"],
    )

waiting = [
    i for i in instances
    if i["state"] == "Terminating:Wait"
]

if waiting:
    nodes = kube("get", "nodes")["items"]
    pods = kube("get", "pods", "-A")["items"]

    blockers = []

    for instance in waiting:
        node_name = next((
            node["metadata"]["name"]
            for node in nodes
            if node["spec"].get("providerID", "").endswith(
                "/" + instance["id"]
            )
        ), None)

        print(
            "\nChecking node:",
            node_name or instance["id"],
        )

        if node_name is None:
            print("Node already unregistered from Kubernetes")
            continue

        for pod in pods:
            if pod["spec"].get("nodeName") != node_name:
                continue

            if pod["status"].get("phase") in (
                "Succeeded", "Failed"
            ):
                continue

            namespace = pod["metadata"]["namespace"]
            name = pod["metadata"]["name"]
            owners = pod["metadata"].get(
                "ownerReferences", []
            )

            daemonset = any(
                owner["kind"] == "DaemonSet"
                for owner in owners
            )

            # Expected services during a complete cluster
            # worker shutdown.
            allowed = (
                daemonset
                or namespace == "kube-system"
                or (
                    namespace == "llm"
                    and name.startswith((
                        "qwen3-gpu-observer-",
                        "qwen3-maintenance-supervisor-",
                    ))
                )
            )

            print(
                "Expected shutdown pod:" if allowed
                else "BLOCKER:",
                f"{namespace}/{name}",
            )

            if not allowed:
                blockers.append(f"{namespace}/{name}")

    if blockers:
        raise SystemExit(
            "STOP: Unexpected active workloads: "
            + ", ".join(blockers)
        )


print("\n========== RELEASE TERMINATION HOOKS ==========")

for instance in waiting:
    hooks = aws(
        "autoscaling", "describe-lifecycle-hooks",
        "--auto-scaling-group-name", instance["asg"],
    )["LifecycleHooks"]

    assert any(
        h["LifecycleHookName"] == HOOK
        and h["LifecycleTransition"]
            == "autoscaling:EC2_INSTANCE_TERMINATING"
        for h in hooks
    ), "STOP: Expected termination hook not found"

    # Refresh state in case the hook advanced automatically.
    current = aws(
        "autoscaling", "describe-auto-scaling-groups",
        "--auto-scaling-group-names", instance["asg"],
    )["AutoScalingGroups"][0]

    state = next((
        i["LifecycleState"]
        for i in current["Instances"]
        if i["InstanceId"] == instance["id"]
    ), None)

    if state != "Terminating:Wait":
        print(instance["id"], "already progressed:", state)
        continue

    print("Releasing:", instance["id"])

    subprocess.run([
        "aws", "autoscaling", "complete-lifecycle-action",
        "--region", REGION,
        "--auto-scaling-group-name", instance["asg"],
        "--lifecycle-hook-name", HOOK,
        "--instance-id", instance["id"],
        "--lifecycle-action-result", "CONTINUE",
    ], check=True)

print("\n========== VERIFY EC2 TERMINATION ==========")

ids = [i["id"] for i in instances]

if ids:
    for attempt in range(12):
        response = aws(
            "ec2", "describe-instances",
            "--instance-ids", *ids,
        )

        states = {
            i["InstanceId"]: i["State"]["Name"]
            for reservation in response["Reservations"]
            for i in reservation["Instances"]
        }

        for instance_id in ids:
            print(
                instance_id,
                states.get(instance_id, "not returned"),
            )

        if all(
            states.get(i) == "terminated"
            for i in ids
        ):
            break

        if attempt < 11:
            time.sleep(5)


print("\n========== FINAL AUTO SCALING CHECK ==========")

all_clear = True

for group, asg_name in groups:
    asg = aws(
        "autoscaling", "describe-auto-scaling-groups",
        "--auto-scaling-group-names", asg_name,
    )["AutoScalingGroups"][0]

    remaining = [
        (i["InstanceId"], i["LifecycleState"])
        for i in asg["Instances"]
    ]

    print(group, "remaining:", remaining)
    all_clear = all_clear and not remaining

if all_clear:
    print("\nPASS: All EKS worker instances removed")
else:
    print("\nPENDING: AWS termination still in progress")
