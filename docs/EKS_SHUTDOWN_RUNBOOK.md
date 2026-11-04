# EKS Worker Shutdown Runbook

Cluster: llm-pipeline-v1
Region: us-east-2

## Known recurring issue

Scaling EKS managed nodegroups to desiredSize=0 does not
guarantee immediate EC2 termination.

Our Auto Scaling groups have a termination lifecycle hook:

- Name: Terminate-LC-Hook
- Heartbeat timeout: 1800 seconds
- Default result: CONTINUE

The cpu-system t3.large instance has repeatedly remained
running in Terminating:Wait after successful nodegroup
scale-down updates.

This can also happen to GPU nodes.

## Standard shutdown sequence

1. Save benchmark results and relevant logs.
2. Scale the qwen3 StatefulSet to zero.
3. Wait for qwen3-0 and qwen3-1 to be deleted.
4. Scale gpu-a10g and cpu-system nodegroups to zero.
5. Check Auto Scaling instances, not just EKS desiredSize.
6. Run the lifecycle-hook finalization script:

   python3 scripts/finalize_eks_shutdown.py

7. Confirm all EC2 worker instances have terminated.

## Important

- UPDATING means EKS is still processing the request.
- ACTIVE with desiredSize=0 does not prove EC2 shutdown.
- Terminating:Wait can mean the instance is still running.
- Terminating:Proceed means AWS is completing termination.
- Verify the final EC2 state or empty ASG instance lists.
- Do not delete lifecycle hooks from EKS-managed groups.
- Do not bypass graceful shutdown if application workloads
  are still active.
- EKS control-plane and EBS charges continue after workers
  terminate.

## Next deployment

After bringing the CPU node and GPU nodes online, confirm
the inference StatefulSet and supporting services are
healthy before benchmarking.
