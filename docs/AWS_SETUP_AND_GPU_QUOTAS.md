# AWS Setup and GPU Quota Guide

This guide helps users prepare their own AWS environment
for the distributed LLM inference framework.

It covers:
- AWS identity and region
- EKS cluster discovery
- GPU instance selection
- GPU vCPU service quotas
- Kubernetes node labels and storage
- Deployment configuration
- Pre-deployment validation

No AWS resources are created by the discovery commands.

The repository includes read-only AWS account, region
and EKS cluster discovery. Full configuration validation,
GPU quota checks and deployment rendering are separate
follow-up capabilities.

## 1. Prerequisites

Install or access:

- AWS CLI v2
- kubectl
- An AWS account with appropriate IAM permissions
- An existing EKS cluster, if deploying to EKS

AWS CloudShell already provides the AWS CLI and normally
uses credentials associated with your console session.

For local development, configure IAM Identity Center
authentication where available:

    aws configure sso

Avoid hardcoding AWS credentials in repository files.

## 2. Find your AWS account ID

Run:

```bash
aws sts get-caller-identity \
  --query Account \
  --output text
```

This returns the AWS account ID for the active identity.

The framework should eventually discover this automatically.

Do not copy an account ID from example manifests.

## 3. Select an AWS region

Inspect the configured region:

```bash
aws configure get region
```

CloudShell may also provide AWS_REGION or
AWS_DEFAULT_REGION environment variables.

Select the region that will host your workload:

```bash
export AWS_REGION=us-east-2
```

Replace `us-east-2` with your intended region.

Use that region consistently for EKS, EC2,
storage and Service Quotas.

GPU instance availability and quota values
differ by AWS region and account.

## 4. Find an EKS cluster

List clusters in your selected region:

```bash
aws eks list-clusters \
  --region "$AWS_REGION" \
  --query 'clusters' \
  --output table
```

Select the intended cluster:

```bash
export CLUSTER_NAME=your-cluster-name
```

Inspect it:

```bash
aws eks describe-cluster \
  --region "$AWS_REGION" \
  --name "$CLUSTER_NAME" \
  --query 'cluster.{Name:name,Status:status,Version:version,VPC:resourcesVpcConfig.vpcId}' \
  --output table
```

The cluster should exist and be ACTIVE.

If no cluster exists, provision one using the
AWS EKS documentation before continuing.

The framework should never select a cluster
automatically when multiple candidates exist.

## 5. Find GPU node groups

List managed node groups:

```bash
aws eks list-nodegroups \
  --region "$AWS_REGION" \
  --cluster-name "$CLUSTER_NAME" \
  --output table
```

Select a node group:

```bash
export NODEGROUP_NAME=your-gpu-nodegroup
```

Inspect its configuration:

```bash
aws eks describe-nodegroup \
  --region "$AWS_REGION" \
  --cluster-name "$CLUSTER_NAME" \
  --nodegroup-name "$NODEGROUP_NAME" \
  --query 'nodegroup.{Name:nodegroupName,Status:status,InstanceTypes:instanceTypes,CapacityType:capacityType,Scaling:scalingConfig,Labels:labels}' \
  --output json
```

Record:
- EC2 instance type
- On-Demand or Spot capacity type
- Desired/minimum/maximum node counts
- Kubernetes node labels

The reference deployment expects nodes with:

    pool: gpu

If your node group uses a different label, update
the rendered node selector or configure a compatible
persistent label on your node group.

Do not assume every EKS node has a GPU.

## 6. Choose a GPU instance type

Reference NVIDIA A10G examples:

| Instance type | GPUs | vCPUs |
|---|---:|---:|
| g5.xlarge | 1 | 4 |
| g5.2xlarge | 1 | 8 |
| g5.4xlarge | 1 | 16 |

Confirm the instance specification in your region:

```bash
aws ec2 describe-instance-types \
  --region "$AWS_REGION" \
  --instance-types g5.2xlarge \
  --query 'InstanceTypes[].{Type:InstanceType,vCPUs:VCpuInfo.DefaultVCpus,GPUCount:GpuInfo.Gpus[0].Count}' \
  --output table
```

Check which availability zones advertise the instance:

```bash
aws ec2 describe-instance-type-offerings \
  --region "$AWS_REGION" \
  --location-type availability-zone \
  --filters Name=instance-type,Values=g5.2xlarge \
  --query 'InstanceTypeOfferings[].Location' \
  --output table
```

An instance type being offered does not guarantee
that AWS currently has free capacity to launch it.

## 7. Understand GPU vCPU quotas

EC2 On-Demand instance quotas are regional and
measured in vCPUs, not GPU counts.

For G5 instances, find:

    Running On-Demand G and VT instances

For P-family On-Demand instances, find:

    Running On-Demand P instances

Spot instances use different quotas, including:

    All G and VT Spot Instance Requests

A quota of zero can prevent GPU instances from launching.

### Calculate the required quota

Formula:

    required_vCPUs = instance_count * vCPUs_per_instance

Example: two g5.2xlarge instances.

    2 * 8 = 16 vCPUs

You need at least 16 available G-family On-Demand
vCPUs, accounting for any other G/VT instances
already consuming that regional quota.

Requesting 24 or 32 vCPUs is an optional headroom
choice. AWS decides whether to approve the increase.

## 8. Check your current GPU quota

Open AWS Service Quotas:

https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/

Select the correct AWS region.

Search for:

    Running On-Demand G and VT instances

Inspect your applied quota value.

Alternatively, using AWS CLI:

```bash
aws service-quotas list-service-quotas \
  --service-code ec2 \
  --region "$AWS_REGION" \
  --query "Quotas[?contains(QuotaName, 'G and VT')].[QuotaName,QuotaCode,Value,Adjustable]" \
  --output table
```

This returns relevant quota names, codes and values.

Copy the appropriate On-Demand quota code, not a
Spot quota code.

## 9. Request a GPU vCPU quota increase

### AWS Console

1. Open the AWS Service Quotas console.
2. Select the correct AWS region.
3. Choose Amazon Elastic Compute Cloud (Amazon EC2).
4. Search for "Running On-Demand G and VT instances".
5. Open the quota details.
6. Select the request increase option.
7. Enter the desired total regional quota value.
8. Submit and monitor the request status.

The desired value is the new total quota, not
an incremental amount to add.

For example, request 32 to set a target
regional quota of 32 vCPUs.

AWS may approve automatically, open a Support case,
partially approve, or reject the request.

### AWS CLI (optional)

First discover the correct quota code:

```bash
G_QUOTA_CODE=$(aws service-quotas list-service-quotas \
  --service-code ec2 \
  --region "$AWS_REGION" \
  --query "Quotas[?QuotaName=='Running On-Demand G and VT instances'] | [0].QuotaCode" \
  --output text)

echo "$G_QUOTA_CODE"
```

If the output is `None` or empty, stop and inspect
the quota through the AWS Console.

Otherwise, inspect the current applied quota:

```bash
aws service-quotas get-service-quota \
  --region "$AWS_REGION" \
  --service-code ec2 \
  --quota-code "$G_QUOTA_CODE" \
  --query 'Quota.{Name:QuotaName,Value:Value,Adjustable:Adjustable}' \
  --output table
```

Only when you intentionally want to submit a request:

```bash
aws service-quotas request-service-quota-increase \
  --region "$AWS_REGION" \
  --service-code ec2 \
  --quota-code "$G_QUOTA_CODE" \
  --desired-value 32
```

Replace 32 with your intended total quota.

This submits a real AWS service quota request.
It does not itself launch GPU instances.

Appropriate IAM permissions and, where applicable,
permissions for Service Quotas to open Support cases
are required.

### Track an increase request

```bash
aws service-quotas \
  list-requested-service-quota-change-history-by-quota \
  --region "$AWS_REGION" \
  --service-code ec2 \
  --quota-code "$G_QUOTA_CODE" \
  --output table
```

Wait for an approved and effective applied quota
before attempting a GPU node-group launch.

Quota approval is separate from EC2 capacity.

## 10. Discover Kubernetes configuration

Connect kubectl to your selected EKS cluster:

```bash
aws eks update-kubeconfig \
  --region "$AWS_REGION" \
  --name "$CLUSTER_NAME"
```

This changes your local kubeconfig/current context.
It does not create or deploy cluster resources.

Confirm the current cluster before continuing:

```bash
kubectl config current-context
```

Inspect namespaces:

```bash
kubectl get namespaces
```

Inspect nodes and GPU labels:

```bash
kubectl get nodes -L pool
```

If your nodes are active, inspect available resources:

```bash
kubectl describe nodes
```

For the reference manifests using nvidia.com/gpu,
confirm that the NVIDIA device plugin registers
GPU allocatable resources on the worker nodes.

EKS also supports NVIDIA DRA for suitable newer
clusters, but those resources are not interchangeable
with the current reference manifest without changes.

## 11. Find model storage

List StorageClasses:

```bash
kubectl get storageclass
```

List PVCs in your chosen namespace:

```bash
kubectl get pvc -n llm
```

Replace `llm` with your deployment namespace.

The existing default deployment references:

    model-files-qwen3-0
    model-files-qwen3-1

These are example deployment references, not PVCs
automatically created for a new user's account.

Verify that:
- PVCs exist in the intended namespace.
- PVCs are bound to suitable persistent volumes.
- The expected model checkpoint files exist.
- The paths inside the containers match the model files.
- Storage access and placement are compatible
  with the selected nodes.

EBS-backed PVCs typically require the EBS CSI
driver and its required IAM permissions.

Do not create new PVCs or download large checkpoints
as part of an automatic discovery operation.

## 12. Match configuration values

| Configuration field | Source |
|---|---|
| aws.region | Selected AWS region |
| aws.account_id | sts get-caller-identity |
| aws.cluster_name | eks list-clusters |
| aws.namespace | Existing or explicitly chosen namespace |
| model.id | Selected model checkpoint identifier |
| model.checkpoint_path | Actual container model path |
| model.pvc_rank0 | Existing/provisioned rank 0 PVC |
| model.pvc_rank1 | Existing/provisioned rank 1 PVC |
| runtime.backend | Validated backend, currently vllm |
| runtime.pipeline_parallelism | Compatible execution topology |
| runtime.tensor_parallelism | Compatible execution topology |
| deployment.gpu_node_selector | Actual Kubernetes node labels |
| deployment.gpu_instance_type | Selected GPU EC2 type |
| deployment.replicas | Initially 0 |

## 13. Before deployment

Verify all of the following:

- AWS account and region are correct.
- EKS cluster is accessible.
- Intended node groups are configured.
- GPU quota is sufficient.
- Instance type is offered in the intended AZs.
- GPU drivers and device plugin are available.
- Model storage is provisioned and accessible.
- Model planning and manifest validation succeed.
- No unsupported model/backend configuration is used.
- Kubernetes resource requests are compatible.
- Deployment replicas remain zero until explicitly started.
- AWS budgets and cost monitoring are configured.

An EKS cluster control plane, persistent volumes,
load balancers and running instances can incur charges.

Offline rendering alone does not launch resources.

## 14. Troubleshooting

### EC2 vCPU quota exceeded

Check the regional instance-family service quota.
Request an appropriate increase.

### InsufficientInstanceCapacity

This is not necessarily a quota problem.
Check the instance type and Availability Zones.

### GPU Pods remain Pending

Inspect:

    kubectl describe pod POD_NAME -n NAMESPACE

Check node selectors, taints, GPU allocatable
resources, scheduling and storage.

### PVCs remain Pending

Check StorageClass, EBS CSI permissions,
volume provisioning and availability zones.

### kubectl reports unauthorized

Verify the AWS identity, EKS access permissions,
and selected Kubernetes context.

## Official AWS references

EC2 instance quotas:
https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html

Service Quotas:
https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html

EC2 accelerated instance specifications:
https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html

EKS kubeconfig:
https://docs.aws.amazon.com/cli/latest/reference/eks/update-kubeconfig.html

NVIDIA GPUs on EKS:
https://docs.aws.amazon.com/eks/latest/userguide/device-management-nvidia.html

EBS CSI storage:
https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html

## 15. Generate a local configuration automatically

Install the configuration dependency:

```bash
python3 -m pip install -e '.[config]'
```

From the repository root, run:

```bash
python3 scripts/v3_5_aws_configure.py \
  --template examples/aws/deployment.example.yaml \
  --output .llm-eks/deployment.yaml
```

Override the selected region if necessary:

```bash
python3 scripts/v3_5_aws_configure.py \
  --region us-east-2 \
  --cluster your-eks-cluster
```

These are read-only AWS discovery operations.

The output is created with restrictive file permissions
and its directory is excluded from Git.

The tool currently discovers account ID, region and
EKS cluster identity. When multiple clusters exist,
it does not select one without an explicit choice.

GPU node-group configuration, PVCs, available GPU
vCPU quotas and Kubernetes access still require
verification before deployment.

The tool never creates resources or requests quotas.
The configuration is not yet a supported input to
the Kubernetes deployment renderer.

## 16. Discover GPU resources and model storage

The following commands are read-only. They do not
create AWS resources, change EKS node groups, or
deploy Kubernetes workloads.

### Discover GPU node groups and regional quota

```bash
python3 scripts/v3_5_aws_enrich.py \
  --config .llm-eks/deployment.yaml \
  --output .llm-eks/deployment.discovered.yaml
```

The selected instance type must match an available
managed node group.

If no matching group is found, verify the configured
instance type against your cluster's node groups.
Do not assume a quota increase is necessary.

### Discover existing model PVCs

```bash
python3 scripts/v3_5_aws_storage.py \
  --config .llm-eks/deployment.discovered.yaml \
  --output .llm-eks/deployment.storage.yaml
```

For Qwen3-14B, automatic storage discovery checks
the reference PVC names.

Other models require explicit PVC selection using
--pvc-rank0 and --pvc-rank1.

PVC discovery checks that the selected volumes are
distinct and Bound.

It does not inspect model files or guarantee that
EBS volumes can mount on the intended GPU nodes.

### Run the environment doctor

```bash
python3 scripts/v3_5_aws_doctor.py \
  --config .llm-eks/deployment.storage.yaml \
  --check-k8s
```

Doctor findings use PASS, WARN and FAIL.

A result with zero FAIL findings does not mean
that all deployment prerequisites are verified.
Review warnings and unverified checks before deployment.

Current checks include:

- Configured AWS target and runtime topology
- Safe deployment replica count
- Managed GPU group and node selector
- Regional GPU quota limit
- Node-group maximum and desired capacity
- Kubernetes namespace accessibility
- Ready nodes advertising NVIDIA GPU resources
- Model PVC existence and Bound status

The doctor does not currently verify checkpoint
contents, unused quota, GPU allocation by other
Pods, EBS volume AZ placement, or live inference.

All generated local configurations are stored
under .llm-eks/ and excluded from Git.

## 17. Render the discovered deployment configuration

The V3.5 renderer generates an isolated Kubernetes
manifest bundle using previously discovered settings.

Current supported reference:
- Qwen3-14B BF16
- vLLM backend
- Pipeline parallelism 2
- Tensor parallelism 1
- Two g5.xlarge On-Demand GPU workers
- Separate CPU coordinator and recovery deployments

The CPU node group must currently be provided explicitly.

Example:

```bash
python3 scripts/v3_5_render_discovered.py \
  --config .llm-eks/deployment.storage.yaml \
  --cpu-nodegroup cpu-system \
  --output-dir test-results/v35-discovered-reference
```

Replace cpu-system with your actual managed CPU
node-group name.

The output directory must not already exist.

The renderer produces 14 JSON resources with v35-
resource names, including:

- GPU deployments and services
- CPU coordinator and recovery
- Model metadata and embedded source
- Recovery state, service account and RBAC

Namespace-specific service addresses, execution-plan
references, resource names and node selectors are
updated consistently.

All deployments have zero replicas.

Recovery remains paused and automatic recovery
remains disabled.

The renderer does not communicate with Kubernetes,
create resources, download checkpoints or start GPUs.

Generated files under test-results/ are Git-ignored.

IMPORTANT: The renderer currently relies on stored
discovery results. It does not independently refresh
AWS quotas, Kubernetes storage state or instance
availability.

Before deployment, inspect the generated manifests
and confirm the CPU node group, GPU node group,
model checkpoint contents, storage placement and
runtime readiness.

This renderer intentionally supports the validated
Qwen3-14B reference configuration only.

Other model architectures and model replacement
workflows require their own verified execution plans.
