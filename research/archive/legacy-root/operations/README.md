# Historical EKS operational scripts

These scripts were used during the V1 development phase.

- pause.sh: scale workloads and node groups down
- resume.sh: restore workers and saved replica counts
- teardown.sh: permanently delete EKS resources

WARNING: These files contain cluster-specific settings.
They are not supported public CLI commands.

teardown.sh is destructive and can delete model storage,
Kubernetes workloads, IAM resources and EKS infrastructure.

Do not execute these scripts without reviewing their
targets, AWS identity and all resource-deletion actions.

The three scripts remain together because pause/resume
share local state and teardown references pause.sh.

For historical documentation only. Archiving does not
disable their AWS operations or prevent execution.
