# V3.1 Distributed Failure Policy

## Scope

V3.1 uses fail-stop recovery for distributed inference.

Fatal conditions include:
- incompatible startup configuration or protocol version
- invalid control messages or broken control connections
- worker execution failure
- distributed operation timeout
- unrecoverable transport or codec failure

## Failed generation

After a fatal failure:
- Reject new requests.
- Fail all pending requests.
- Mark the pipeline generation unhealthy.
- Do not reuse any KV caches from the failed generation.
- Do not resume partially transferred activations.
- Do not retry native distributed operations on an uncertain communicator.

## Local process supervision

The local process supervisor:
- starts a distributed worker group
- enforces a total execution deadline
- propagates nonzero exit status
- terminates its child process group on failure
- escalates to SIGKILL when necessary
- does not automatically retry failed generations

## Kubernetes integration requirements

Kubernetes recovery must later:
- identify the failed pipeline generation
- stop all workers in that generation
- ensure old workers cannot participate in a new generation
- create fresh distributed communicators
- reload model stages and initialize empty KV caches
- validate rank membership and startup capabilities
- report readiness only after the new group is healthy

A local process supervisor cannot by itself terminate Pods
running on other Kubernetes nodes.

## Limitations

These tests validate local subprocess supervision and failure
propagation. They do not establish coordinated Kubernetes
recovery or guarantee termination of custom plugins that
spawn detached processes outside the supervised process group.
