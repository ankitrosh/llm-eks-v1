# Historical AWS CodeBuild specifications

This directory contains AWS CodeBuild specifications
used during earlier V2 and V3 experiments.

They include compiler, quantization, SDPA,
Marlin, prefix caching and runtime validation
experiments.

These files are retained for reproducibility
and historical reference.

They are not part of the current supported
V3.5 public CLI workflow.

The original paths are preserved in Git history.

## Important

Existing external CodeBuild automation using
StartBuild buildspec overrides may need to update
its paths.

The root-level buildspec.yml remains in place
for default CodeBuild compatibility.

These historical specifications may reference
old AWS infrastructure, images or repository paths.
Review them before attempting to execute them.
