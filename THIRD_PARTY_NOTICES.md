# Third-Party Software and Model Notices

The llm-eks-runtime framework is licensed under
Apache License 2.0. See LICENSE.

The framework integrates with third-party software,
runtime images, model checkpoints, and cloud services.
Those components retain their own license terms.

## vLLM

Project: https://github.com/vllm-project/vllm

License: Apache-2.0

The reference GPU deployment uses vLLM. The custom
scheduler example subclasses a vLLM scheduler class.

The example runtime image is:

    vllm/vllm-openai:v0.32.0

The standalone llm-eks-runtime Python package does
not bundle the vLLM server or its container image.

Review the license notices of the actual container
image and its dependencies before redistribution.

## Hugging Face Transformers

Project: https://github.com/huggingface/transformers

License: Apache-2.0

Used in historical and reference-model workflows.

Transformers and its dependencies are separately
distributed third-party software.

## Qwen3 Models

Reference model:

https://huggingface.co/Qwen/Qwen3-14B

License listed by the model publisher: Apache-2.0.

The Qwen3-8B model is also used for offline model
inspection and deployment-planning examples.

Model weights and tokenizer files are distributed
separately and are not included in the framework
Python wheel.

Users must review the license and model card for
the exact checkpoint and revision they deploy.

## PyTorch

Project: https://github.com/pytorch/pytorch

PyTorch is distributed under its own licensing terms.
Its binaries and bundled dependencies can have
additional license obligations.

See the license files supplied with the installed
PyTorch distribution.

## NVIDIA CUDA and NCCL

CUDA and NCCL are NVIDIA technologies used in the
GPU inference and communication environments.

Their software distributions and container components
have separate license terms.

The Apache-2.0 license for llm-eks-runtime does not
grant additional rights to NVIDIA software.

## AWS and Kubernetes

AWS services and managed resources are subject to
AWS service terms.

Kubernetes and other ecosystem components retain
their respective open-source licenses.

## Distribution guidance

If you redistribute a container image, Python
environment, checkpoint, or third-party software
along with this framework, review and preserve
the license and attribution materials required
by those actual distributed components.

This document identifies significant integrations.
It is not a complete software bill of materials
or an exhaustive third-party license audit.

Any third-party source incorporated directly into
this repository must also retain the attribution
and license notices required by its original license.
