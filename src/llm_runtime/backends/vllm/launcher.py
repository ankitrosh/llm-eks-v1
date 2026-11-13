"""CPU-only vLLM distributed launch planner."""

from dataclasses import dataclass

from llm_runtime.extensions import (
    CompatibilityError,
    PluginSpec,
)


@dataclass(frozen=True)
class VLLMLaunchConfig:
    model: str
    nnodes: int
    gpus_per_node: int
    pipeline_parallel_size: int
    tensor_parallel_size: int
    master_addr: str
    master_port: int = 29501
    max_model_len: int = 2048
    dtype: str = "bfloat16"
    port: int = 8000

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("Model path or ID is required")

        for name in (
            "nnodes",
            "gpus_per_node",
            "pipeline_parallel_size",
            "tensor_parallel_size",
            "master_port",
            "max_model_len",
            "port",
        ):
            value = getattr(self, name)

            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid {name}: {value}")

        total_gpus = self.nnodes * self.gpus_per_node

        required_gpus = (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
        )

        # Initial implementation: one distributed model
        # instance with homogeneous nodes, no replicas.
        if required_gpus != total_gpus:
            raise CompatibilityError(
                f"PP * TP = {required_gpus}, "
                f"but available GPUs = {total_gpus}"
            )

        if not self.master_addr:
            raise ValueError("master_addr is required")

        if self.nnodes > 1 and self.master_addr in (
            "localhost",
            "127.0.0.1",
        ):
            raise ValueError(
                "Multi-node master_addr must be reachable "
                "from the worker nodes"
            )

        if self.dtype not in (
            "auto",
            "bfloat16",
            "float16",
            "float32",
        ):
            raise ValueError(f"Unsupported dtype: {self.dtype}")


class VLLMBackend:
    """Build version-dependent CLI plans.

    Does not execute a subprocess or allocate a GPU.
    """

    def __init__(self, config: VLLMLaunchConfig):
        self.config = config

    def command_for_rank(self, rank: int) -> list[str]:
        c = self.config

        if type(rank) is not int or not 0 <= rank < c.nnodes:
            raise ValueError(f"Invalid node rank: {rank}")

        cmd = [
            "vllm",
            "serve",
            c.model,
            "--distributed-executor-backend",
            "mp",
            "--tensor-parallel-size",
            str(c.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(c.pipeline_parallel_size),
            "--nnodes",
            str(c.nnodes),
            "--node-rank",
            str(rank),
            "--master-addr",
            c.master_addr,
            "--master-port",
            str(c.master_port),
            "--dtype",
            c.dtype,
            "--max-model-len",
            str(c.max_model_len),
        ]

        if rank == 0:
            cmd.extend([
                "--host",
                "0.0.0.0",
                "--port",
                str(c.port),
            ])
        else:
            cmd.append("--headless")

        return cmd

    def plan(self) -> dict:
        return {
            "schema_version": 1,
            "backend": "vllm",
            "model": self.config.model,
            "pipeline_parallel_size":
                self.config.pipeline_parallel_size,
            "tensor_parallel_size":
                self.config.tensor_parallel_size,
            "nodes": [
                {
                    "rank": rank,
                    "command": self.command_for_rank(rank),
                }
                for rank in range(self.config.nnodes)
            ],
        }


def create_vllm_backend(context):
    if context.backend != "vllm":
        raise CompatibilityError(
            "vLLM backend selected with wrong context"
        )

    deployment = context.runtime.get("deployment", {})

    model = deployment.get(
        "model_path",
        context.model.get("model_id"),
    )

    config = VLLMLaunchConfig(
        model=model,
        nnodes=deployment["nnodes"],
        gpus_per_node=deployment["gpus_per_node"],
        pipeline_parallel_size=deployment["pp"],
        tensor_parallel_size=deployment["tp"],
        master_addr=deployment["master_addr"],
        master_port=deployment.get("master_port", 29501),
        max_model_len=deployment.get("max_model_len", 2048),
        dtype=deployment.get("dtype", "bfloat16"),
        port=deployment.get("port", 8000),
    )

    return VLLMBackend(config)


def register_vllm_backend(registry):
    registry.register(
        PluginSpec(
            kind="backend",
            name="vllm",
            supported_backends=frozenset({"vllm"}),
            factory=create_vllm_backend,
        )
    )
