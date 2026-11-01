"""Two-stage Qwen3-14B NCCL greedy inference and benchmarking."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from qwen3_stage_loader import Qwen3Stage


def now_ms():
    return time.perf_counter_ns() / 1e6


def forward_timed(stage, x, pos, cache):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    t0 = now_ms()
    start.record()
    result = stage.forward_local(x, pos, cache)
    end.record()
    torch.cuda.synchronize()

    return result, start.elapsed_time(end), now_ms() - t0


def blocking_ms(operation):
    torch.cuda.synchronize()
    t0 = now_ms()
    operation()
    torch.cuda.synchronize()
    return now_ms() - t0


def summarize(values):
    values = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": values[int(0.95 * (len(values) - 1))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--model-dir", default="models/qwen3-14b")
    parser.add_argument("--split", type=int, default=20)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prompt-len", type=int, default=0)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--tag", default="smoke")
    args = parser.parse_args()

    assert args.rank in (0, 1)
    assert args.output_len >= 1
    assert args.runs >= 1
    assert args.warmups >= 0
    assert args.prompt_len >= 0

    torch.cuda.set_device(0)

    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda:0"),
    )

    rank = dist.get_rank()
    assert rank == args.rank
    assert dist.get_world_size() == 2
    assert torch.cuda.is_bf16_supported()

    print(f"[rank{rank}] Loading Qwen3-14B...", flush=True)

    stage = Qwen3Stage(
        args.model_dir, rank, args.split
    ).load()

    tokenizer = (
        AutoTokenizer.from_pretrained(args.model_dir)
        if rank == 0 else None
    )

    torch.cuda.synchronize()
    dist.barrier(device_ids=[0])

    weights_mib = torch.cuda.memory_allocated() / 2**20
    print(
        f"[rank{rank}] READY "
        f"weights={weights_mib:.2f} MiB",
        flush=True,
    )

    if rank == 0:
        if args.prompt_len:
            seed = tokenizer.encode(
                "Distributed inference partitions transformer layers "
                "across devices. ",
                add_special_tokens=False,
            )
            ids = (
                seed * (
                    (args.prompt_len + len(seed) - 1)
                    // len(seed)
                )
            )[:args.prompt_len]
        else:
            ids = tokenizer.encode(
                args.prompt, add_special_tokens=False
            )

        assert len(ids) > 0

        prompt = torch.tensor(
            [ids], device="cuda", dtype=torch.long
        )
        print(
            f"[rank0] Prompt tokens: {len(ids)}",
            flush=True,
        )
    else:
        prompt = None

    results = []

    with torch.inference_mode():
        for request_id in range(args.warmups + args.runs):
            is_warmup = request_id < args.warmups

            if rank == 0:
                control = torch.tensor(
                    [prompt.shape[1], args.output_len],
                    device="cuda",
                    dtype=torch.int64,
                )
            else:
                control = torch.empty(
                    2, device="cuda", dtype=torch.int64
                )

            dist.broadcast(control, src=0)
            prompt_len, output_len = map(
                int, control.cpu().tolist()
            )

            assert (
                prompt_len + output_len - 1
                <= stage.cfg.max_position_embeddings
            )

            cache = stage.make_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            request_start = now_ms()
            previous_completion = request_start

            generated = []
            intervals = []
            compute_gpu = []
            compute_wall = []
            communication = []
            previous_token = None

            for step in range(output_len):
                position = (
                    0 if step == 0
                    else prompt_len + step - 1
                )
                sequence_len = (
                    prompt_len if step == 0 else 1
                )

                if rank == 0:
                    input_ids = (
                        prompt if step == 0
                        else previous_token.view(1, 1)
                    )

                    hidden, gpu_ms, wall_ms = forward_timed(
                        stage, input_ids, position, cache
                    )

                    compute_gpu.append(float(gpu_ms))
                    compute_wall.append(wall_ms)

                    communication.append(
                        blocking_ms(
                            lambda: dist.send(hidden, dst=1)
                        )
                    )
                    del hidden

                    token = torch.empty(
                        1, device="cuda", dtype=torch.int64
                    )

                    communication.append(
                        blocking_ms(
                            lambda: dist.recv(token, src=1)
                        )
                    )

                    previous_token = token
                    generated.append(int(token.item()))

                else:
                    hidden = torch.empty(
                        (1, sequence_len, stage.cfg.hidden_size),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )

                    communication.append(
                        blocking_ms(
                            lambda: dist.recv(hidden, src=0)
                        )
                    )

                    logits, gpu_ms, wall_ms = forward_timed(
                        stage, hidden, position, cache
                    )

                    compute_gpu.append(float(gpu_ms))
                    compute_wall.append(wall_ms)
                    del hidden

                    token = logits.argmax(
                        dim=-1
                    ).to(torch.int64).contiguous()

                    communication.append(
                        blocking_ms(
                            lambda: dist.send(token, dst=0)
                        )
                    )
                    del token, logits

                completion = now_ms()
                if rank == 0:
                    intervals.append(
                        completion - previous_completion
                    )
                previous_completion = completion

            torch.cuda.synchronize()
            total_ms = now_ms() - request_start

            expected_cache_len = prompt_len + output_len - 1
            actual_cache_len = cache.get_seq_length(
                stage.start
            )
            assert actual_cache_len == expected_cache_len, (
                actual_cache_len, expected_cache_len
            )

            row = {
                "rank": rank,
                "request_id": request_id,
                "warmup": is_warmup,
                "host": platform.node(),
                "model": "Qwen/Qwen3-14B",
                "split": args.split,
                "dtype": "bf16",
                "transport": "nccl",
                "attention": "eager",
                "batch_size": 1,
                "prompt_tokens": prompt_len,
                "output_tokens": output_len,
                "total_ms": total_ms,
                "prefill_gpu_ms": compute_gpu[0],
                "decode_gpu_ms": compute_gpu[1:],
                "total_gpu_compute_ms": sum(compute_gpu),
                "compute_wall_ms": compute_wall,
                "communication_blocking_ms": communication,
                "token_intervals_ms": (
                    intervals if rank == 0 else None
                ),
                "generated_ids": (
                    generated if rank == 0 else None
                ),
                "weights_mib": weights_mib,
                "peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() / 2**20
                ),
                "prefill_activation_bytes": (
                    prompt_len * stage.cfg.hidden_size * 2
                ),
                "decode_activation_bytes": (
                    stage.cfg.hidden_size * 2
                ),
            }

            if not is_warmup:
                results.append(row)

                if rank == 0:
                    print(
                        f"[rank0] run={len(results)} "
                        f"E2E={total_ms:.2f} ms "
                        f"TTFT={intervals[0]:.2f} ms "
                        f"output={tokenizer.decode(generated)!r}",
                        flush=True,
                    )
                else:
                    print(
                        f"[rank1] run={len(results)} "
                        f"compute={sum(compute_gpu):.2f} ms",
                        flush=True,
                    )

            del cache

    Path("results").mkdir(exist_ok=True)
    path = Path("results") / (
        f"{args.tag}_rank{rank}.jsonl"
    )

    with path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"[rank{rank}] Saved {path}", flush=True)

    if rank == 0:
        print(
            "E2E summary:",
            summarize([r["total_ms"] for r in results]),
        )
        print(
            "TTFT summary:",
            summarize([
                r["token_intervals_ms"][0]
                for r in results
            ]),
        )

        if args.output_len > 1:
            decode = [
                x for r in results
                for x in r["token_intervals_ms"][1:]
            ]
            print(
                "Decode intervals:",
                summarize(decode),
            )

    dist.barrier(device_ids=[0])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
