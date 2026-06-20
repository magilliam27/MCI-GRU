#!/usr/bin/env python
"""Benchmark LambdaRankICLoss forward/backward time on CPU or CUDA."""

from __future__ import annotations

import argparse
import json
import time

import torch

from mci_gru.training.losses import LambdaRankICLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-stocks", type=int, default=503)
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to benchmark. 'auto' selects CUDA when available.",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    pred_base = torch.randn(
        args.batch_size,
        args.n_stocks,
        generator=generator,
        device=device,
    )
    target = torch.randn(
        args.batch_size,
        args.n_stocks,
        generator=generator,
        device=device,
    )
    criterion = LambdaRankICLoss(
        max_pairs_per_day=args.max_pairs,
        temperature=args.temperature,
    ).to(device)

    def run_once() -> float:
        pred = pred_base.detach().clone().requires_grad_(True)
        loss = criterion(pred, target)
        loss.backward()
        return float(loss.detach().cpu().item())

    for _ in range(args.warmup):
        run_once()
    synchronize(device)

    timings = []
    losses = []
    for _ in range(args.reps):
        started = time.perf_counter()
        losses.append(run_once())
        synchronize(device)
        timings.append(time.perf_counter() - started)

    result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "n_stocks": args.n_stocks,
        "max_pairs": args.max_pairs,
        "warmup": args.warmup,
        "reps": args.reps,
        "seed": args.seed,
        "temperature": args.temperature,
        "mean_seconds": sum(timings) / len(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "last_loss": losses[-1],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
