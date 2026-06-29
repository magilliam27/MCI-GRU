#!/usr/bin/env python
"""Benchmark the optimized LambdaRankIC loss against the previous pair path."""

from __future__ import annotations

import argparse
import gc
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from mci_gru.training.losses import (
    LambdaRankICLoss,
    _cap_pair_indices,
    _standardize_cross_section,
    _zero_loss_like,
)


def _baseline_average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty(values.numel(), dtype=values.dtype, device=values.device)
    sorted_values = values[order]
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


class BaselineTriuLambdaRankICLoss(nn.Module):
    """Previous rank-loop plus triu_indices, filter, then cap implementation."""

    def __init__(
        self,
        max_pairs_per_day: int = 4096,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.max_pairs_per_day = max_pairs_per_day
        self.temperature = temperature
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for p, t in zip(pred, target, strict=True):
            mask = torch.isfinite(p) & torch.isfinite(t)
            valid_count = int(mask.sum().item())
            if valid_count < 2:
                continue

            score_z = _standardize_cross_section(p[mask], self.eps)
            label_ranks = _baseline_average_ranks(t[mask].detach())
            pred_ranks = _baseline_average_ranks(score_z.detach())
            left, right = torch.triu_indices(
                valid_count,
                valid_count,
                offset=1,
                device=score_z.device,
            )

            label_diff = label_ranks[left] - label_ranks[right]
            ordered = label_diff != 0
            if not bool(ordered.any()):
                continue

            left = left[ordered]
            right = right[ordered]
            left, right = _cap_pair_indices(left, right, self.max_pairs_per_day)
            label_diff = label_ranks[left] - label_ranks[right]
            direction = torch.sign(label_diff)
            score_diff = score_z[left] - score_z[right]

            pair_loss = F.softplus(-(direction * score_diff) / self.temperature)
            n_float = float(valid_count)
            pred_rank_sep = (pred_ranks[right] - pred_ranks[left]).abs()
            weights = (
                12.0 * pred_rank_sep * label_diff.abs() / (n_float * (n_float * n_float - 1.0))
            ).detach()
            weight_sum = weights.sum()
            if (not torch.isfinite(weight_sum)) or float(weight_sum.item()) <= self.eps:
                weights = (12.0 * label_diff.abs() / (n_float * (n_float * n_float - 1.0))).detach()
                weight_sum = weights.sum()
            if (not torch.isfinite(weight_sum)) or float(weight_sum.item()) <= self.eps:
                continue
            losses.append((weights * pair_loss).sum() / (weight_sum + self.eps))

        if not losses:
            return _zero_loss_like(pred)
        return torch.stack(losses).mean()


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_loss(
    name: str,
    loss_fn: nn.Module,
    pred_template: torch.Tensor,
    target: torch.Tensor,
    reps: int,
    warmup: int,
) -> dict[str, float | str]:
    for _ in range(warmup):
        pred = pred_template.detach().clone().requires_grad_(True)
        loss = loss_fn(pred, target)
        loss.backward()
    _sync_if_needed(pred_template.device)
    gc.collect()

    times = []
    final_loss = 0.0
    for _ in range(reps):
        pred = pred_template.detach().clone().requires_grad_(True)
        _sync_if_needed(pred_template.device)
        started = time.perf_counter()
        loss = loss_fn(pred, target)
        loss.backward()
        _sync_if_needed(pred_template.device)
        times.append(time.perf_counter() - started)
        final_loss = float(loss.detach().cpu())

    return {
        "name": name,
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
        "loss": final_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-stocks", type=int, default=500)
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch.manual_seed(args.seed)
    pred = torch.randn(args.batch_size, args.n_stocks, device=device)
    target = torch.randn(args.batch_size, args.n_stocks, device=device)
    target[:, ::37] = float("nan")

    baseline = BaselineTriuLambdaRankICLoss(
        max_pairs_per_day=args.max_pairs,
        temperature=args.temperature,
    )
    optimized = LambdaRankICLoss(
        max_pairs_per_day=args.max_pairs,
        temperature=args.temperature,
    )

    baseline_check = baseline(pred.detach().clone().requires_grad_(True), target)
    optimized_check = optimized(pred.detach().clone().requires_grad_(True), target)
    results = {
        "device": str(device),
        "shape": [args.batch_size, args.n_stocks],
        "max_pairs": args.max_pairs,
        "temperature": args.temperature,
        "reps": args.reps,
        "warmup": args.warmup,
        "abs_loss_diff": abs(float(baseline_check.detach() - optimized_check.detach())),
        "baseline": _time_loss("baseline_triu", baseline, pred, target, args.reps, args.warmup),
        "optimized": _time_loss("optimized", optimized, pred, target, args.reps, args.warmup),
    }
    results["speedup"] = results["baseline"]["mean_s"] / results["optimized"]["mean_s"]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
