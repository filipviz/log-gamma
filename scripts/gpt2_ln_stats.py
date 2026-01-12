#!/usr/bin/env python3
"""Collect basic LayerNorm gamma stats from a pretrained GPT-2 checkpoint."""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import GPT2LMHeadModel


def _to_numpy(param):
    return param.detach().float().cpu().numpy()


def _flatten(params):
    return np.concatenate([p.reshape(-1) for p in params], axis=0)


def _ln_sort_key(name):
    if name == "ln_f":
        return (1, 0, 0)
    parts = name.split(".")
    layer = int(parts[1])
    ln_order = 0 if parts[2] == "ln_1" else 1
    return (0, layer, ln_order)


def collect_ln_gammas(model):
    groups = {"ln_1": [], "ln_2": [], "ln_f": []}
    layernorms = {}
    for name, param in model.named_parameters():
        if name.endswith(".ln_1.weight"):
            values = _to_numpy(param)
            groups["ln_1"].append(values)
            layer_idx = int(name.split(".")[2])
            layernorms[f"h.{layer_idx:02d}.ln_1"] = values
        elif name.endswith(".ln_2.weight"):
            values = _to_numpy(param)
            groups["ln_2"].append(values)
            layer_idx = int(name.split(".")[2])
            layernorms[f"h.{layer_idx:02d}.ln_2"] = values
        elif name.endswith(".ln_f.weight"):
            values = _to_numpy(param)
            groups["ln_f"].append(values)
            layernorms["ln_f"] = values
    layernorms = dict(sorted(layernorms.items(), key=lambda item: _ln_sort_key(item[0])))
    return groups, layernorms


def summarize(arr):
    negative = (arr < 0).sum()
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "pct_negative": float(negative) / float(arr.size) * 100.0,
    }


def plot_hist_groups(group_values, bins, output_path):
    labels = ["ln_1", "ln_2", "ln_f", "all"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, label in zip(axes, labels):
        ax.hist(group_values[label], bins=bins, color="#2f4b7c", alpha=0.85)
        ax.set_title(label)
        ax.set_xlabel("gamma")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_hist_layernorms(layernorms, bins, output_path, ncols=6):
    names = list(layernorms.keys())
    n = len(names)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    for idx, name in enumerate(names):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        ax.hist(layernorms[name], bins=bins, color="#2f4b7c", alpha=0.85)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("gamma")
        ax.grid(True, alpha=0.2)
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")
    axes[0, 0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Collect LN gamma stats for pretrained GPT-2")
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2",
                        help="Hugging Face model id, e.g. gpt2 or openai-community/gpt2")
    parser.add_argument("--output_dir", type=str, default="outputs/gpt2_ln_stats",
                        help="Where to write stats + plots")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    print(f"Loading {args.model_name}...")
    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    groups, layernorms = collect_ln_gammas(model)
    flat_groups = {k: _flatten(v) for k, v in groups.items()}
    all_gammas = _flatten(list(flat_groups.values()))

    stats = {
        "all": summarize(all_gammas),
        "groups": {k: summarize(v) for k, v in flat_groups.items()},
        "layernorms": {name: summarize(values) for name, values in layernorms.items()},
    }

    stats_path = os.path.join(args.output_dir, "ln_gamma_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    group_hist_path = os.path.join(args.output_dir, "ln_gamma_hist_groups.png")
    group_values = {**flat_groups, "all": all_gammas}
    plot_hist_groups(group_values, bins=args.bins, output_path=group_hist_path)

    layer_hist_path = os.path.join(args.output_dir, "ln_gamma_hist_layernorms.png")
    plot_hist_layernorms(layernorms, bins=args.bins, output_path=layer_hist_path)

    print(f"Wrote stats to {stats_path}")
    print(f"Wrote histogram to {group_hist_path}")
    print(f"Wrote histogram to {layer_hist_path}")


if __name__ == "__main__":
    main()
