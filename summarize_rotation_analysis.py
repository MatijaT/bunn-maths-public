#!/usr/bin/env python
"""
Summarize BuNN rotation analysis on Minesweeper.

Reads .npz files produced by analyze_rotation_nodes_minesweeper.py and
prints Markdown tables showing how |Δ| = |prob_full - prob_variant|
relates to node/edge heterophily.

Node-level:
  - heterophily(v) = fraction of neighbors with different label
  - For each heterophily bin, report:
      #nodes, mean/std |Δ_full-noRB|, mean/std |Δ_full-identity|

Edge-level:
  - edge_heterophily(e) = 1{ y[src] != y[dst] }
  - For homophilous vs heterophilous edges, report:
      #edges, mean |Δ_full-noRB|, mean |Δ_full-identity|

Usage:
  python summarize_rotation_analysis.py \
      --dir rotation_analysis \
      --pattern "rotation_seed*.npz"
"""

import argparse
import glob
import os

import numpy as np


def load_all_npz(dir_path: str, pattern: str):
    pattern_path = os.path.join(dir_path, pattern)
    files = sorted(glob.glob(pattern_path))
    if not files:
        raise RuntimeError(f"No files matched pattern {pattern_path!r}")

    print(f"Found {len(files)} analysis files:")
    for f in files:
        print(f"  - {f}")

    node_delta_noRB_list = []
    node_delta_id_list = []
    node_heter_list = []

    edge_delta_noRB_list = []
    edge_delta_id_list = []
    edge_heter = None

    for f in files:
        data = np.load(f)
        delta_noRB = data["delta_noRB"]          # (N,)
        delta_id = data["delta_identity"]        # (N,)
        heter = data["heterophily"]              # (N,)

        node_delta_noRB_list.append(delta_noRB)
        node_delta_id_list.append(delta_id)
        node_heter_list.append(heter)

        # Edge-level
        e_delta_noRB = data["edge_delta_noRB"]         # (E,)
        e_delta_id = data["edge_delta_identity"]       # (E,)
        e_heter = data["edge_heterophily"]             # (E,)

        edge_delta_noRB_list.append(e_delta_noRB)
        edge_delta_id_list.append(e_delta_id)

        if edge_heter is None:
            edge_heter = e_heter
        else:
            # Sanity check: edge_heterophily should be identical across seeds
            if not np.array_equal(edge_heter, e_heter):
                print(f"[WARN] edge_heterophily differs across seeds in file {f}")

    # Stack over seeds: shape (S, N) and (S, E)
    node_delta_noRB_all = np.stack(node_delta_noRB_list, axis=0)
    node_delta_id_all = np.stack(node_delta_id_list, axis=0)
    node_heter_all = np.stack(node_heter_list, axis=0)

    edge_delta_noRB_all = np.stack(edge_delta_noRB_list, axis=0)
    edge_delta_id_all = np.stack(edge_delta_id_list, axis=0)

    # Average over seeds: shape (N,) and (E,)
    node_delta_noRB_mean = node_delta_noRB_all.mean(axis=0)
    node_delta_id_mean = node_delta_id_all.mean(axis=0)
    node_heter_mean = node_heter_all.mean(axis=0)  # should be identical across seeds, but average anyway

    edge_delta_noRB_mean = edge_delta_noRB_all.mean(axis=0)
    edge_delta_id_mean = edge_delta_id_all.mean(axis=0)

    return (
        node_heter_mean,
        node_delta_noRB_mean,
        node_delta_id_mean,
        edge_heter,
        edge_delta_noRB_mean,
        edge_delta_id_mean,
    )


def summarize_nodes(heter, delta_noRB, delta_id):
    """
    heter, delta_noRB, delta_id: (N,) arrays
    """
    print("\n### Node-level summary (binned by heterophily)\n")

    bins = [0.0, 0.25, 0.5, 0.75, 1.01]
    bin_labels = ["[0.00, 0.25)", "[0.25, 0.50)", "[0.50, 0.75)", "[0.75, 1.00]"]

    print("| Heterophily range | #nodes | mean |Δ_full-noRB| | std |Δ_full-noRB| | mean |Δ_full-identity| | std |Δ_full-identity| |")
    print("|-------------------|--------|------------------|------------------|------------------------|------------------------|")

    for (low, high), label in zip(zip(bins[:-1], bins[1:]), bin_labels):
        mask = (heter >= low) & (heter < high)
        n = int(mask.sum())
        if n == 0:
            print(f"| {label:<17} | {n:6d} |        -         |        -         |          -             |          -             |")
            continue

        d_noRB = delta_noRB[mask]
        d_id = delta_id[mask]

        mean_noRB = d_noRB.mean()
        std_noRB = d_noRB.std()

        mean_id = d_id.mean()
        std_id = d_id.std()

        print(
            f"| {label:<17} | {n:6d} | {mean_noRB:16.4f} | {std_noRB:16.4f} | {mean_id:22.4f} | {std_id:22.4f} |"
        )

    # Global stats
    print("\nGlobal node-level means:")
    print(f"  mean |Δ_full-noRB|      = {delta_noRB.mean():.4f}")
    print(f"  mean |Δ_full-identity|  = {delta_id.mean():.4f}")


def summarize_edges(edge_heter, edge_delta_noRB, edge_delta_id):
    """
    edge_heter: (E,) int array in {0,1}
    edge_delta_noRB, edge_delta_id: (E,)
    """
    print("\n### Edge-level summary (homophilous vs heterophilous edges)\n")

    mask_homo = edge_heter == 0
    mask_heter = edge_heter == 1

    def stats(mask):
        if mask.sum() == 0:
            return 0, float("nan"), float("nan")
        d_noRB = edge_delta_noRB[mask]
        d_id = edge_delta_id[mask]
        return int(mask.sum()), d_noRB.mean(), d_id.mean()

    n_homo, mean_noRB_homo, mean_id_homo = stats(mask_homo)
    n_heter, mean_noRB_heter, mean_id_heter = stats(mask_heter)

    print("| Edge type      | #edges | mean |Δ_full-noRB| | mean |Δ_full-identity| |")
    print("|----------------|--------|------------------|------------------------|")
    print(
        f"| homophilous    | {n_homo:6d} | {mean_noRB_homo:16.4f} | {mean_id_homo:22.4f} |"
    )
    print(
        f"| heterophilous  | {n_heter:6d} | {mean_noRB_heter:16.4f} | {mean_id_heter:22.4f} |"
    )

    # Global edge-level means
    print("\nGlobal edge-level means:")
    print(f"  mean |Δ_full-noRB|      = {edge_delta_noRB.mean():.4f}")
    print(f"  mean |Δ_full-identity|  = {edge_delta_id.mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="rotation_analysis",
        help="Directory containing rotation_seed*.npz files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="rotation_seed*.npz",
        help="Glob pattern to match .npz files inside --dir.",
    )
    args = parser.parse_args()

    (
        node_heter,
        node_delta_noRB,
        node_delta_id,
        edge_heter,
        edge_delta_noRB,
        edge_delta_id,
    ) = load_all_npz(args.dir, args.pattern)

    summarize_nodes(node_heter, node_delta_noRB, node_delta_id)
    summarize_edges(edge_heter, edge_delta_noRB, edge_delta_id)


if __name__ == "__main__":
    main()

