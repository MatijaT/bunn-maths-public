#!/usr/bin/env python
"""
Flip analysis: on which nodes do rotation variants break the full BuNN's
correct predictions on Minesweeper?

We use the .npz files produced by analyze_rotation_nodes_minesweeper.py.
For each seed, we:

  - Restrict to test nodes.
  - Compute predictions for full / noRB / identity by thresholding p(y=1) at 0.5.
  - Focus on nodes where the full model is correct.
  - For each heterophily bin, compute:
      * #correct_full (test) nodes in that bin,
      * fraction of those that become wrong under noRB,
      * fraction of those that become wrong under identity.

We then average these fractions across seeds and print a Markdown table.

Usage:
  python summarize_rotation_flip_analysis.py \
      --dir rotation_analysis \
      --pattern "rotation_seed*.npz"
"""

import argparse
import glob
import os

import numpy as np


def compute_per_seed_stats(path, bins):
    data = np.load(path)

    y = data["y"]                  # (N,)
    test_mask = data["test_mask"].astype(bool)
    heter = data["heterophily"]    # (N,)

    prob_full = data["prob_full"]        # (N,)
    prob_noRB = data["prob_noRB"]        # (N,)
    prob_id = data["prob_identity"]      # (N,)

    # Binary predictions via threshold 0.5 (Minesweeper is binary)
    pred_full = (prob_full >= 0.5).astype(int)
    pred_noRB = (prob_noRB >= 0.5).astype(int)
    pred_id = (prob_id >= 0.5).astype(int)

    # Restrict to test nodes
    test_idx = test_mask

    y_test = y[test_idx]
    heter_test = heter[test_idx]
    pred_full_test = pred_full[test_idx]
    pred_noRB_test = pred_noRB[test_idx]
    pred_id_test = pred_id[test_idx]

    # Nodes where the full model is correct on test set
    correct_full = (pred_full_test == y_test)

    # Among these, which become wrong under each variant?
    mis_noRB = (pred_noRB_test != y_test) & correct_full
    mis_id = (pred_id_test != y_test) & correct_full

    num_bins = len(bins) - 1
    n_correct_per_bin = np.zeros(num_bins, dtype=np.int64)
    frac_flip_noRB = np.full(num_bins, np.nan, dtype=float)
    frac_flip_id = np.full(num_bins, np.nan, dtype=float)

    for b in range(num_bins):
        low, high = bins[b], bins[b + 1]
        if b == num_bins - 1:
            # include high end in last bin
            in_bin = (heter_test >= low) & (heter_test <= high)
        else:
            in_bin = (heter_test >= low) & (heter_test < high)

        # Among test nodes in this bin, look at those full got right
        bin_mask = in_bin & correct_full
        n_correct = int(bin_mask.sum())
        n_correct_per_bin[b] = n_correct

        if n_correct == 0:
            continue

        n_flip_noRB = int((mis_noRB & in_bin).sum())
        n_flip_id = int((mis_id & in_bin).sum())

        frac_flip_noRB[b] = n_flip_noRB / n_correct
        frac_flip_id[b] = n_flip_id / n_correct

    # Global across all heterophily bins
    n_correct_total = int(correct_full.sum())
    if n_correct_total > 0:
        frac_flip_noRB_global = mis_noRB.sum() / n_correct_total
        frac_flip_id_global = mis_id.sum() / n_correct_total
    else:
        frac_flip_noRB_global = np.nan
        frac_flip_id_global = np.nan

    return (
        n_correct_per_bin,
        frac_flip_noRB,
        frac_flip_id,
        n_correct_total,
        frac_flip_noRB_global,
        frac_flip_id_global,
    )


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

    pattern_path = os.path.join(args.dir, args.pattern)
    files = sorted(glob.glob(pattern_path))
    if not files:
        raise RuntimeError(f"No files matched pattern {pattern_path!r}")

    print(f"Found {len(files)} analysis files:")
    for f in files:
        print(f"  - {f}")

    # Same bins as before
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    bin_labels = ["[0.00, 0.25)", "[0.25, 0.50)", "[0.50, 0.75)", "[0.75, 1.00]"]
    num_bins = len(bin_labels)

    # Collect per-seed stats
    n_correct_bins_all = []
    frac_noRB_all = []
    frac_id_all = []
    n_correct_total_all = []
    frac_noRB_global_all = []
    frac_id_global_all = []

    for f in files:
        (
            n_correct_per_bin,
            frac_flip_noRB,
            frac_flip_id,
            n_correct_total,
            frac_flip_noRB_global,
            frac_flip_id_global,
        ) = compute_per_seed_stats(f, bins)

        n_correct_bins_all.append(n_correct_per_bin)
        frac_noRB_all.append(frac_flip_noRB)
        frac_id_all.append(frac_flip_id)
        n_correct_total_all.append(n_correct_total)
        frac_noRB_global_all.append(frac_flip_noRB_global)
        frac_id_global_all.append(frac_flip_id_global)

    n_correct_bins_all = np.stack(n_correct_bins_all, axis=0)  # (S, B)
    frac_noRB_all = np.stack(frac_noRB_all, axis=0)            # (S, B)
    frac_id_all = np.stack(frac_id_all, axis=0)                # (S, B)

    n_correct_total_all = np.array(n_correct_total_all)
    frac_noRB_global_all = np.array(frac_noRB_global_all)
    frac_id_global_all = np.array(frac_id_global_all)

    # Mean/std across seeds (ignoring NaNs for fractions)
    mean_n_correct_bins = n_correct_bins_all.mean(axis=0)
    mean_frac_noRB = np.nanmean(frac_noRB_all, axis=0)
    std_frac_noRB = np.nanstd(frac_noRB_all, axis=0)

    mean_frac_id = np.nanmean(frac_id_all, axis=0)
    std_frac_id = np.nanstd(frac_id_all, axis=0)

    mean_n_correct_total = n_correct_total_all.mean()
    mean_frac_noRB_global = np.nanmean(frac_noRB_global_all)
    std_frac_noRB_global = np.nanstd(frac_noRB_global_all)
    mean_frac_id_global = np.nanmean(frac_id_global_all)
    std_frac_id_global = np.nanstd(frac_id_global_all)

    print("\n### Flip analysis: among test nodes where the full BuNN is correct\n")
    print(
        "For each heterophily bin, we report the average number of test nodes "
        "that the full model gets right, and the fraction of those that become "
        "wrong under the no-rotate-back and identity variants."
    )

    print(
        "\n| Heterophily range | mean #correct_full (test) | frac flipped by noRB (mean ± std) | "
        "frac flipped by identity (mean ± std) |"
    )
    print(
        "|-------------------|---------------------------|------------------------------------|"
        "------------------------------------------|"
    )

    for b, label in enumerate(bin_labels):
        m_n = mean_n_correct_bins[b]
        m_noRB = mean_frac_noRB[b]
        s_noRB = std_frac_noRB[b]
        m_id = mean_frac_id[b]
        s_id = std_frac_id[b]

        if np.isnan(m_noRB):
            noRB_str = " - "
        else:
            noRB_str = f"{m_noRB:.3f} ± {s_noRB:.3f}"

        if np.isnan(m_id):
            id_str = " - "
        else:
            id_str = f"{m_id:.3f} ± {s_id:.3f}"

        print(
            f"| {label:<17} | {m_n:25.1f} | {noRB_str:34} | {id_str:40} |"
        )

    print("\nGlobal (all heterophily bins combined):")
    print(f"  mean #correct_full (test)      = {mean_n_correct_total:.1f}")
    print(
        f"  frac flipped by noRB           = {mean_frac_noRB_global:.3f} ± {std_frac_noRB_global:.3f}"
    )
    print(
        f"  frac flipped by identity       = {mean_frac_id_global:.3f} ± {std_frac_id_global:.3f}"
    )


if __name__ == "__main__":
    main()

