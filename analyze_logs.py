import glob
import re
import numpy as np


def parse_final_line(line):
    """
    Generic parser for lines like:

      FINAL: config=gcn, model_type=gcn, seed=0, val_auc=0.9012, test_auc=0.8975
      FINAL: config=small, seed=0, val_auc=0.8051, test_auc=0.8011
      FINAL RESULT model=mlp, seed=3, val_roc=0.88, test_roc=0.87

    We:
      - find the part after 'FINAL'
      - split it into key=value chunks
      - pick out config/model, seed, val_auc, test_auc
    """

    if "FINAL" not in line:
        return None

    # Take everything after the first 'FINAL'
    if "FINAL:" in line:
        payload = line.split("FINAL:", 1)[1]
    else:
        payload = line.split("FINAL", 1)[1]

    # Split on commas into key=value pieces
    parts = payload.strip().split(",")
    kv = {}
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        kv[key] = val

    # Config / model name
    config = kv.get("config") or kv.get("model_type") or kv.get("model")
    if config is None:
        return None

    # Seed (optional; default 0 if missing or malformed)
    seed_str = kv.get("seed", "0")
    try:
        seed = int(seed_str)
    except ValueError:
        seed = 0

    def pick_float(keys):
        for k in keys:
            if k in kv:
                try:
                    return float(kv[k])
                except ValueError:
                    continue
        return None

    # Try several possible AUC key names
    val_auc = pick_float(["val_auc", "val_roc", "valAUC", "val"])
    test_auc = pick_float(["test_auc", "test_roc", "testAUC", "test"])

    if test_auc is None:
        # If we can't find test AUC, this line isn't useful for us
        return None
    if val_auc is None:
        val_auc = test_auc  # fallback

    return {
        "config": config,
        "seed": seed,
        "val_auc": val_auc,
        "test_auc": test_auc,
    }


def collect_results(pattern, label):
    """
    pattern: glob pattern (e.g. 'logs/baseline_*.log' or 'logs/bunn_*.log')
    label: label for this group in the printout.
    """
    files = sorted(glob.glob(pattern))
    print(f"\n=== {label}: found {len(files)} log files matching '{pattern}' ===")
    if not files:
        return

    results_by_config = {}

    for path in files:
        with open(path, "r") as f:
            lines = f.readlines()

        parsed = None
        # Scan from bottom up, so we get the last FINAL line in the file
        for line in reversed(lines):
            if "FINAL" in line:
                parsed = parse_final_line(line)
                if parsed is not None:
                    break

        if parsed is None:
            print(f"  [WARN] No parsable 'FINAL' line found in {path}")
            continue

        cfg = parsed["config"]
        seed = parsed["seed"]
        test_auc = parsed["test_auc"]

        results_by_config.setdefault(cfg, []).append(test_auc)

    if not results_by_config:
        print("  [WARN] No results parsed.")
        return

    print(f"\n  {'Config':<20} {'#Seeds':<8} {'Test AUC (mean ± std)':<25}")
    print("  " + "-" * 60)
    for cfg, aucs in sorted(results_by_config.items()):
        arr = np.array(aucs, dtype=float)
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(f"  {cfg:<20} {len(arr):<8d} {mean:.4f} ± {std:.4f}")


def main():
    # 1) All baseline models (GCN, MLP, SAGE variants, GAT-sep if present, etc.)
    collect_results("logs/baseline_*.log", label="Baseline models")

    # 2) All BuNN configs (small / medium / big_dense / big_sparse / etc.)
    collect_results("logs/bunn_*.log", label="BuNN configs")


if __name__ == "__main__":
    main()

