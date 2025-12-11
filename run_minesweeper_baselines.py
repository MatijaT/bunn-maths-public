#!/usr/bin/env python
"""
Minesweeper baselines: GCN, GraphSAGE (with & without skip), MLP, GAT_sep.

Usage (interactive):
    python run_minesweeper_baselines.py --model gcn --seed 0 --device cuda

Models:
    - mlp        : node-wise MLP (no edges)
    - gcn        : standard GCNConv stack
    - gat_sep
    - sage_root  : SAGEConv with root_weight=True (skip / ego path)
    - sage_noroot: SAGEConv with root_weight=False (neighbors-only)
"""

import os
import time
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.metrics import roc_auc_score


# seeding & logging

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str, model_name: str, seed: int):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"baseline_{model_name}_seed{seed}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logging.info(f"Logging to {log_path}")
    return log_path



#data

def load_minesweeper(device: str = "cpu"):
    root = Path(__file__).resolve().parent / "data" / "heterophilic"

    dataset = HeterophilousGraphDataset(root=str(root), name="Minesweeper")
    data = dataset[0]

    # Use split 0 (of 10)
    if data.train_mask.dim() > 1:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    logging.info(f"Loaded Minesweeper: {data.num_nodes} nodes, {data.num_edges} edges")
    logging.info(
        f"Train={int(data.train_mask.sum())}, "
        f"Val={int(data.val_mask.sum())}, "
        f"Test={int(data.test_mask.sum())}"
    )
    logging.info(f"Num features={data.num_node_features}, num classes={dataset.num_classes}")
    logging.info(f"Class distribution (all nodes) = {torch.bincount(data.y).tolist()}")

    data = data.to(device)
    return data, dataset.num_classes


# Models

class MLPNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.net(x)


class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x


class SAGENet(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3,
                 dropout=0.5, root_weight=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, root_weight=root_weight))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, root_weight=root_weight))
        self.convs.append(SAGEConv(hidden_dim, out_dim, root_weight=root_weight))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x

class GATSepNet(nn.Module):
    """
    GAT 'separated' baseline:
      - Multi-head GATConv layers
      - Heads are concatenated in hidden layers, final layer uses a single head
    """
    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_layers=3, dropout=0.5, heads=4):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.heads = heads

        # First layer: R^{in_dim} -> R^{hidden_dim} via (hidden_dim/heads) × heads
        self.convs.append(
            GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True)
        )
        # Middle layers: keep hidden_dim width
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
            )
        # Final layer: concat=False so output has dimension out_dim
        self.convs.append(
            GATConv(hidden_dim, out_dim, heads=1, concat=False)
        )

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x

@dataclass
class BaselineConfig:
    name: str
    model_type: str       
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    epochs: int


BASELINE_CONFIGS = {
    # fairly strong but not insane
    "mlp": BaselineConfig(
        name="mlp",
        model_type="mlp",
        hidden_dim=128,
        num_layers=3,
        dropout=0.5,
        lr=1e-3,
        epochs=800,
    ),
    "gcn": BaselineConfig(
        name="gcn",
        model_type="gcn",
        hidden_dim=128,
        num_layers=3,
        dropout=0.5,
        lr=1e-3,
        epochs=800,
    ),
    "sage_root": BaselineConfig(
        name="sage_root",
        model_type="sage_root",   # GraphSAGE with skip (root_weight=True)
        hidden_dim=128,
        num_layers=3,
        dropout=0.5,
        lr=1e-3,
        epochs=800,
    ),
    "sage_noroot": BaselineConfig(
        name="sage_noroot",
        model_type="sage_noroot", # GraphSAGE with root_weight=False
        hidden_dim=128,
        num_layers=3,
        dropout=0.5,
        lr=1e-3,
        epochs=800,
    ),
    "gat_sep": BaselineConfig(
        name="gat_sep",
        model_type="gat_sep",     # multi-head GAT baseline
        hidden_dim=128,
        num_layers=3,
        dropout=0.5,
        lr=1e-3,
        epochs=800,
    ),
}


def build_model(cfg: BaselineConfig, in_dim: int, out_dim: int) -> nn.Module:
    if cfg.model_type == "mlp":
        return MLPNet(in_dim, cfg.hidden_dim, out_dim,
                      num_layers=cfg.num_layers,
                      dropout=cfg.dropout)
    elif cfg.model_type == "gcn":
        return GCNNet(in_dim, cfg.hidden_dim, out_dim,
                      num_layers=cfg.num_layers,
                      dropout=cfg.dropout)
    elif cfg.model_type == "sage_root":
        return SAGENet(in_dim, cfg.hidden_dim, out_dim,
                       num_layers=cfg.num_layers,
                       dropout=cfg.dropout,
                       root_weight=True)
    elif cfg.model_type == "sage_noroot":
        return SAGENet(in_dim, cfg.hidden_dim, out_dim,
                       num_layers=cfg.num_layers,
                       dropout=cfg.dropout,
                       root_weight=False)
    elif cfg.model_type == "gat_sep":
        return GATSepNet(
            in_dim, cfg.hidden_dim, out_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            heads=4,
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")



def train_and_evaluate(cfg: BaselineConfig, seed: int, device: str):
    set_seed(seed)
    data, num_classes = load_minesweeper(device=device)

    model = build_model(cfg, data.num_node_features, num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Config '{cfg.name}', model_type={cfg.model_type}, seed={seed}")
    logging.info(f"Parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_auc = 0.0
    best_test_auc = 0.0
    best_epoch = 0
    patience = max(200, cfg.epochs // 5)
    patience_counter = 0

    x, y = data.x, data.y
    edge_index = data.edge_index
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
                prob = F.softmax(out, dim=1)[:, 1]
                pred = out.argmax(dim=1)

                train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
                val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
                test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

                val_auc = roc_auc_score(
                    y[val_mask].cpu().numpy(),
                    prob[val_mask].cpu().numpy()
                )
                test_auc = roc_auc_score(
                    y[test_mask].cpu().numpy(),
                    prob[test_mask].cpu().numpy()
                )

                if val_auc > best_val_auc + 1e-4:
                    best_val_auc = val_auc
                    best_test_auc = test_auc
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                elapsed = time.time() - start
                eta = (elapsed / max(1, epoch)) * cfg.epochs - elapsed

                logging.info(
                    f"Epoch {epoch:4d}/{cfg.epochs} | "
                    f"Loss {loss:.4f} | "
                    f"TrainAcc {train_acc:.3f} | ValAcc {val_acc:.3f} | TestAcc {test_acc:.3f} | "
                    f"ValAUC {val_auc:.4f} | TestAUC {test_auc:.4f} | "
                    f"BestValAUC {best_val_auc:.4f} (Test {best_test_auc:.4f} @ {best_epoch}) | "
                    f"Time {elapsed/60:.1f}m | ETA {eta/60:.1f}m"
                )

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.time() - start
    logging.info(
        f"Done: model={cfg.model_type}, seed={seed}, "
        f"best_val_auc={best_val_auc:.4f}, best_test_auc={best_test_auc:.4f}, "
        f"time={total_time/60:.1f}m"
    )
    return best_val_auc, best_test_auc



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(BASELINE_CONFIGS.keys()),
        help="Which baseline model to run (mlp, gcn, sage_root, sage_noroot, gat_sep).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (can come from SLURM_ARRAY_TASK_ID).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu'. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    cfg = BASELINE_CONFIGS[args.model]
    log_path = setup_logging(args.log_dir, cfg.name, args.seed)
    logging.info(f"Device: {device}")
    logging.info(f"Using baseline config: {cfg}")
    logging.info(f"Log file: {log_path}")

    val_auc, test_auc = train_and_evaluate(cfg, args.seed, device)
    logging.info(
        f"FINAL: model={cfg.model_type}, seed={args.seed}, "
        f"val_auc={val_auc:.4f}, test_auc={test_auc:.4f}"
    )


if __name__ == "__main__":
    main()
