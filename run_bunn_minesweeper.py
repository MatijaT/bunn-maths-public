#!/usr/bin/env python
"""
BuNN on Minesweeper with multiple configs and φ / rotation ablations.

Configs:
  - small
  - medium
  - medium_identity
  - medium_noRB
  - medium_phi_sage_root
  - medium_phi_sage_noroot
  - large (128 bundles, 8 layers, K=8, t=1, 2000 epochs)

Usage example (single run):
  python run_bunn_minesweeper.py --config medium --seed 0 --device cuda --log-dir logs

Usage with Slurm array (your submit_bunn_minesweeper_array.sh):
  CONFIG=medium sbatch --array=0-9 submit_bunn_minesweeper_array.sh
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import SAGEConv


@dataclass
class BuNNConfig:
    name: str
    num_bundles: int
    num_layers: int
    phi_type: str          # 'mlp', 'sage_root', 'sage_noroot'
    phi_layers: int
    phi_hidden: Optional[int]
    K: int                 # Taylor order
    t: float               # diffusion time
    dropout: float
    lr: float
    epochs: int
    rotation_mode: str = "learned"   # 'learned', 'identity', 'no_rotate_back'


def load_minesweeper(device: torch.device, logger: logging.Logger):

    root = Path(__file__).resolve().parent / "data" / "heterophilic"

    dataset = HeterophilousGraphDataset(root=str(root), name="Minesweeper")
    data = dataset[0]

    # Select first split if multiple
    if data.train_mask.dim() > 1:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    logger.info(
        f"Loaded Minesweeper: {data.num_nodes} nodes, {data.num_edges} edges"
    )
    logger.info(
        f"Train={int(data.train_mask.sum())}, "
        f"Val={int(data.val_mask.sum())}, "
        f"Test={int(data.test_mask.sum())}"
    )
    logger.info(
        f"Num features={data.num_node_features}, num classes={dataset.num_classes}"
    )
    logger.info(
        f"Class distribution (all nodes) = "
        f"{torch.bincount(data.y).tolist()}"
    )

    data = data.to(device)
    return data, dataset.num_classes


def compute_laplacian(edge_index: torch.Tensor, num_nodes: int, device: torch.device):
    """
    Dense random-walk Laplacian: L = I - D^{-1} A
    """
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
    degree = A.sum(dim=1)
    D_inv = torch.diag(1.0 / (degree + 1e-10))
    L = torch.eye(num_nodes, device=device) - D_inv @ A
    return L


# φ networks


class PhiMLP(nn.Module):
    def __init__(self, d_in: int, num_bundles: int, num_layers: int, hidden_dim: Optional[int]):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * num_bundles

        layers = [nn.Linear(d_in, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, num_bundles))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (N, d_in) -> angles (N, num_bundles)
        return self.net(h)


class PhiSAGE(nn.Module):
    """
    Graph-based φ using SAGEConv.

    root_weight=True  -> 'sage_root'  (with skip)
    root_weight=False -> 'sage_noroot' (no explicit skip)
    """

    def __init__(self, d_in: int, num_bundles: int, num_layers: int, root_weight: bool):
        super().__init__()
        hidden = 2 * num_bundles
        convs = []
        convs.append(SAGEConv(d_in, hidden, root_weight=root_weight))
        for _ in range(num_layers - 1):
            convs.append(SAGEConv(hidden, hidden, root_weight=root_weight))
        self.convs = nn.ModuleList(convs)
        self.out = nn.Linear(hidden, num_bundles)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = h
        for conv in self.convs:
            x = F.gelu(conv(x, edge_index))
        return self.out(x)



# BuNN model
class BuNN(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        config: BuNNConfig,
    ):
        super().__init__()
        self.config = config
        self.num_bundles = config.num_bundles
        self.num_layers = config.num_layers
        self.t = config.t
        self.K = config.K
        self.rotation_mode = config.rotation_mode

        self.d_bundle = 2
        self.total_dim = 2 * self.num_bundles  # single channel; can be extended to multi-channel

        self.input_embed = nn.Linear(d_in, self.total_dim)

        # φ networks: one per BuNN layer
        self.phi_type = config.phi_type
        phi_nets = []
        for _ in range(self.num_layers):
            if self.phi_type == "mlp":
                phi_nets.append(
                    PhiMLP(self.total_dim, self.num_bundles, config.phi_layers, config.phi_hidden)
                )
            elif self.phi_type in ("sage_root", "sage_noroot"):
                phi_nets.append(
                    PhiSAGE(
                        self.total_dim,
                        self.num_bundles,
                        config.phi_layers,
                        root_weight=(self.phi_type == "sage_root"),
                    )
                )
            else:
                raise ValueError(f"Unknown phi_type: {self.phi_type}")
        self.phi_nets = nn.ModuleList(phi_nets)

        # Per-layer linear transforms in bundle space
        self.layer_transforms = nn.ModuleList(
            [nn.Linear(self.total_dim, self.total_dim) for _ in range(self.num_layers)]
        )

        self.dropout = nn.Dropout(config.dropout)
        self.output_linear = nn.Linear(self.total_dim, d_out)

        # Laplacian buffer
        self.register_buffer("L", None)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #  Laplacian 

    def precompute_laplacian(self, L: torch.Tensor):
        self.L = L  # (N, N)

    #  Rotations 

    def apply_bundle_rotations(self, h: torch.Tensor, angles: torch.Tensor, transpose: bool = False):
 
        N, D = h.shape
        B = self.num_bundles

        h_reshaped = h.view(N, B, 2)           # (N, B, 2)
        x = h_reshaped[:, :, 0]               # (N, B)
        y = h_reshaped[:, :, 1]               # (N, B)

        cos_t = torch.cos(angles)
        sin_t = torch.sin(angles)

        if transpose:
            x_rot = cos_t * x - sin_t * y
            y_rot = sin_t * x + cos_t * y
        else:
            x_rot = cos_t * x + sin_t * y
            y_rot = -sin_t * x + cos_t * y

        out = torch.stack([x_rot, y_rot], dim=-1).view(N, 2 * B)
        return out

    #  Heat diffusion 

    def heat_diffusion_taylor(self, H: torch.Tensor):
        """
        Approximate exp(-t L) H with Taylor series:
          exp(-tL) H ≈ sum_{k=0}^K (-t)^k / k! (L^k H).
        """
        if self.L is None:
            raise RuntimeError("Laplacian not set. Call precompute_laplacian first.")

        result = H.clone()
        term = H.clone()
        for k in range(1, self.K + 1):
            term = (-self.t / k) * (self.L @ term)
            result = result + term
        return result

    #  Forward 

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, d_in)
        edge_index: (2, E)
        """
        h = self.input_embed(x)
        N = x.shape[0]

        for layer_idx in range(self.num_layers):
            phi_net = self.phi_nets[layer_idx]

            # Compute angles
            if self.phi_type == "mlp":
                angles = phi_net(h)  # (N, B)
            else:
                angles = phi_net(h, edge_index)  # (N, B)

            if self.rotation_mode == "identity":
                angles = torch.zeros_like(angles)

            # Linear in rotated bundle space + heat diffusion
            W = self.layer_transforms[layer_idx].weight  # (D, D)
            b = self.layer_transforms[layer_idx].bias    # (D,)

            # Rotate into bundle frame
            h_bundle = self.apply_bundle_rotations(h, angles, transpose=True)  # (N, D)
            H = h_bundle @ W.T + b  # (N, D)

            # Diffuse
            H_diff = self.heat_diffusion_taylor(H)

            if self.rotation_mode == "no_rotate_back":
                h_out = H_diff
            else:
                # Rotate back to global frame
                h_out = self.apply_bundle_rotations(H_diff, angles, transpose=False)

            # Residual + nonlinearity
            h = h + F.gelu(self.dropout(h_out))

        return self.output_linear(h)


# Training loop


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate(config: BuNNConfig, seed: int, device: torch.device):
    logger = logging.getLogger(__name__)

    logger.info(f"Config '{config.name}', seed={seed}")
    set_seed(seed)

    # Load data
    data, num_classes = load_minesweeper(device, logger)
    x = data.x
    y = data.y
    edge_index = data.edge_index
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # Laplacian
    logger.info("Computing Laplacian...")
    L = compute_laplacian(edge_index, data.num_nodes, device)
    logger.info("Laplacian done.")

    # Model
    model = BuNN(
        d_in=data.num_node_features,
        d_out=num_classes,
        config=config,
    ).to(device)
    model.precompute_laplacian(L)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"BuNN parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_auc = 0.0
    best_test_auc = 0.0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        # ---- Train ----
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---- Evaluate ----
        if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
                prob = F.softmax(out, dim=1)[:, 1]
                pred = out.argmax(dim=1)

                def acc(mask):
                    return (pred[mask] == y[mask]).float().mean().item()

                train_acc = acc(train_mask)
                val_acc = acc(val_mask)
                test_acc = acc(test_mask)

                val_auc = roc_auc_score(
                    y[val_mask].detach().cpu().numpy(),
                    prob[val_mask].detach().cpu().numpy(),
                )
                test_auc = roc_auc_score(
                    y[test_mask].detach().cpu().numpy(),
                    prob[test_mask].detach().cpu().numpy(),
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_test_auc = test_auc
                    best_epoch = epoch

                elapsed = (time.time() - start_time) / 60.0
                if epoch > 1:
                    eta = elapsed / epoch * (config.epochs - epoch)
                else:
                    eta = 0.0

                logger.info(
                    f"Epoch {epoch:4d}/{config.epochs} | "
                    f"Loss {loss:.4f} | "
                    f"TrainAcc {train_acc:.3f} | "
                    f"ValAcc {val_acc:.3f} | "
                    f"TestAcc {test_acc:.3f} | "
                    f"ValAUC {val_auc:.4f} | "
                    f"TestAUC {test_auc:.4f} | "
                    f"BestValAUC {best_val_auc:.4f} (Test {best_test_auc:.4f} @ {best_epoch}) | "
                    f"Time {elapsed:.1f}m | ETA {eta:.1f}m"
                )

    total_time = (time.time() - start_time) / 60.0
    logger.info(
        f"Done: config={config.name}, seed={seed}, "
        f"best_val_auc={best_val_auc:.4f}, best_test_auc={best_test_auc:.4f}, "
        f"time={total_time:.1f}m"
    )

    # This line is what analyze_logs.py parses:
    logger.info(
        f"FINAL: config={config.name}, seed={seed}, "
        f"val_auc={best_val_auc:.4f}, test_auc={best_test_auc:.4f}"
    )


def get_config_dict() -> Dict[str, BuNNConfig]:
    """
    Returns a dict of all configs we support.
    """

    return {
        # Small didactic: 16 bundles, 2 layers, φ = MLP, K=4
        "small": BuNNConfig(
            name="small",
            num_bundles=16,
            num_layers=2,
            phi_type="mlp",
            phi_layers=2,
            phi_hidden=None,
            K=4,
            t=1.0,
            dropout=0.0,
            lr=3e-4,
            epochs=400,
            rotation_mode="learned",
        ),

        # Medium: 32 bundles, 4 layers, φ = MLP, K=4
        "medium": BuNNConfig(
            name="medium",
            num_bundles=32,
            num_layers=4,
            phi_type="mlp",
            phi_layers=4,
            phi_hidden=None,
            K=4,
            t=1.0,
            dropout=0.2,
            lr=3e-4,
            epochs=800,
            rotation_mode="learned",
        ),

        # Medium with identity rotations
        "medium_identity": BuNNConfig(
            name="medium_identity",
            num_bundles=32,
            num_layers=4,
            phi_type="mlp",
            phi_layers=4,
            phi_hidden=None,
            K=4,
            t=1.0,
            dropout=0.2,
            lr=3e-4,
            epochs=800,
            rotation_mode="identity",
        ),

        # Medium with no rotate-back
        "medium_noRB": BuNNConfig(
            name="medium_noRB",
            num_bundles=32,
            num_layers=4,
            phi_type="mlp",
            phi_layers=4,
            phi_hidden=None,
            K=4,
            t=1.0,
            dropout=0.2,
            lr=3e-4,
            epochs=800,
            rotation_mode="no_rotate_back",
        ),

        # Medium with φ = SAGE (root_weight=True)
        "medium_phi_sage_root": BuNNConfig(
            name="medium_phi_sage_root",
            num_bundles=32,
            num_layers=4,
            phi_type="sage_root",
            phi_layers=4,
            phi_hidden=None,
            K=4,
            t=1.0,
            dropout=0.2,
            lr=3e-4,
            epochs=800,
            rotation_mode="learned",
        ),

        # Medium with φ = SAGE (root_weight=False)
        "medium_phi_sage_noroot": BuNNConfig(
            name="medium_phi_sage_noroot",
            num_bundles=32,
            num_layers=4,
            phi_type="sage_noroot",
            phi_layers=4,
            phi_hidden=None,
            K=4,
            t=1.0,
            dropout=0.2,
            lr=3e-4,
            epochs=800,
            rotation_mode="learned",
        ),

        # Larger model: 128 bundles, 8 layers, φ = SAGE(root),
        "large": BuNNConfig(
            name="large",
            num_bundles=128,
            num_layers=8,
            phi_type="sage_root",
            phi_layers=8,
            phi_hidden=None,
            K=8,
            t=1.0,
            dropout=0.2,
            lr=3e-5,
            epochs=2000,
            rotation_mode="learned",
        ),
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        help="BuNN config key (small, medium, medium_identity, medium_noRB, "
             "medium_phi_sage_root, medium_phi_sage_noroot, large)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    configs = get_config_dict()
    if args.config not in configs:
        print("Unknown config:", args.config)
        print("Available configs:", ", ".join(sorted(configs.keys())))
        sys.exit(1)

    config = configs[args.config]
    log_path = os.path.join(args.log_dir, f"bunn_{config.name}_seed{args.seed}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Logging to {log_path}")

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device.type}")
    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    logger.info(f"Using config: {config}")
    logger.info(f"Log file: {log_path}")

    train_and_evaluate(config, args.seed, device)


if __name__ == "__main__":
    main()

