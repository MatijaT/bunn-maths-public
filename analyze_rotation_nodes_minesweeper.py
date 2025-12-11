#!/usr/bin/env python
"""
Analyze where BuNN rotations matter on Minesweeper.

Trains three medium-size BuNN variants with MLP φ on Minesweeper:

  - "full":        learned rotations with rotate-back
  - "no_rotate":   learned rotations but do NOT rotate back
  - "identity":    identity rotations (angles = 0)

Then saves per-node and per-edge data to a .npz file:

Per-node (length N):
  - y:              labels
  - train_mask, val_mask, test_mask
  - degree:         node degrees
  - heterophily:    fraction of neighbors with different label
  - prob_full:      predicted P(y=1) for full model
  - prob_noRB:      predicted P(y=1) for no-rotate-back model
  - prob_identity:  predicted P(y=1) for identity model
  - delta_noRB:     |prob_full - prob_noRB|
  - delta_identity: |prob_full - prob_identity|

Per-edge (length E):
  - edge_src, edge_dst: endpoints of each edge
  - edge_heterophily:   1 if y[src] != y[dst], else 0
  - edge_delta_noRB:     0.5 * (delta_noRB[src] + delta_noRB[dst])
  - edge_delta_identity: 0.5 * (delta_identity[src] + delta_identity[dst])

Usage (single GPU node):

  python analyze_rotation_nodes_minesweeper.py \
      --seed 0 \
      --device cuda \
      --output rotation_analysis/rotation_seed0.npz
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.utils import to_dense_adj


# ----------------------------------------------------------------------
# Config and utils
# ----------------------------------------------------------------------


@dataclass
class BuNNConfig:
    num_bundles: int = 32
    num_layers: int = 4
    phi_layers: int = 4
    phi_hidden: Optional[int] = None
    K: int = 4
    t: float = 1.0
    dropout: float = 0.2
    lr: float = 3e-4
    epochs: int = 800
    rotation_mode: str = "learned"  # "learned", "identity", "no_rotate_back"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_minesweeper(device: torch.device, root: str = "/tmp/heterophilic"):
    dataset = HeterophilousGraphDataset(root=root, name="Minesweeper")
    data = dataset[0]

    # Use first split if multiple
    if data.train_mask.dim() > 1:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    data = data.to(device)
    num_classes = dataset.num_classes
    return data, num_classes


def compute_laplacian(edge_index: torch.Tensor, num_nodes: int, device: torch.device):
    """
    Dense random-walk Laplacian: L = I - D^{-1} A
    """
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
    degree = A.sum(dim=1)
    D_inv = torch.diag(1.0 / (degree + 1e-10))
    L = torch.eye(num_nodes, device=device) - D_inv @ A
    return L


# ----------------------------------------------------------------------
# Phi network: simple MLP on bundle features
# ----------------------------------------------------------------------


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
        return self.net(h)  # (N, num_bundles) of angles


# ----------------------------------------------------------------------
# BuNN model (MLP φ only, medium config)
# ----------------------------------------------------------------------


class BuNN(nn.Module):
    def __init__(self, d_in: int, d_out: int, config: BuNNConfig):
        super().__init__()
        self.config = config
        self.num_bundles = config.num_bundles
        self.num_layers = config.num_layers
        self.t = config.t
        self.K = config.K
        self.rotation_mode = config.rotation_mode

        self.d_bundle = 2
        self.total_dim = 2 * self.num_bundles

        self.input_embed = nn.Linear(d_in, self.total_dim)

        # φ networks: one per layer
        self.phi_nets = nn.ModuleList(
            [
                PhiMLP(
                    d_in=self.total_dim,
                    num_bundles=self.num_bundles,
                    num_layers=config.phi_layers,
                    hidden_dim=config.phi_hidden,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Per-layer linear transforms
        self.layer_transforms = nn.ModuleList(
            [nn.Linear(self.total_dim, self.total_dim) for _ in range(self.num_layers)]
        )

        self.dropout = nn.Dropout(config.dropout)
        self.output_linear = nn.Linear(self.total_dim, d_out)

        self.register_buffer("L", None)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---- Laplacian ----
    def precompute_laplacian(self, L: torch.Tensor):
        self.L = L

    # ---- Rotations ----
    def apply_bundle_rotations(self, h: torch.Tensor, angles: torch.Tensor, transpose: bool = False):
        """
        h: (N, 2*B), angles: (N, B)
        Interpret (h[:, 2k], h[:, 2k+1]) as 2D vectors.
        """
        N, D = h.shape
        B = self.num_bundles
        assert D == 2 * B, f"Expected 2 * num_bundles, got {D}"

        h_reshaped = h.view(N, B, 2)
        x = h_reshaped[:, :, 0]
        y = h_reshaped[:, :, 1]

        cos_t = torch.cos(angles)
        sin_t = torch.sin(angles)

        if transpose:
            # inverse rotation
            x_rot = cos_t * x - sin_t * y
            y_rot = sin_t * x + cos_t * y
        else:
            # forward rotation
            x_rot = cos_t * x + sin_t * y
            y_rot = -sin_t * x + cos_t * y

        out = torch.stack([x_rot, y_rot], dim=-1).view(N, 2 * B)
        return out

    # ---- Heat diffusion (Taylor) ----
    def heat_diffusion_taylor(self, H: torch.Tensor):
        if self.L is None:
            raise RuntimeError("Laplacian not set. Call precompute_laplacian first.")

        result = H.clone()
        term = H.clone()
        for k in range(1, self.K + 1):
            term = (-self.t / k) * (self.L @ term)
            result = result + term
        return result

    # ---- Forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_embed(x)
        N = x.shape[0]

        for layer_idx in range(self.num_layers):
            phi_net = self.phi_nets[layer_idx]

            angles = phi_net(h)  # (N, B)

            if self.rotation_mode == "identity":
                angles = torch.zeros_like(angles)

            # Linear in rotated bundle space + heat diffusion
            W = self.layer_transforms[layer_idx].weight  # (D, D)
            b = self.layer_transforms[layer_idx].bias    # (D,)

            # Rotate into bundle frame
            h_bundle = self.apply_bundle_rotations(h, angles, transpose=True)
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


# ----------------------------------------------------------------------
# Training + analysis
# ----------------------------------------------------------------------


def train_bunn_model(
    config: BuNNConfig,
    seed: int,
    device: torch.device,
    logger: logging.Logger,
    data_root: str,
):
    logger.info(f"Training BuNN, rotation_mode={config.rotation_mode}, seed={seed}")

    set_seed(seed)

    # Load data
    data, num_classes = load_minesweeper(device, root=data_root)
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
    model = BuNN(d_in=data.num_node_features, d_out=num_classes, config=config).to(device)
    model.precompute_laplacian(L)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"BuNN parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_auc = 0.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Evaluate periodically
        if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                out = model(x)
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
                    best_epoch = epoch
                    # Save best state (on CPU to save GPU memory)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                elapsed = (time.time() - start_time) / 60.0
                eta = elapsed / epoch * (config.epochs - epoch) if epoch > 1 else 0.0

                logger.info(
                    f"[{config.rotation_mode:>13}] Epoch {epoch:4d}/{config.epochs} | "
                    f"Loss {loss:.4f} | "
                    f"TrainAcc {train_acc:.3f} | "
                    f"ValAcc {val_acc:.3f} | "
                    f"TestAcc {test_acc:.3f} | "
                    f"ValAUC {val_auc:.4f} | "
                    f"TestAUC {test_auc:.4f} | "
                    f"BestValAUC {best_val_auc:.4f} @ {best_epoch} | "
                    f"Time {elapsed:.1f}m | ETA {eta:.1f}m"
                )

    total_time = time.time() - start_time
    logger.info(
        f"Done rotation_mode={config.rotation_mode}, seed={seed}, "
        f"best_val_auc={best_val_auc:.4f}, time={total_time/60:.1f}m"
    )

    # Reload best state into model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, data


def compute_node_stats(edge_index: torch.Tensor, y: torch.Tensor, num_nodes: int, device: torch.device):
    """
    Compute degree and heterophily per node.
    Heterophily(v) = (# neighbors with different label) / degree(v)
    """
    edge_index_cpu = edge_index.to("cpu")
    y_cpu = y.to("cpu")

    src, dst = edge_index_cpu
    num_nodes = int(num_nodes)

    degree = torch.zeros(num_nodes, dtype=torch.float32)
    diff_count = torch.zeros(num_nodes, dtype=torch.float32)

    for u, v in zip(src.tolist(), dst.tolist()):
        degree[u] += 1.0
        if y_cpu[u] != y_cpu[v]:
            diff_count[u] += 1.0

    heterophily = torch.zeros(num_nodes, dtype=torch.float32)
    mask = degree > 0
    heterophily[mask] = diff_count[mask] / degree[mask]

    return degree.to(device), heterophily.to(device)


def compute_edge_stats(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    delta_noRB: torch.Tensor,
    delta_id: torch.Tensor,
):
    """
    Compute per-edge stats:

      edge_src, edge_dst
      edge_heterophily = 1{ y[u] != y[v] }
      edge_delta_noRB   = 0.5 * (delta_noRB[u] + delta_noRB[v])
      edge_delta_id     = 0.5 * (delta_id[u] + delta_id[v])
    """
    edge_index_cpu = edge_index.to("cpu")
    y_cpu = y.to("cpu").numpy()
    delta_noRB_cpu = delta_noRB.to("cpu").numpy()
    delta_id_cpu = delta_id.to("cpu").numpy()

    src, dst = edge_index_cpu
    src_np = src.numpy()
    dst_np = dst.numpy()

    edge_heterophily = (y_cpu[src_np] != y_cpu[dst_np]).astype(np.int64)
    edge_delta_noRB = 0.5 * (delta_noRB_cpu[src_np] + delta_noRB_cpu[dst_np])
    edge_delta_id = 0.5 * (delta_id_cpu[src_np] + delta_id_cpu[dst_np])

    return src_np, dst_np, edge_heterophily, edge_delta_noRB, edge_delta_id


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output",
        type=str,
        default="rotation_analysis/rotation_seed0.npz",
        help="Path to save per-node/edge analysis (.npz)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/tmp/heterophilic",
        help="Root directory for HeterophilousGraphDataset cache.",
    )
    args = parser.parse_args()

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Simple logging to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Device: {device.type}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {args.output}")

    # Base config for "medium" BuNN
    base_config = BuNNConfig(
        num_bundles=32,
        num_layers=4,
        phi_layers=4,
        phi_hidden=None,
        K=4,
        t=1.0,
        dropout=0.2,
        lr=3e-4,
        epochs=800,
        rotation_mode="learned",
    )

    # Train three variants with the SAME seed
    configs = {
        "full": base_config,
        "noRB": BuNNConfig(**{**base_config.__dict__, "rotation_mode": "no_rotate_back"}),
        "identity": BuNNConfig(**{**base_config.__dict__, "rotation_mode": "identity"}),
    }

    models = {}
    data = None

    for name, cfg in configs.items():
        # Re-seed so differences come from architecture, not init
        set_seed(args.seed)
        model, data = train_bunn_model(cfg, args.seed, device, logger, data_root=args.data_root)
        models[name] = model

    assert data is not None
    x = data.x
    y = data.y
    edge_index = data.edge_index
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # Compute node stats
    degree, heterophily = compute_node_stats(edge_index, y, data.num_nodes, device)

    # Get probabilities from each model
    node_probs = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(x)
            prob = F.softmax(out, dim=1)[:, 1]
        node_probs[name] = prob.detach().cpu()

    prob_full = node_probs["full"]
    prob_noRB = node_probs["noRB"]
    prob_id = node_probs["identity"]

    delta_noRB = (prob_full - prob_noRB).abs()
    delta_id = (prob_full - prob_id).abs()

    # Some quick summary stats
    logger.info(f"Mean |Δ_full-noRB| over all nodes: {delta_noRB.mean().item():.4f}")
    logger.info(f"Mean |Δ_full-identity| over all nodes: {delta_id.mean().item():.4f}")

    # Compute edge-level stats
    edge_src, edge_dst, edge_heterophily, edge_delta_noRB, edge_delta_id = compute_edge_stats(
        edge_index, y, delta_noRB, delta_id
    )

    # Save everything to npz
    result = {
        "seed": np.array([args.seed], dtype=np.int64),
        # Node-level
        "y": y.detach().cpu().numpy(),
        "train_mask": train_mask.detach().cpu().numpy(),
        "val_mask": val_mask.detach().cpu().numpy(),
        "test_mask": test_mask.detach().cpu().numpy(),
        "degree": degree.detach().cpu().numpy(),
        "heterophily": heterophily.detach().cpu().numpy(),
        "prob_full": prob_full.numpy(),
        "prob_noRB": prob_noRB.numpy(),
        "prob_identity": prob_id.numpy(),
        "delta_noRB": delta_noRB.numpy(),
        "delta_identity": delta_id.numpy(),
        # Edge-level
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_heterophily": edge_heterophily,
        "edge_delta_noRB": edge_delta_noRB,
        "edge_delta_identity": edge_delta_id,
    }

    np.savez(args.output, **result)
    logger.info(f"Saved per-node and per-edge analysis to {args.output}")


if __name__ == "__main__":
    main()
