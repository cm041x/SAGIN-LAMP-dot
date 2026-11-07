import argparse
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.reproduce import find_setting, load_table, table_dir
from src import training, utils
from src.layers.convolutions.lamp_conv import normalized_adjacency

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Figures 2-4: Jacobian spectrum, stability ratio and input-to-state stability"
    )
    parser.add_argument("figure", choices=["eigenvalues", "stability", "iss"])
    parser.add_argument("table", help="Table file of the trained entry, e.g. paper/clc.yaml")
    parser.add_argument("entry", help="Entry of the table, e.g. T400_K3")
    parser.add_argument("--run", type=int, default=0, help="Which seed of the entry to load")
    parser.add_argument("--steps", type=int, default=50, help="Input scales for the ISS curve")
    parser.add_argument("--max-scale", type=float, default=50.0)
    return parser.parse_args()


def load_trained_model(table_path, entry_name, run):
    table = load_table(table_path)
    setting = find_setting(table, table["entries"][entry_name])
    dataset = utils.prepare_dataset(table["task"], setting)
    run_dir = os.path.join(table_dir(table_path), entry_name, f"run_{run}")
    with open(os.path.join(run_dir, "hyperparameters.json")) as f:
        configuration = json.load(f)
    model = training.build_model(configuration, dataset, torch.device("cpu"))
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    model.eval()
    return model.conv, dataset, run_dir


def jacobian(conv, x, h, edge_index, edge_weight):
    h = conv.impulse_map(x, h)
    n = h.shape[0]
    eye = torch.eye(n)
    W = conv.W - conv.W.T
    V = conv.S.linear.weight - conv.S.linear.weight.T
    Z = conv.K.linear.weight + conv.K.linear.weight.T
    sym = torch.sparse_coo_tensor(*normalized_adjacency(edge_index, edge_weight, "sym"), (n, n))
    antisym = normalized_adjacency(edge_index, edge_weight, "antisym")
    antisym = torch.sparse_coo_tensor(*antisym, (n, n))
    M = (
        torch.kron(-W, eye)
        - torch.kron(-V, sym.to_dense())
        + conv.beta * torch.kron(Z, antisym.to_dense())
        - torch.kron(conv.mixing.dissipativity(), eye)
    )
    pre_activation = M @ h.T.reshape(-1)
    return torch.diag(1 - torch.tanh(pre_activation) ** 2) @ M


def snapshots(dataset, split):
    data = getattr(dataset, f"{split}_data")
    return list(data[0]) if dataset.SEQUENCE_LEVEL else list(data)


@torch.no_grad()
def eigenvalues(conv, dataset, out_dir, count=3):
    h = None
    for t, snapshot in enumerate(snapshots(dataset, "train")[:count]):
        if h is None:
            h = torch.zeros(snapshot.x.shape[0], conv.W.shape[0])
        edge_weight = getattr(snapshot, "edge_attr", None)
        J = jacobian(conv, snapshot.x, h, snapshot.edge_index, edge_weight)
        eigs = torch.linalg.eigvals(J).numpy()
        h = conv(snapshot.x, h, snapshot.edge_index, edge_weight=edge_weight)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(eigs.real, eigs.imag, s=20, alpha=0.65, color="#1a2a8f", edgecolors="white")
        ax.axvline(0, color="r", linestyle="--", alpha=0.6)
        ax.axhline(0, color="0.5", linestyle="--", alpha=0.4)
        ax.annotate(
            rf"$\max\,\mathrm{{Re}}(\lambda) = {eigs.real.max():.3g}$",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.7"),
        )
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        fig.savefig(os.path.join(out_dir, f"eigenvalues_t{t}.pdf"), bbox_inches="tight")
        plt.close(fig)


@torch.no_grad()
def stability(conv, dataset, out_dir):
    ratios, h = [], None
    for snapshot in snapshots(dataset, "train"):
        if h is None:
            h = torch.zeros(snapshot.x.shape[0], conv.W.shape[0])
        edge_weight = getattr(snapshot, "edge_attr", None)
        J = jacobian(conv, snapshot.x, h, snapshot.edge_index, edge_weight)
        log_norm = torch.linalg.eigvalsh((J + J.T) / 2).max().item()
        ratios.append(np.exp(log_norm * conv.epsilon * conv.L) / conv.xi)
        h = conv(snapshot.x, h, snapshot.edge_index, edge_weight=edge_weight)
    np.save(os.path.join(out_dir, "stability_ratio.npy"), np.array(ratios))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios, color="#1a2a8f", label=r"$R_t$")
    ax.axhline(1.0, color="r", linestyle="--", label=r"$R_t = 1$")
    ax.set_xlabel("Time Step $t$")
    ax.set_ylabel(r"Ratio $R_t = e^{\omega \epsilon L} / \xi$")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "stability_ratio.pdf"), bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def iss(conv, dataset, out_dir, steps, max_scale):
    test = snapshots(dataset, "test")
    scales = np.linspace(0, max_scale, steps)
    peaks, finals = [], []
    for scale in scales:
        h = torch.zeros(test[0].x.shape[0], conv.W.shape[0])
        peak = 0.0
        for snapshot in test:
            edge_weight = getattr(snapshot, "edge_attr", None)
            h = conv(snapshot.x * scale, h, snapshot.edge_index, edge_weight=edge_weight)
            peak = max(peak, torch.norm(h).item())
        peaks.append(peak)
        finals.append(torch.norm(h).item())
    np.savez(os.path.join(out_dir, "iss.npz"), scales=scales, peaks=peaks, finals=finals)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scales, peaks, label=r"Peak $\sup_t \|H(t)\|_F$")
    ax.plot(scales, finals, linestyle="--", label=r"Terminal $\|H(T)\|_F$")
    ax.set_xlabel(r"Input Scaling Factor $\alpha$")
    ax.set_ylabel(r"$\|H(t)\|_F$")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "iss.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    conv, dataset, run_dir = load_trained_model(args.table, args.entry, args.run)
    out_dir = os.path.join(run_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    if args.figure == "eigenvalues":
        eigenvalues(conv, dataset, out_dir)
    elif args.figure == "stability":
        stability(conv, dataset, out_dir)
    else:
        iss(conv, dataset, out_dir, args.steps, args.max_scale)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
