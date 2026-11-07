import copy
import json
import logging
import os
import random
import traceback

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import Batch

from src.models import MODELS, DynGESN, EvolveGCN_H

logger = logging.getLogger(__name__)

SPLITS = {"train": "Train", "val": "Validation", "test": "Test"}


def f1(y_pred, y_true):
    return f1_score(y_true.int(), (torch.sigmoid(y_pred) > 0.5).int())


def auroc(y_pred, y_true):
    return roc_auc_score(y_true, torch.sigmoid(y_pred))


SCORES = {
    "MAE": torch.nn.functional.l1_loss,
    "MSE": torch.nn.functional.mse_loss,
    "BinaryClassificationLoss": torch.nn.functional.binary_cross_entropy_with_logits,
    "F1": f1,
    "RocAuc": auroc,
}
HIGHER_IS_BETTER = ("F1", "RocAuc")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(configuration, dataset, device):
    model = MODELS[configuration["Model"]["name"]](
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        level=dataset.LEVEL,
        **configuration["Model"],
    )
    return model.to(device)


def predict_sequences(model, dataset, split, seq_ids, device):
    data = getattr(dataset, f"{split}_data")
    batches = [
        Batch.from_data_list([data[i][t] for i in seq_ids]).to(device)
        for t in range(dataset.sequence_length)
    ]
    if isinstance(model, DynGESN):
        y = model([b.edge_index for b in batches], [b.x for b in batches])
    else:
        h = None
        for batch in batches:
            y, h = model(batch, h)
    y_pred = y[batches[-1].mask]
    return y_pred, batches[-1].y.reshape_as(y_pred)


def predict_snapshots(model, snapshots, device):
    h = None
    y_pred, y_true = [], []
    for snapshot in snapshots:
        snapshot = snapshot.to(device)
        y, h = model(snapshot, h)
        y_pred.append(y)
        y_true.append(snapshot.y)
    y_true = torch.cat(y_true)
    return torch.cat(y_pred).reshape_as(y_true), y_true


def scores_of(y_pred, y_true, criterion, metrics):
    scores = {k: float(SCORES[k](y_pred, y_true)) for k in SCORES if k in metrics}
    return {"Loss": float(criterion(y_pred, y_true)), **scores}


@torch.no_grad()
def evaluate(model, dataset, split, seq_batches, criterion, metrics, device):
    model.eval()
    if not dataset.SEQUENCE_LEVEL:
        model.reset_state()
        y_pred, y_true = predict_snapshots(model, getattr(dataset, f"{split}_data"), device)
        return scores_of(y_pred.cpu(), y_true.cpu(), criterion, metrics)
    y_pred, y_true = [], []
    for seq_ids in seq_batches:
        model.reset_state()
        batch_pred, batch_true = predict_sequences(model, dataset, split, seq_ids, device)
        y_pred.append(batch_pred.cpu())
        y_true.append(batch_true.cpu())
    return scores_of(torch.cat(y_pred), torch.cat(y_true), criterion, metrics)


def train_epoch(model, dataset, batches, optimizer, criterion, device):
    model.train()
    losses = []
    for batch in batches:
        model.reset_state()
        optimizer.zero_grad()
        if dataset.SEQUENCE_LEVEL:
            y_pred, y_true = predict_sequences(model, dataset, "train", batch, device)
        else:
            y_pred, y_true = predict_snapshots(model, batch, device)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def fit(configuration, defaults, dataset, seed, out_dir, device, deterministic=False):
    set_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    model = build_model(configuration, dataset, device)
    optimizer = getattr(torch.optim, configuration["Optimizer"].get("name", "Adam"))(
        model.parameters(),
        lr=configuration["Optimizer"]["learning_rate"],
        weight_decay=configuration["Optimizer"]["weight_decay"],
    )
    criterion = SCORES[defaults["criterion"]]
    metrics = defaults.get("metrics", [])
    override = configuration.get("Override", {})
    batch_size = override.get("batch_size", defaults["batch_size"])
    epochs = override.get("epochs", defaults["epochs"])
    patience = override.get("patience", defaults["patience"])

    seq_batches = {}
    if dataset.SEQUENCE_LEVEL:
        batches = DataLoader(
            torch.arange(len(dataset.train_data)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=isinstance(model, EvolveGCN_H),
        )
        n_val, n_test = len(dataset.val_data), len(dataset.test_data)
        seq_batches = {
            "train": batches,
            "val": DataLoader(torch.arange(n_val), batch_size=n_val),
            "test": DataLoader(torch.arange(n_test), batch_size=n_test),
        }
    else:
        batches = DataLoader(list(dataset.train_data), batch_size=batch_size, collate_fn=list)
    if isinstance(model, DynGESN):
        model.initialize(dataset)
        model = model.to(device)

    best_loss, best_epoch, best_state = np.inf, 0, None
    patience_loss, patience_epoch = np.inf, 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, dataset, batches, optimizer, criterion, device)
        val_loss = evaluate(
            model, dataset, "val", seq_batches.get("val"), criterion, metrics, device
        )["Loss"]
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            logger.warning("Non-finite loss at epoch %d, stopping", epoch + 1)
            break
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if val_loss < patience_loss - defaults["min_delta"]:
            patience_loss, patience_epoch = val_loss, epoch
        if epoch - patience_epoch > patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = pd.DataFrame(
        {
            "Set": name,
            **evaluate(model, dataset, split, seq_batches.get(split), criterion, metrics, device),
        }
        for split, name in SPLITS.items()
    )
    os.makedirs(out_dir, exist_ok=True)
    scores.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    with open(os.path.join(out_dir, "hyperparameters.json"), "w") as f:
        json.dump({**configuration, "seed": seed, "deterministic": deterministic}, f, indent=4)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    logger.info("%s: best epoch %d, val loss %.4f", out_dir, best_epoch + 1, best_loss)
    return scores


def is_done(out_dir, configuration, seed, deterministic):
    scores = os.path.join(out_dir, "scores.csv")
    hyperparameters = os.path.join(out_dir, "hyperparameters.json")
    if not os.path.isfile(scores) or not os.path.isfile(hyperparameters):
        return False
    with open(hyperparameters) as f:
        return json.load(f) == {**configuration, "seed": seed, "deterministic": deterministic}


def fit_or_log(device=None, **run):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    try:
        return fit(device=device or torch.device("cuda"), **run)
    except Exception:
        logger.error("Run %s failed:\n%s", run["out_dir"], traceback.format_exc())


def add_device_arguments(parser):
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=list(range(torch.cuda.device_count())),
        help="GPU ids to spread the runs over (default: every visible GPU)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel runs per GPU")
    parser.add_argument(
        "--cpu", action="store_true", help="Train on CPU, one run at a time (default without GPUs)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministic GPU kernels: identical reruns on the same hardware, about 1.7x slower",
    )


def run_all(runs, args):
    runs = [
        {**run, "deterministic": args.deterministic}
        for run in runs
        if not is_done(run["out_dir"], run["configuration"], run["seed"], args.deterministic)
    ]
    logger.info("%d runs to train", len(runs))
    if args.cpu or not args.gpus:
        return [fit_or_log(torch.device("cpu"), **run) for run in runs]
    if not runs:
        return []
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import ray

    workers_per_gpu = min(-(-len(runs) // len(args.gpus)), args.workers)
    ray.init(
        num_gpus=len(args.gpus),
        num_cpus=len(args.gpus) * workers_per_gpu,
        ignore_reinit_error=True,
    )
    remote_fit = ray.remote(num_gpus=1 / workers_per_gpu, num_cpus=1)(fit_or_log)
    datasets = {}
    for run in runs:
        datasets.setdefault(id(run["dataset"]), ray.put(run["dataset"]))
    return ray.get(
        [remote_fit.remote(**{**run, "dataset": datasets[id(run["dataset"])]}) for run in runs]
    )


def best_trial(run_dirs, monitor):
    pick = max if monitor in HIGHER_IS_BETTER else min
    scores = {}
    for trial, run_dir in run_dirs.items():
        path = os.path.join(run_dir, "scores.csv")
        if os.path.isfile(path):
            value = pd.read_csv(path).set_index("Set").loc["Validation", monitor]
            if np.isfinite(value):
                scores[trial] = value
    return pick(scores, key=scores.get) if scores else None


def summarize(run_dirs):
    paths = [os.path.join(d, "scores.csv") for d in run_dirs]
    runs = pd.concat(pd.read_csv(p) for p in paths if os.path.isfile(p))
    return runs.groupby("Set", sort=False).agg(["mean", "std"])
