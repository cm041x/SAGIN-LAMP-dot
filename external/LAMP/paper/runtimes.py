import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from paper.reproduce import find_setting, load_table
from src import training, utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Table 7: training and inference time per epoch on ColoredLeafCounting"
    )
    parser.add_argument("--table", default="paper/clc.yaml")
    parser.add_argument("--entries", nargs="+", default=["T200_K3", "T400_K3", "T600_K3"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def timed(function, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    table = load_table(args.table)

    rows = []
    for key in args.entries:
        entry = table["entries"][key]
        setting = find_setting(table, entry)
        dataset = utils.prepare_dataset(table["task"], setting)
        configuration = {"Model": entry["Model"], "Optimizer": entry["Optimizer"]}
        training.set_seed(entry["seeds"][0])
        model = training.build_model(configuration, dataset, device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=entry["Optimizer"]["learning_rate"],
            weight_decay=entry["Optimizer"]["weight_decay"],
        )
        criterion = training.SCORES[setting["default"]["criterion"]]
        train = DataLoader(
            torch.arange(len(dataset.train_data)), batch_size=args.batch_size, shuffle=True
        )
        n_test = len(dataset.test_data)
        test = DataLoader(torch.arange(n_test), batch_size=n_test)

        def train_epoch():
            training.train_epoch(model, dataset, train, optimizer, criterion, device)

        def inference():
            training.evaluate(model, dataset, "test", test, criterion, [], device)

        train_epoch()
        inference()
        epoch_times = [timed(train_epoch, device) for _ in range(args.epochs)]
        inference_times = [timed(inference, device) for _ in range(args.epochs)]
        rows.append(
            {
                "entry": key,
                "epoch time (s)": f"{np.mean(epoch_times):.2f} ± {np.std(epoch_times, ddof=1):.2f}",
                "inference time (s)": f"{np.mean(inference_times):.2f} ± {np.std(inference_times, ddof=1):.2f}",
            }
        )
    print(pd.DataFrame(rows).set_index("entry").to_string())


if __name__ == "__main__":
    main()
