import argparse
import logging
import os

import pandas as pd
import yaml

from config import paths
from src import training, utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain the LAMP entries of a paper table and compare them with the paper"
    )
    parser.add_argument("table", help="Table file, e.g. paper/clc.yaml")
    parser.add_argument("--entries", nargs="+", help="Only these entries (default: all)")
    training.add_device_arguments(parser)
    return parser.parse_args()


def load_table(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_setting(table, entry):
    dataset = entry.get("dataset", table.get("dataset"))
    return next(
        setting
        for setting in utils.dataset_settings(table["task"], dataset)
        if all(setting[k] == v for k, v in entry.get("setting", {}).items())
    )


def table_dir(table_path):
    name = os.path.splitext(os.path.basename(table_path))[0]
    return os.path.join(paths.EXPERIMENTS_PATH, "paper", name)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    table = load_table(args.table)
    entries = {
        key: entry
        for key, entry in table["entries"].items()
        if not args.entries or key in args.entries
    }

    datasets, runs, run_dirs = {}, [], {}
    for key, entry in entries.items():
        setting = find_setting(table, entry)
        setting_key = (entry.get("dataset"), tuple(sorted(entry.get("setting", {}).items())))
        if setting_key not in datasets:
            datasets[setting_key] = utils.prepare_dataset(table["task"], setting)
        configuration = {
            "Model": entry["Model"],
            "Optimizer": entry["Optimizer"],
            "Override": entry.get("Override", {}),
            "Validation": {"monitor": table["monitor"]},
        }
        out_dir = os.path.join(table_dir(args.table), key)
        run_dirs[key] = [os.path.join(out_dir, f"run_{i}") for i in range(len(entry["seeds"]))]
        runs += [
            {
                "configuration": configuration,
                "defaults": setting["default"],
                "dataset": datasets[setting_key],
                "seed": seed,
                "out_dir": run_dir,
            }
            for seed, run_dir in zip(entry["seeds"], run_dirs[key])
        ]
    training.run_all(runs, args)

    rows = []
    for key, entry in entries.items():
        test = training.summarize(run_dirs[key]).loc["Test"]
        for metric, (paper_mean, paper_std) in entry["paper"].items():
            rows.append(
                {
                    "entry": key,
                    "metric": metric,
                    "mean": test[metric, "mean"],
                    "std": test[metric, "std"],
                    "paper mean": paper_mean,
                    "paper std": paper_std,
                }
            )
    results = pd.DataFrame(rows).set_index(["entry", "metric"])
    results.to_csv(os.path.join(table_dir(args.table), "results.csv"))
    print(results.round(4).to_string())


if __name__ == "__main__":
    main()
