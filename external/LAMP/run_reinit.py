import argparse
import json
import logging
import os

from src import training, utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model assessment: retrain one configuration with several seeds"
    )
    parser.add_argument("--model", default="LAMP")
    parser.add_argument(
        "--task", required=True, help="leaf_counting, node_prediction or link_prediction"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config",
        help="YAML with a single configuration (default: the one selected by main.py)",
    )
    training.add_device_arguments(parser)
    return parser.parse_args()


def selected_configuration(model_name, dataset, config_path):
    if config_path:
        [configuration] = utils.cartesian_product(utils.load_config(config_path))
        return configuration
    model_dir = utils.run_dir("tuning", model_name, dataset)
    with open(os.path.join(model_dir, "best_run.txt")) as f:
        trial = f.read().strip()
    with open(os.path.join(model_dir, f"hyperparameters_{trial}", "hyperparameters.json")) as f:
        configuration = json.load(f)
    del configuration["seed"], configuration["deterministic"]
    return configuration


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    runs, run_dirs = [], {}
    for dataset_config in utils.dataset_settings(args.task, args.dataset):
        dataset = utils.prepare_dataset(args.task, dataset_config)
        defaults = dataset_config["default"]
        configuration = selected_configuration(args.model, dataset, args.config)
        seeds = [defaults["seed"] + i for i in range(defaults["N_REINIT"])]
        model_dir = utils.run_dir("reinit", args.model, dataset)
        run_dirs[model_dir] = [os.path.join(model_dir, f"run_{i}") for i in range(len(seeds))]
        runs += [
            {
                "configuration": configuration,
                "defaults": defaults,
                "dataset": dataset,
                "seed": seed,
                "out_dir": run_dir,
            }
            for seed, run_dir in zip(seeds, run_dirs[model_dir])
        ]
    training.run_all(runs, args)

    for model_dir, dirs in run_dirs.items():
        summary = training.summarize(dirs)
        summary.to_csv(os.path.join(model_dir, "scores.csv"))
        logging.info("%s:\n%s", model_dir, summary)


if __name__ == "__main__":
    main()
