import argparse
import logging
import os

from config import paths
from src import training, utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model selection: train every configuration of a hyperparameter grid"
    )
    parser.add_argument("--model", default="LAMP")
    parser.add_argument(
        "--task", required=True, help="leaf_counting, node_prediction or link_prediction"
    )
    parser.add_argument("--dataset", required=True)
    training.add_device_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    grid = os.path.join(
        paths.TUNING_HYPERPARAMETERS_PATH, args.task, args.dataset, f"{args.model}.yaml"
    )
    configurations = utils.cartesian_product(utils.load_config(grid))
    monitor = configurations[0]["Validation"]["monitor"]

    runs, model_dirs = [], []
    for dataset_config in utils.dataset_settings(args.task, args.dataset):
        dataset = utils.prepare_dataset(args.task, dataset_config)
        defaults = dataset_config["default"]
        model_dir = utils.run_dir("tuning", args.model, dataset)
        model_dirs.append(model_dir)
        runs += [
            {
                "configuration": configuration,
                "defaults": defaults,
                "dataset": dataset,
                "seed": defaults["seed"],
                "out_dir": os.path.join(model_dir, f"hyperparameters_{i}"),
            }
            for i, configuration in enumerate(configurations)
        ]
    training.run_all(runs, args)

    for model_dir in model_dirs:
        run_dirs = {
            i: os.path.join(model_dir, f"hyperparameters_{i}") for i in range(len(configurations))
        }
        best = training.best_trial(run_dirs, monitor)
        with open(os.path.join(model_dir, "best_run.txt"), "w") as f:
            f.write(str(best))
        logging.info("%s: best configuration is hyperparameters_%s", model_dir, best)


if __name__ == "__main__":
    main()
