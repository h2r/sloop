import argparse
import os
import pickle
from sloop.oopomdp.experiments.experiment_foref_langprior import\
    create_world, make_trial, get_prior, obj_letter_map, obj_id_map
from sloop.oopomdp.experiments.interface import load_files
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop.models.heuristics.model import KeywordModel
import matplotlib.pyplot as plt
import numpy as np
from sciex import Experiment

def main():
    parser = argparse.ArgumentParser(description="regenerate keyword trials.")
    parser.add_argument("exp_path", type=str, help="Path to experiments")
    parser.add_argument("--clear", action="store_true",
                        help="Clear files except for the pickle file in keyword trial directory.")
    args = parser.parse_args()

    all_trials = []
    exp_path = args.exp_path
    if args.exp_path.endswith("/"):
        exp_path = os.path.dirname(args.exp_path)

    for name in os.listdir(exp_path):
        if not os.path.isdir(os.path.join(exp_path, name)):
            continue
        specific_name = name.split("_")[2]

        print("Opening trial %s" % specific_name)
        with open(os.path.join(exp_path, name, "trial.pkl"), "rb") as f:
            trial = pickle.load(f)
        config = trial.config
        config["solver_args"]["num_sims"] = 10000
        config["solver_args"]["planning_time"] = -1.
        config["solver_args"]["max_depth"] = 20
        config["exec_args"]["max_time"] = 1e9
        config["exec_args"]["max_steps"] = 1000
        config["exec_args"]["visualize"] = False
        all_trials.append(trial)

        if args.clear:
            # Remove all files except for trial.pkl
            for filename in os.listdir(os.path.join(exp_path, name)):
                if filename != "trial.pkl":
                    print("REMOVING %s" % os.path.join(exp_path, name, filename))
                    os.remove(os.path.join(exp_path, name, filename))
    if len(all_trials) == 0:
        print("NO TRIALS")
        return

    # Save trials and generate scripts
    exp_name = "%s_Longer" % os.path.basename(exp_path)
    outdir = os.path.abspath(os.path.join(exp_path, os.pardir))
    exp = Experiment(exp_name, all_trials, outdir, verbose=True, add_timestamp=False)
    exp.generate_trial_scripts(split=400, prefix="run_longer", exist_ok=True)
    print("Find multiple computers to run these experiments.")

if __name__ == "__main__":
    main()
