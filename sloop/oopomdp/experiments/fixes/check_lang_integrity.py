import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sciex import Experiment
from distutils.dir_util import copy_tree

def main():
    parser = argparse.ArgumentParser(description="Remake trial script splits.")
    parser.add_argument("exp_path", type=str, help="Path to experiments")
    args = parser.parse_args()

    exp_path = args.exp_path
    if args.exp_path.endswith("/"):
        exp_path = os.path.dirname(args.exp_path)

    cases = []

    seed_to_lang = {}
    for name in os.listdir(exp_path):
        if not os.path.isdir(os.path.join(exp_path, name)):
            continue

        global_name, seed, specific_name = name.split("_")
        print("Opening trial %s" % specific_name)
        with open(os.path.join(exp_path, name, "trial.pkl"), "rb") as f:
            trial = pickle.load(f)
        lang = trial.config["language"]
        map_name = trial.config["map_name"]
        if (map_name, seed) not in seed_to_lang:
            seed_to_lang[(map_name, seed)] = lang
        else:
            if seed_to_lang[(map_name, seed)] != lang:
                cases.append((name, seed_to_lang[(map_name, seed)], lang))

    for name, expected, actual in cases:
        print("%s has unexpected language" % name)
        print("    Expected: \"%s\"" % expected)
        print("    In-trial: \"%s\"" % actual)
            
if __name__ == "__main__":
    main()
