# Given paths to two experiment folders,
# output a change of the seeds

import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sciex import Experiment

path1 = "../results/sep7/all-joint"
path2 = "../results/sep9/annotated"

def main():
    parser = argparse.ArgumentParser(description="Remake trial script splits.")
    parser.add_argument("path1", type=str, help="Path to output fixed trials")
    parser.add_argument("path2", type=str, help="Path to output fixed trials")        
    args = parser.parse_args()

    seed_to_lang = {}

    allmaps = {}
    cases = set()
    for exp_path in [args.path1, args.path2]:
        lang_to_targets = {}
        for name in os.listdir(exp_path):
            if not os.path.isdir(os.path.join(exp_path, name)):
                continue

            global_name, seed, specific_name = name.split("_")
            print("Opening trial %s" % specific_name)
            with open(os.path.join(exp_path, name, "trial.pkl"), "rb") as f:
                trial = pickle.load(f)
            lang = trial.config["language"]
            map_name = trial.config["map_name"]
            world = trial.config["world"]
            for ch in {"R", "G", "B"}:
                if ch in world:
                    break
            if lang not in lang_to_targets:
                lang_to_targets[lang] = {}
            if ch not in lang_to_targets[lang]:
                lang_to_targets[lang][ch] = []
            lang_to_targets[lang][ch].append(name)
            cases.add((map_name,lang,ch))
        allmaps[exp_path] = lang_to_targets

    with open("seed_mapping.txt", "w") as f:
        for map_name, lang, ch in sorted(cases, key=lambda x: x[0]):
            seeds1 = set()
            for name in allmaps[args.path1][lang][ch]:
                global_name, seed, specific_name = name.split("_")
                seeds1.add(seed)
            seeds2 = set()
            for name in allmaps[args.path2][lang][ch]:
                global_name, seed, specific_name = name.split("_")
                seeds2.add(seed)
            f.write("%s: %s --> %s\n" % (map_name, str(seeds1), str(seeds2)))

if __name__ == "__main__":
    main()
