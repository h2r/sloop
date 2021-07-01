import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sciex import Experiment
from distutils.dir_util import copy_tree

path1 = "../results/sep7/all-joint"
path2 = "../results/sep9/annotated"

def main():
    parser = argparse.ArgumentParser(description="Remake trial script splits.")
    # parser.add_argument("exp_path", type=str, help="Path to experiments")
    parser.add_argument("out_path", type=str, help="Path to output fixed trials")    
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    # exp_path = args.exp_path
    # if args.exp_path.endswith("/"):
    #     exp_path = os.path.dirname(args.exp_path)

    seed_to_lang = {}

    lang_to_targets = {}
    for exp_path in [path1, path2]:
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
            if map_name not in lang_to_targets:
                lang_to_targets[map_name] = {}
            if lang not in lang_to_targets[map_name]:
                lang_to_targets[map_name][lang] = {}
            if ch not in lang_to_targets[map_name][lang]:
                lang_to_targets[map_name][lang][ch] = []
            lang_to_targets[map_name][lang][ch].append((exp_path, name))

    for map_name in lang_to_targets:
        seed_counter = 0
        for lang in sorted(lang_to_targets[map_name]):
            assert len(lang_to_targets[map_name][lang]) == 2
            for target in lang_to_targets[map_name][lang]:
                for exp_path, name in lang_to_targets[map_name][lang][target]:
                    global_name, seed, specific_name = name.split("_")
                    new_name = "%s_%02d_%s" % (global_name, seed_counter, specific_name)
                    copy_tree(os.path.join(exp_path, name),
                              os.path.join(args.out_path, new_name))
                    print("Copied trial to %s" % new_name)
                seed_counter += 1


if __name__ == "__main__":
    main()
