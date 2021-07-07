# This script matches the seeds of trials in `path` with
# the one in reference path: That is, the same problem
# configuration should have the same seed. The problem
# configuration is defined by the language and the target
# object symbol.

import os
import pickle
import json
import shutil


refpath = "/home/kaiyuzh/repo/spatial-foref/sloop/oopomdp/experiments/results/aug8-12_summary/all-joint"
path = "/home/kaiyuzh/repo/spatial-foref/sloop/oopomdp/experiments/results/aug8-12_summary/LANGUAGES2/GGG"
outdir = path.split("/")[-1]

# Get a (map_name, lang, target) -> seed mapping from ref path
mapping = {}
for filename in os.listdir(refpath):
    if not os.path.isdir(os.path.join(refpath, filename)):
        continue

    with open(os.path.join(refpath, filename, "trial.pkl"), "rb") as f:
        trial = pickle.load(f)
    lang = trial.config["language"]
    map_name = trial.config["map_name"]
    seed = filename.split("_")[1]
    for letter in {"R", "G", "B"}:
        if letter in trial.config["world"]:
            target = letter
            break
    mapping[(map_name, lang, letter)] = seed


# Rename seed in trialsfrom the path
for filename in os.listdir(path):
    if not os.path.isdir(os.path.join(path, filename)):
        continue


    # for ff in os.listdir(os.path.join(path, filename)):
    #     if os.path.isdir(os.path.join(path, filename, ff)):
    #         shutil.rmtree(os.path.join(os.path.join(path, filename, ff)))

    with open(os.path.join(path, filename, "trial.pkl"), "rb") as f:
        trial = pickle.load(f)

    lang = trial.config["language"]
    map_name = trial.config["map_name"]
    for letter in {"R", "G", "B"}:
        if letter in trial.config["world"]:
            target = letter
            break
    try:
        seed = mapping[(map_name, lang, letter)]
    except KeyError as ex:
        if lang == 'The green car is next to the Cort building on the West 6th street side. The red car is in the back of the building that houses rumor and spring, on Johnson court.':
            continue
        import pdb; pdb.set_trace()

    new_filename = "%s_%s_%s" % (filename.split("_")[0], seed, filename.split("_")[2])

    src = os.path.join(path, filename)
    dst = os.path.join(outdir, new_filename)
    os.makedirs(outdir, exist_ok=True)
    if src != dst:
        print("Renaming %s to %s" % (src, dst))
        shutil.move(src, dst)

        # trial.name = new_filename
        # with open(os.path.join(path, new_filename, "trial.pkl"), "wb") as f:
        #     pickle.dump(trial, f)
