# For unique languages (compared to those from a directory of trials),
# bump up their seed value by one digit.

import os
import pickle
import json
import shutil


# Path that contains languages you DON't want to duplicate with.
refpath = "/home/kaiyuzh/repo/spatial-foref/spatial_foref/oopomdp/experiments/results/aug8-12_summary/all-joint"

# Path that contains trials you want to bump up the seeds
path = "/home/kaiyuzh/repo/spatial-foref/spatial_foref/oopomdp/experiments/results/aug8-12_summary/LANGUAGES2/auto-parse2"

# Path to save bumped trials
outdir = path.split("/")[-1]

# Get unique languages from refpath
languages = set()
for filename in os.listdir(refpath):
    if not os.path.isdir(os.path.join(refpath, filename)):
        continue

    with open(os.path.join(refpath, filename, "trial.pkl"), "rb") as f:
        trial = pickle.load(f)
    lang = trial.config["language"]
    languages.add(lang)


# Go through bumping trials
for filename in os.listdir(path):
    if not os.path.isdir(os.path.join(path, filename)):
        continue
    
    with open(os.path.join(path, filename, "trial.pkl"), "rb") as f:
        trial = pickle.load(f)

    if "410" in filename and "washington" in filename:
        import pdb; pdb.set_trace()

    lang = trial.config["language"]
    if lang in languages:
        print("Already contains \"%s\"" % lang)
        continue

    print("Not yet contains \"%s\"" % lang)

    seed = "2%s" % filename.split("_")[1]  # add a '2' in front of every seed
    new_filename = "%s_%s_%s" % (filename.split("_")[0], seed, filename.split("_")[2])

    src = os.path.join(path, filename)
    dst = os.path.join(outdir, new_filename)
    os.makedirs(outdir, exist_ok=True)
    if src != dst:
        print("Renaming %s to %s" % (src, dst))
        # shutil.move(src, dst)
        
        # trial.name = new_filename
        # with open(os.path.join(path, new_filename, "trial.pkl"), "wb") as f:
        #     pickle.dump(trial, f)    
