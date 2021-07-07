# This script extracts unique languages from a experiment
# path for a specific prior type. This is useful when
# you want to add new baseline but want to use languages
# from existing trials.

import os
import pickle
import json

lang_to_dicts = {}
for map_name in {"austin", "cleveland", "honolulu", "denver", "washington_dc"}:
    dirpath = "../../../datasets/SL-OSM-Dataset/sg_parsed_annotated/%s" % map_name
    for filename in os.listdir(dirpath):
        with open(os.path.join(dirpath, filename)) as f:
            sg = json.load(f)
        lang_to_dicts[sg["lang_original"]] = sg


prior_type = "rule#based#ego>ctx>foref>angle"
# path = "/media/kz-wd-ssd/repo/spatial-foref/sloop/oopomdp/experiments/results/sep9/all-joint" #aug8-12_summary/LANGUAGES2/auto-parse2"
path = "../results/NewEvalpomdpAA--onetarget_20210331003649075"
# outdir = "./aug8-12-languages2"
outdir = "./march31_2021-languages-%s" % prior_type
uniq_langs = set()
for filename in sorted(os.listdir(path)):
    if not os.path.isdir(os.path.join(path, filename)):
        continue
    if filename.split("_")[2].split("-")[0] == prior_type:
        with open(os.path.join(path, filename, "trial.pkl"), "rb") as f:
            trial = pickle.load(f)
        lang = trial.config["language"]
        if lang in uniq_langs:
            continue
        else:
            uniq_langs.add(lang)
        map_name = trial.config["map_name"]
        seed = filename.split("_")[1]

        os.makedirs(os.path.join(outdir, map_name), exist_ok=True)
        with open(os.path.join(outdir, map_name, "sg-%s-%s.json" % (map_name, seed)), "w") as f:
            try:
                json.dump(lang_to_dicts[lang],
                          f, indent=4, sort_keys=True)
            except KeyError:
                import pdb; pdb.set_trace()
