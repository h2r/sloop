import argparse
import os
import pickle
from spatial_foref.oopomdp.experiments.experiment_foref_langprior import\
    create_world, make_trial, get_prior, obj_letter_map, obj_id_map
from spatial_foref.oopomdp.experiments.interface import load_files
from spatial_lang.data.map_info_dicts.map_info_dataset import MapInfoDataset
from spatial_foref.models.heuristics.model import KeywordModel, GaussianPointModel
from spatial_foref.oopomdp.experiments.constants import *
from spatial_foref.oopomdp.experiments.trial import SloopPriorTrial
import matplotlib.pyplot as plt
import numpy as np
from sciex import Experiment
import copy


def plot_prior(prior, mapdim, factor=100):
    for objid in prior:
        hm = np.zeros((*mapdim, 4))
        for x,y in prior[objid]:
            hm[x,y] = np.array([0.6, 0.2, 0.0, factor*prior[objid][(x,y)]])
        plt.imshow(hm)
    plt.show()
    plt.clf()

def make_name(oldname, prior_type):
    parts = oldname.split("_")
    chunks = parts[2].split("-")
    chunks[0] = prior_type
    parts[2] = "-".join(chunks)
    return "_".join(parts)
    

def main():
    parser = argparse.ArgumentParser(description="generate informed gaussian trials.")
    parser.add_argument("exp_path", type=str, help="Path to experiments")
    args = parser.parse_args()
    
    all_trials = []
    exp_path = args.exp_path
    if args.exp_path.endswith("/"):
        exp_path = os.path.dirname(args.exp_path)

    mapinfo = MapInfoDataset()
        
    for name in os.listdir(exp_path):
        if not os.path.isdir(os.path.join(exp_path, name)):
            continue
        specific_name = name.split("_")[2]
        prior_type = specific_name.split("-")[0]
        if prior_type != "informed":
            continue
        
        print("Opening trial %s" % specific_name)
        with open(os.path.join(exp_path, name, "trial.pkl"), "rb") as f:
            trial = pickle.load(f)
        config = trial.config
        map_name = config["map_name"]
        worldstr = config["world"]

        if map_name not in mapinfo.landmarks:
            mapinfo.load_by_name(map_name)

        obj_poses = {}
        for y, line in enumerate(worldstr.split("\n")):
            x = -1
            obj = None
            for letter in {"R", "G", "B"}:
                try:
                    x = line.index(letter)
                    obj = letter_symbol_map[letter]
                    break
                except ValueError:
                    pass
            if x != -1:
                obj_poses[obj] = (x,y)

        print("Building Informed model with variance 5...")
        inf5_model = GaussianPointModel(5)
        inf5_prior, _ = get_prior("informed-5", inf5_model, None, map_name,
                                  mapinfo, obj_poses=obj_poses)
        config5 = copy.deepcopy(config)
        config5["prior"] = inf5_prior
        trial_inf5 = SloopPriorTrial(make_name(name, "informed#5"), config5)

        print("Building Informed model with variance 15...")
        inf15_model = GaussianPointModel(15)
        inf15_prior, _ = get_prior("informed-15", inf15_model, None, map_name,
                                  mapinfo, obj_poses=obj_poses)
        config15 = copy.deepcopy(config)
        config15["prior"] = inf15_prior
        trial_inf15 = SloopPriorTrial(make_name(name, "informed#15"), config15)
        
        # # To verify, uncomment these
        # map_dim = mapinfo.map_dims(map_name)
        # plot_prior(trial_inf5.config["prior"], map_dim, factor=100)
        # plot_prior(trial_inf15.config["prior"], map_dim, factor=200)        
        
        all_trials.append(trial_inf5)
        all_trials.append(trial_inf15)


    # Save trials and generate scripts
    exp_name = "%s_GaussianInformed" % os.path.basename(exp_path)
    outdir = os.path.join(exp_path, os.pardir)
    exp = Experiment(exp_name, all_trials, outdir, verbose=True, add_timestamp=False)
    exp.generate_trial_scripts(split=400, prefix="run_longer", exist_ok=True)
    print("Find multiple computers to run these experiments.")
            
if __name__ == "__main__":
    main()
        

    
