import argparse
import os
import pickle
from sloop.oopomdp.experiments.experiment_foref_langprior import\
    create_world, make_trial, get_prior, obj_letter_map, obj_id_map
from sloop.oopomdp.experiments.interface import load_files
from spatial_lang.data.map_info_dicts.map_info_dataset import MapInfoDataset
from sloop.models.heuristics.model import KeywordModel
import matplotlib.pyplot as plt
import numpy as np
from sciex import Experiment

def plot_prior(prior, mapdim):
    for objid in prior:
        hm = np.zeros((*mapdim, 4))
        for x,y in prior[objid]:
            hm[x,y] = np.array([0.6, 0.2, 0.0, prior[objid][(x,y)]])
        plt.imshow(hm)
    plt.show()
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="regenerate keyword trials.")
    parser.add_argument("exp_path", type=str, help="Path to experiments")
    parser.add_argument("--clear", action="store_true",
                        help="Clear files except for the pickle file in keyword trial directory.")
    args = parser.parse_args()

    # Load spacy etc
    loaded_things = load_files()

    print("Loading mapinfo")
    mapinfo = MapInfoDataset()

    all_trials = []
    exp_path = args.exp_path
    if args.exp_path.endswith("/"):
        exp_path = os.path.dirname(args.exp_path)

    for name in os.listdir(exp_path):
        if not os.path.isdir(os.path.join(exp_path, name)):
            continue
        specific_name = name.split("_")[2]
        if not specific_name.startswith("keyword"):
            continue
        print("Opening trial %s" % specific_name)
        with open(os.path.join(exp_path, name, "trial.pkl"), "rb") as f:
            trial = pickle.load(f)
        config = trial.config
        sg_dict = config["prior_metadata"]["sg_dict"]  # no need to reparse the language
        map_name = config["map_name"]
        if map_name not in mapinfo.landmarks:
            mapinfo.load_by_name(map_name)

        oldprior = config["prior"]

        # Generate new prior
        keyword_model = KeywordModel()
        prior, metadata = get_prior("keyword", keyword_model, sg_dict,
                                    map_name, mapinfo, obj_id_map, **loaded_things)
        config["prior"] = prior
        config["prior_metadata"] = metadata
        all_trials.append(trial)

        # # To verify, uncomment these
        # map_dim = mapinfo.map_dims(map_name)
        # plot_prior(oldprior, map_dim)
        # plot_prior(trial.config["prior"], map_dim)

        if args.clear:
            # Remove all files except for trial.pkl
            for filename in os.listdir(os.path.join(exp_path, name)):
                if filename != "trial.pkl":
                    print("REMOVING %s" % os.path.join(exp_path, name, filename))
                    os.remove(os.path.join(exp_path, name, filename))

    # Save trials and generate scripts
    exp_name = os.path.basename(exp_path)
    outdir = os.path.abspath(os.path.join(exp_path, os.pardir))
    exp = Experiment(exp_name, all_trials, outdir, verbose=True, add_timestamp=False)
    exp.generate_trial_scripts(split=7, prefix="run_keyword", exist_ok=True)
    print("Find multiple computers to run these experiments.")

if __name__ == "__main__":
    main()
