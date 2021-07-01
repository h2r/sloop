import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sciex import Experiment

def main():
    parser = argparse.ArgumentParser(description="Remake trial script splits.")
    parser.add_argument("exp_path", type=str, help="Path to experiments")
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
        all_trials.append(trial)
    if len(all_trials) == 0:
        print("NO TRIALS")
        return
                    
    # Save trials and generate scripts
    exp_name = os.path.basename(exp_path)
    outdir = os.path.join(exp_path, os.pardir)
    exp = Experiment(exp_name, all_trials, outdir, verbose=True, add_timestamp=False)
    exp.generate_trial_scripts(split=400, prefix="run_longer", exist_ok=True)
    print("Find multiple computers to run these experiments.")
            
if __name__ == "__main__":
    main()

