# Load sloop trials and convert them into sloop trial objects

import os
import pickle
import copy
from sloop.oopomdp.experiments.trial import SloopPriorTrial
from sloop.oopomdp.experiments.states_result import StatesResult

exp_path = "../../../../results/all-joint"

for trial_name in os.listdir(exp_path):
    if not os.path.isdir(os.path.join(exp_path, trial_name)):
        continue
    # with open(os.path.join(exp_path, trial_name, "trial.pkl"), "rb") as f:
    #     trial = pickle.load(f)
    # sloop_trial = SloopPriorTrial(trial.name, trial.config, verbose=True)
    # with open(os.path.join(exp_path, trial_name, "sloop_trial.pkl"), "wb") as f:
    #     pickle.dump(sloop_trial, f)

    with open(os.path.join(exp_path, trial_name, "states.pkl"), "rb") as f:
        trial = pickle.load(f)
    sloop_trial = SloopPriorTrial(trial.name, trial.config, verbose=True)
    with open(os.path.join(exp_path, trial_name, "sloop_states.pkl"), "wb") as f:
        pickle.dump(sloop_trial, f)

    print(trial_name)
