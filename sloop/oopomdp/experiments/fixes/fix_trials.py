# Load sloop trials and convert them into sloop trial objects

import os
from sloop.oopomdp.experiments.trial import SloopPriorTrial

exp_path = "~/repo/sloop/results/all-joint"

for trial_name in os.listdir(exp_path):
    if not os.path.isdir(os.path.join(exp_path, trial_name)):
        continue
