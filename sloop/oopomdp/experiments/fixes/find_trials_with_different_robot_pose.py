"""
Checks if the robot pose are equal for the trials under the same
test case (i.e. map name and seed). If not, for each robot pose
that was used for a test case, output the trials that use that
robot pose and count how many there are.
"""

import os
import json
import pickle
import yaml
from spatial_foref.oopomdp.env.env import *
from spatial_foref.oopomdp.experiments.constants import obj_id_map
from pprint import pprint
import sys

# results_path = "/media/kz-wd-ssd/repo/spatial-foref/spatial_foref/oopomdp/experiments/results/sep9/all-joint"
results_path = "/media/kz-wd-ssd/repo/spatial-foref/spatial_foref/oopomdp/experiments/results/april-1-2021/all-joint"
robot_poses_path = "./april1_2021-robot_poses"  # output
os.makedirs(robot_poses_path, exist_ok=True)

# Map from case (map_name, seed) to {robot_pose -> [trials]}
case_to_robotpose = {}


trial_filenames = []
for filename in sorted(os.listdir(results_path)):
    if not os.path.isdir(os.path.join(results_path, filename)):
        continue
    trial_filenames.append(filename)

for i, filename in enumerate(trial_filenames):
    sys.stdout.write("Reading trials [%d/%d]\r" % (i+1, len(trial_filenames)))
    sys.stdout.flush()

    with open(os.path.join(results_path, filename, "trial.pkl"), "rb") as f:
        trial = pickle.load(f)

    _, seed, _ = filename.split("_")
    case = (trial.config["map_name"], seed)

    worldstr = trial.config["world"]
    env = interpret(worldstr, obj_id_map=obj_id_map)

    for objid in env.state.object_states:
        if isinstance(env.state.object_states[objid], RobotState):
            robot_state = env.state.object_states[objid]
    robot_pose = robot_state["pose"]

    if case not in case_to_robotpose:
        case_to_robotpose[case] = {}

    if robot_pose not in case_to_robotpose[case]:
        case_to_robotpose[case][robot_pose] = []

    case_to_robotpose[case][robot_pose].append(filename)

sys.stdout.write("\n")
sys.stdout.flush()

majority = {}
for case in case_to_robotpose:
    if len(case_to_robotpose[case]) == 1:
        print("Trials have the same robot pose for {}".format(case))
        continue

    print("---------------- {} ---------------".format(case))
    for robot_pose in case_to_robotpose[case]:
        print("pose: {}    count: {}".format(robot_pose, len(case_to_robotpose[case][robot_pose])))
        pprint(case_to_robotpose[case][robot_pose])

    majority_robot_pose = max(case_to_robotpose[case],
                              key=lambda robot_pose: len(case_to_robotpose[case][robot_pose]))

    majority[case] = majority_robot_pose

print("=============== MAJORITIES ===============")
pprint(majority)
with open(os.path.join(robot_poses_path, "robot_poses.yaml"), "w") as f:
    yaml.dump(majority, f)
