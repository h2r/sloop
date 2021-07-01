import argparse
import os
import pickle
import yaml
import re
import time
import pomdp_py
from spatial_foref.oopomdp.domain.action import NAME_TO_ACTION
import spatial_foref.oopomdp.problem as mos
from spatial_foref.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from spatial_foref.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
import spatial_foref.utils as util
import subprocess


def read_action(logline):
    action_txt = re.search("action:\s+[a-zA-Z-]+", logline).group()
    action_name = action_txt.split("action:")[1].strip()
    return NAME_TO_ACTION[action_name]

def load_problem(trial_path):
    with open(os.path.join(trial_path, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Recreate the environment
    prior = config["prior"]
    worldstr = config["world"]
    obj_id_map = config["obj_id_map"]
    problem_args = config["problem_args"]

    problem = mos.MosOOPOMDP("r",
                             prior=prior,
                             grid_map=worldstr,
                             obj_id_map=obj_id_map,
                             **problem_args)
    return problem, config

def main():
    parser = argparse.ArgumentParser(description="replay a trial")
    parser.add_argument("trial_path", type=str, help="Path to trial directory")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.trial_path, "states.pkl"), "rb") as f:
        states = pickle.load(f)

    with open(os.path.join(args.trial_path, "log.txt")) as f:
        log = f.readlines()

    problem, config = load_problem(args.trial_path)
    robot_id = problem.agent.robot_id

    # Dummy planner (never actually used)
    dummy_planner = pomdp_py.POUCT()

    # Create visualization
    map_name = config["map_name"]
    bg_path = FILEPATHS[map_name]["map_png"]
    viz = mos.MosViz(problem.env, controllable=False, bg_path=bg_path, res=25)
    if viz.on_init() == False:
        raise Exception("Environment failed to initialize")
    viz.update(robot_id,
               None,
               None,
               None,
               problem.agent.cur_belief)
    img = viz.on_render()
    viz.record_game_state(img)

    # Iterate over states. Do belief update
    for i in range(len(states)-1):
        action = read_action(log[i])
        # if problem.env.state != states[i]:
        #     if env.state
        #     "States at step {}"\
        #     " not equal\n env.state:{}\n saved state:{}".format(i, problem.env.state, states[i])
        reward = problem.env.state_transition(action, execute=True,
                                              robot_id=robot_id)
        observation = problem.env.provide_observation(problem.agent.observation_model, action)

        # Updates
        problem.agent.clear_history()  # truncate history
        problem.agent.update_history(action, observation)
        mos.belief_update(problem.agent, action, observation,
                          problem.env.state.object_states[robot_id],
                          dummy_planner)

        # Visualize
        robot_pose = problem.env.state.object_states[robot_id].pose
        viz_observation = mos.MosOOObservation({})
        if isinstance(action, mos.LookAction) or isinstance(action, mos.FindAction):
            viz_observation = \
                problem.env.sensors[robot_id].observe(robot_pose,
                                                      problem.env.state)
        viz.update(robot_id,
                   action,
                   observation,
                   viz_observation,
                   problem.agent.cur_belief)
        viz.on_loop()
        img = viz.on_render()
        viz.record_game_state(img)

        print(log[i])
        time.sleep(0.05)

    if args.save:
        print("Saving images...")
        util.save_images_and_compress(viz.img_history,
                                      os.path.join(args.trial_path))
        subprocess.Popen(["nautilus", args.trial_path])

    # Print info about this replay (language parsing)
    language = config["language"]
    if "sg_dict" in config["prior_metadata"]:
        sg_dict = config["prior_metadata"]["sg_dict"]
    else:
        sg_dict = {}
    print("Language: \"%s\"" % language)
    print("sg_dict: \"%s\"" % sg_dict)
    print("MAP: %s" % map_name)

if __name__ == "__main__":
    main()
