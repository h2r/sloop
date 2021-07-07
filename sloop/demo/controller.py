"""This is the main program that controls the drone to search for a target
object in AirSim.
"""
import torch
import os, sys
os.environ["SPACY_WARNING_IGNORE"] = "W008"
os.environ['SDL_VIDEO_WINDOW_POS'] = str(0) + "," + str(0)

from sloop.oopomdp.experiments.interface import input_problem_config, load_files
from sloop.oopomdp.experiments.constants import obj_id_map
from sloop.oopomdp.experiments.experiment_foref_langprior import\
    create_world, make_trial, get_prior, obj_letter_map, obj_id_map
from sloop.oopomdp.env.env import MosEnvironment, interpret
from sloop.oopomdp.env.env import make_laser_sensor, equip_sensors
from sloop.oopomdp.env.visual import MosViz
from sloop.oopomdp.domain.action import *
from sloop.models.heuristics.rules import BASIC_RULES, ForefRule
from sloop.models.heuristics.model import KeywordModel, RuleBasedModel, MixtureSLUModel
from sloop.demo.problem import AirSimSearchEnvironment
from sloop.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop.demo.utils import quat_to_euler, to_radians
from sloop.demo.visual import AirSimSearchViz
from sloop.demo.maps.topological_graph import rrt_topo
from sloop.demo.runner import AirSimDemo
import time
from pprint import pprint

def make_model(prior_type, map_name, device, resources_path="../oopomdp/experiments/resources"):
    basic_rules = BASIC_RULES
    relative_rules = {}
    relative_predicates = {"front", "behind", "left", "right"}
    foref_models = {}
    model_name = "ego_ctx_foref_angle"
    iteration = 2
    for predicate in relative_predicates:
        if predicate in {"front", "behind"}:
            model_keyword = "front"
        else:
            model_keyword = "left"
        foref_model_path = os.path.join(
            resources_path, "models",
            "iter%s_%s:%s:%s"
            % (iteration, model_name.replace("_", "-"),
               model_keyword, map_name.replace("_", ",")),
            "%s_model.pt" % model_keyword)
        if os.path.exists(foref_model_path):
            print("Loading %s model for %s" % (model_name, predicate))
            nn_model = torch.load(foref_model_path, map_location=device)
            foref_models[predicate] = nn_model.predict_foref
            relative_rules[predicate] = ForefRule(predicate)
        else:
            import pdb; pdb.set_trace()
            raise ValueError("Pytorch model [%s] for %s does not exist!" % (model_name, predicate))
    rules = {**basic_rules,
             **relative_rules}
    print("All Rules (%s):" % (model_name))
    pprint(list(rules.keys()))

    if prior_type == "rule":
        model = RuleBasedModel(rules)
    else:
        model = MixtureSLUModel(rules)
    return model, foref_models


# Creates a AirSimSearch trial
def run_instance(**kwargs):
    # Configurations
    sensor_range = kwargs.get("sensor_range", 3)
    sensor = make_laser_sensor(90, (1, sensor_range), 0.5, False)
    problem_args = {"sigma": kwargs.get("sigma", 0.01),
                    "epsilon": kwargs.get("epsilon", 1.0),
                    "agent_has_map": kwargs.get("agent_has_map", True),
                    "reward_small": kwargs.get("small", 10),
                    "sensors": {"r": sensor},
                    "no_look": kwargs.get("no_look", True)}
    solver_args = {"max_depth": kwargs.get("max_depth", 30),
                   "discount_factor": kwargs.get("discount_factor", 0.95),
                   "planning_time": kwargs.get("planning_time", -1.),
                   "num_sims": kwargs.get("num_sims", 300),
                   "exploration_const": kwargs.get("exploration_const", 1000)}
    exec_args = {"max_time": kwargs.get("max_time", 360),
                 "max_steps": kwargs.get("max_steps", 200),
                 "visualize": kwargs.get("visualize", True)}
    config = {"problem_args": problem_args,
              "solver_args": solver_args,
              "exec_args": exec_args}
    config["map_name"] = kwargs.get("map_name", "neighborhoods")
    config["robot_pose"] = kwargs.get("robot_pose", (20, 20))
    config["target_poses"] = kwargs.get("target_poses", [(14, 19)])
    target_objects = ["R", "G", "B"]
    dims = (41,41)
    worldstr = create_world(dims[0], dims[1],
                            config["robot_pose"],
                            set({}),
                            config["target_poses"],
                            target_objects)
    config["world"] = worldstr
    # VIDEO 1
    # config["language"] = "The red car is by the side of Main street in front of House Z1 D,"\
    #                      "while the green car is within House Z7 D on the right side of North street."
    config["language"] = "The red car is behind House Z5 D around ZainRoberts Five House "
    config["obj_id_map"] = obj_id_map

    mapinfo = MapInfoDataset()
    mapinfo.load_by_name(config["map_name"])

    prior_type = "mixture"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, foref_models = make_model(prior_type, config["map_name"], device)

    print("Generating prior...")
    prior, metadata = get_prior(prior_type, model,
                                config["language"],
                                config["map_name"],
                                mapinfo,
                                foref_models=foref_models,
                                # 28x28 crop
                                foref_kwargs={"device": device,
                                              "mapsize": (28,28)})
    config["prior_type"] = prior_type
    config["prior"] = prior
    if config["map_name"] == "neighborhoods":
        config["airsim_config"] = {
            "init_point_pair" : [(20, 20), (160.0, -1500.0)],#[(35, 20), (12990.0, -1540.0)],
            "second_point_pair" : [(5, 5), (-12400.0, -14240.0)],#(20, 20), (160.0, -1500.0)],#[(5, 5), (-12510.0, -14450.0)],
            "flight_height": kwargs.get("height", 3.2),
            "landmark_heights": kwargs.get("landmark_heights", {})
        }

    print("Creating demo trial")
    trial = AirSimDemo("airsim-{}_100_{}".format(config["map_name"], prior_type),
                       config)
    trial.run()

run_instance(target_poses=[(13, 32)],
             robot_pose=(20,20))
# video 1
#[(27,21), (37,14)])
