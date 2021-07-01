# Basically reads the config.yaml and writes the language and sg_dict files as
# a separate result file

import argparse
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from spatial_foref.oopomdp.experiments.lang_result import LangResult

def main():
    parser = argparse.ArgumentParser(description="Remake trial script splits.")
    parser.add_argument("exp_path", type=str, help="Path to experiments")
    args = parser.parse_args()
    
    exp_path = args.exp_path
    if args.exp_path.endswith("/"):
        exp_path = os.path.dirname(args.exp_path)    

    for name in os.listdir(exp_path):
        if not os.path.isdir(os.path.join(exp_path, name)):
            continue

        # Read config file
        config_fpath = os.path.join(exp_path, name, "config.yaml")
        if not os.path.exists(config_fpath):
            continue
        with open(config_fpath) as f:
            config = yaml.load(f, Loader=yaml.Loader)

        language = config["language"]
        if "prior_metadata" in config and "sg_dict" in config["prior_metadata"]:
            sg_dict = config["prior_metadata"]["sg_dict"]
        else:
            sg_dict = {}

        langres = LangResult(language, sg_dict)
        save_path = os.path.join(exp_path, name, LangResult.FILENAME())
        langres.save(save_path)
        print("Saved language result to %s" % save_path)

if __name__ == "__main__":
    main()
