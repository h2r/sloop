"""
Checks if the contents of langauges dataset (i.e. spatial graphs)
of given paths are the same.
"""
import os
import json

paths = [
    "./march30_2021-languages-keyword#auto",
    "./march30_2021-languages-rule#based#ego>ctx>foref>angle",
    "./march30_2021-languages-rule#based#ego>ctx>foref>angle#auto",
    "./march31_2021-languages-rule#based#ego>ctx>foref>angle"
]


def load_sgs(path):
    sgs = {}
    for map_name in os.listdir(path):
        map_dirpath = os.path.join(paths[0], map_name)
        sgs[map_name] = {}
        for sg_filename in os.listdir(map_dirpath):
            with open(os.path.join(map_dirpath, sg_filename)) as f:
                sg = json.load(f)
            sgs[map_name][sg_filename] = sg
    return sgs

sgs0 = load_sgs(paths[0])
for path in paths[1:]:
    print(path)
    sgs = load_sgs(path)
    assert sgs0 == sgs, "sgs from {} mismatch those in {}".format(paths[0], path)
print("Good.")
