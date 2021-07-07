# Basically returns the annotated frame of reference
from sloop.datasets.dataloader import *
from sloop.datasets.utils import *
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop.models.nn.loss_function import clamp_angle
from sloop.models.nn.metrics import mean_ci_normal
import math
from pprint import pprint

raise Value("TrueForefModel never used")

class TrueForefModel:

    def __init__(self, keyword, map_names, datadir,
                 map_dims=(41,41)):
        mapinfo = MapInfoDataset()
        for map_name in map_names:
            mapinfo.load_by_name(map_name.strip())
        fields = [(FdHint, tuple()),
                  (FdFoRefOrigin, (mapinfo,), {"desired_dims": map_dims}),
                  (FdFoRefAngle, tuple())]
        self.dataset = SpatialRelationDataset.build(keyword,
                                                    map_names,
                                                    datadir,
                                                    fields=fields)

    def predict_foref(self, keyword, landmark_symbol, map_name, mapinfo,
                      hint=None):
        if hint is None:
            return None
        if hint not in self.dataset.df[FdHint.NAME].values:
            print("Hint \"%s\" not in dataset" % hint)
        else:
            df = self.dataset.df
            row = df.loc[df[FdHint.NAME] == hint].iloc[0]
            foref_origin = row[FdFoRefOrigin.NAME].astype(float)
            foref_angle = row[FdFoRefAngle.NAME].astype(float)[0]
            return [*foref_origin, foref_angle]

    def has_hint(self, hint):
        return hint in self.dataset.df[FdHint.NAME].values

    def __call__(self, keyword, landmark_symbol, map_name, mapinfo,
                 hint=None):
        return self.predict_foref(keyword, landmark_symbol, map_name, mapinfo,
                                  hint=hint)


# Compute the difference between blind and non-blind frame of references
dataset_path = "../../datasets/SL-OSM-Dataset/"
sgdir = os.path.join(dataset_path, "sg_parsed")
def test():
    predicates = {"front", "left", "right", "behind"}
    map_names = {"cleveland", "austin", "denver", "honolulu", "washington_dc"}
    map_dims = (41,41)
    results = {}

    tb = {}
    tnotb = {}
    for predicate in predicates:
        print(predicate)
        tb[predicate] = TrueForefModel(predicate, map_names,
                                    os.path.join(dataset_path, "blind/FOR_only/sg_processed"),
                                    map_dims=map_dims)
        tnotb[predicate] = TrueForefModel(predicate, map_names,
                                          os.path.join(dataset_path, "not_blind/FOR_only/sg_processed"),
                                          map_dims=map_dims)

        # Get hints
        hints = tb[predicate].dataset.df[FdHint.NAME].values
        results[predicate] = {"angle_diff": [],
                              "origin_diff": []}
        for hint in hints:
            if tnotb[predicate].has_hint(hint):
                true_blind_foref = tb[predicate].predict_foref(predicate, None, None, None,
                                                               hint=hint)
                true_not_blind_foref = tnotb[predicate].predict_foref(predicate, None, None, None,
                                                                      hint=hint)
                origin_diff = euclidean_dist(true_blind_foref[:2],
                                             true_not_blind_foref[:2])
                angle_diff = math.degrees(clamp_angle(abs(true_blind_foref[2] - true_not_blind_foref[2])))
                results[predicate]["angle_diff"].append(angle_diff)
                results[predicate]["origin_diff"].append(origin_diff)

        mean_angle, ci_angle = mean_ci_normal(results[predicate]["angle_diff"])
        mean_origin, ci_origin = mean_ci_normal(results[predicate]["origin_diff"])
        results[predicate]["__summary__"] = {
            "angle_diff": mean_angle,
            "angle_diff_ci": ci_angle,
            "origin_diff": mean_origin,
            "origin_diff_ci": ci_origin,
        }

        pprint(results[predicate]["__summary__"])

if __name__ == "__main__":
    test()
