import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sloop.utils import json_safe
from sloop.models.nn.base_model import BaseModel
from sloop.models.nn.loss_function import FoRefLoss, clamp_angle
from sloop.models.nn.plotting import *
from sloop.models.nn.metrics import *
from sloop.models.heuristics.rules import ForefRule
from sloop.models.heuristics.model import evaluate as rule_based_evaluate
from sloop.models.heuristics.model import RuleBasedModel
import numpy as np
import json
from pprint import pprint

CONV1_PLAIN = 16
CONV1_KERNEL = 3
CONV2_PLAIN = 16
CONV2_KERNEL = 3
CONV3_PLAIN = 16
CONV3_KERNEL = 3
FOREF_L1 = 128
FOREF_L2 = 64

class MapImgCNN(nn.Module):
    def __init__(self, stride=1, map_dims=(21,21)):
        super(MapImgCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, CONV1_PLAIN,
                               kernel_size=CONV1_KERNEL,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(CONV1_PLAIN)
        self.max_pool1 = nn.MaxPool2d(CONV1_KERNEL, stride=stride, padding=1)

        self.conv2 = nn.Conv2d(CONV1_PLAIN, CONV2_PLAIN,
                               kernel_size=CONV2_KERNEL,
                               stride=stride, padding=1, bias=False)
        self.max_pool2 = nn.MaxPool2d(CONV2_KERNEL, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(CONV2_PLAIN)

        self.conv3 = nn.Conv2d(CONV2_PLAIN, CONV3_PLAIN,
                               kernel_size=CONV3_KERNEL,
                               stride=stride, padding=1, bias=False)
        self.max_pool3 = nn.MaxPool2d(CONV3_KERNEL, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(CONV3_PLAIN)

        self.input_size = map_dims[0]*map_dims[1]
        self.map_dims = map_dims

    def forward(self, x):
        x = x.view(-1, 1, self.map_dims[0], self.map_dims[1])
        out = self.max_pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.max_pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.max_pool3(F.relu(self.bn3(self.conv3(out))))
        out = out.view(out.shape[0], -1)
        return out

class LandmarkForefModel(BaseModel):
    NAME="landmark_foref"

    def __init__(self, keyword, learning_rate=1e-4, map_dims=(21,21)):
        super(LandmarkForefModel, self).__init__()
        self.keyword = keyword
        self.map_dims = map_dims
        self.ego_map_layer = MapImgCNN(map_dims=map_dims)
        self.layer_foref = nn.Sequential(nn.Linear(CONV3_PLAIN*map_dims[0]*map_dims[1], FOREF_L1),
                                         nn.ReLU(),
                                         nn.Linear(FOREF_L1, FOREF_L2),
                                         nn.ReLU(),
                                         nn.Linear(FOREF_L2, 3))
        self.input_size = map_dims[0]*map_dims[1]
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        self.normalizers = {}

    def forward(self, x):
        out = self.ego_map_layer(x)
        out = self.layer_foref(out)
        return out

    @classmethod
    def Input_Fields(cls):
        return [FdBdgImg.NAME]

    @classmethod
    def Output_Fields(cls):
        return [FdFoRefOrigin.NAME,
                FdFoRefAngle.NAME]

    @classmethod
    def get_data(cls, keyword, data_dirpath, map_names,
                 augment_radius=0,
                 augment_dfactor=None,
                 fill_neg=False,
                 rotate_amount=0,
                 add_pr_noise=0.15,
                 balance=True,
                 normalizers=None,
                 desired_dims=None,
                 antonym_as_neg=True, **kwargs):
        mapinfo = MapInfoDataset()
        for map_name in map_names:
            mapinfo.load_by_name(map_name.strip())
        data_ops = cls.compute_ops(mapinfo, keyword=keyword,
                                   augment_radius=augment_radius,
                                   augment_dfactor=augment_dfactor,
                                   fill_neg=fill_neg, rotate_amount=rotate_amount,
                                   add_pr_noise=add_pr_noise)
        fields = [(FdAbsObjLoc, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdObjLoc, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdBdgImg, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdBdgEgoImg, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdFoRefOrigin, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdFoRefAngle, tuple()),
                  (FdLmSym, tuple()),
                  (FdMapName, tuple()),
                  (FdProbSR, tuple())]
        dataset = SpatialRelationDataset.build(keyword, map_names, data_dirpath,
                                               fields=fields,
                                               data_ops=data_ops)
        if normalizers is not None:
            dataset.normalizers = normalizers
        _info_dataset = {keyword: {"fields": fields, "ops": data_ops}}
        return dataset, _info_dataset

    @classmethod
    def Train(cls, model, trainset, device,
              **kwargs):
        # First, train the frame of reference module
        print("Training the frame of reference module")
        criterion = FoRefLoss(trainset, reduction="sum", device=device)
        return super(LandmarkForefModel, cls).Train(model, trainset, device,
                                                      criterion=criterion,
                                                      **kwargs)

    @classmethod
    def make_input(cls, data_sample, dataset, mapinfo,
                   as_batch=True, **kwargs):
        return data_sample[FdBdgImg.NAME].float()

    def predict_foref(self, keyword, landmark_symbol,
                      map_name, mapinfo, device="cpu", map_img=None):
        if keyword != self.keyword:
            print("Given keyword %s != model's keyword %s" % (keyword, self.keyword))
            return None
        if len(self.normalizers) == 0:
            raise ValueError("Normalizers not present.")

        dummy_dataset = SpatialRelationDataset(None, None,
                                               normalizers=self.normalizers)

        # create context image
        if map_img is None:
            bdg_img = make_context_img(mapinfo, map_name, landmark_symbol, dist_factor=2.0).flatten()
        else:
            bdg_img = map_img
        bdg_img = dummy_dataset.normalize(FdBdgImg.NAME, bdg_img)
        prediction = self(torch.tensor(bdg_img).reshape(1,-1).float().to(device))[0]
        foref = read_foref(dummy_dataset, prediction)
        return foref

    @classmethod
    def Eval(cls, keyword, model, dataset, device, save_dirpath,
             suffix="group", **kwargs):
        # cls.Eval_OutputForef(keyword, model, dataset, device,
        #                      save_dirpath, suffix=suffix, relative=False, **kwargs)
        mapinfo = MapInfoDataset()
        for map_name in dataset.map_names:
            mapinfo.load_by_name(map_name.strip())
        map_dims = kwargs.get("map_dims", (21,21))
        test_samples = []

        amount = min(len(dataset), kwargs.get("test_amount", 100))
        indices = list(np.arange(len(dataset)))
        random.Random(100).shuffle(indices)
        for kk in range(amount):
            i = indices[kk]
            abs_obj_loc = dataset[i][FdAbsObjLoc.NAME]
            abs_obj_loc = dataset.rescale(FdAbsObjLoc.NAME, abs_obj_loc)
            map_name = dataset[i][FdMapName.NAME]
            landmark = dataset[i][FdLmSym.NAME]
            bdg_map = dataset.rescale(FdBdgImg.NAME, dataset[i][FdBdgImg.NAME])
            test_samples.append(("DummyObject", landmark, map_name, keyword, abs_obj_loc, bdg_map))
        rules = {keyword: ForefRule(keyword)}
        rbm_model = RuleBasedModel(rules)
        results = rule_based_evaluate(rbm_model, test_samples, mapinfo,
                                      map_dims=map_dims,
                                      foref_model_path=os.path.join(save_dirpath, "%s_model.pt" % keyword),
                                      device=device)

        metrics_dir = os.path.join(save_dirpath, "metrics", suffix)
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        with open(os.path.join(metrics_dir, "rule_based_infomation_metrics.json"), "w") as f:
            json.dump(json_safe(results), f, sort_keys=True, indent=4)
        print("Summary results (Rule based evaluation):")
        pprint(results["__summary__"])


    @classmethod
    def Plot(cls, keyword, model, dataset, device,
             save_dirpath, suffix="plot", **kwargs):
        cls.Plot_OutputForef(keyword, model, dataset, device,
                             save_dirpath, suffix=suffix, relative=False, **kwargs)
