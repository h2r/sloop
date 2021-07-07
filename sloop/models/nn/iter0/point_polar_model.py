# Point model:
#
#  Relative point -> Probability
#
# a simple linear model

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sloop.datasets.dataloader import *
from sloop.models.nn.plotting import *
from sloop.models.nn.metrics import *
from sloop.models.nn.base_model import BaseModel
from sloop.datasets.utils import *

L1 = 48
L2 = 48

class PointPolarModel(BaseModel):
    NAME = "point_polar"
    def __init__(self, keyword, learning_rate=1e-4, **kwargs):
        super(PointPolarModel, self).__init__()
        self.keyword = keyword
        self.fcn_layer = nn.Sequential(nn.Linear(4, L1),
                                       nn.ReLU(),
                                       nn.Linear(L1, L2),
                                       nn.ReLU(),
                                       nn.Linear(L2, 1))
        self.input_size = 2
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

    def forward(self, x):
        out = self.fcn_layer(x)
        return out

    @classmethod
    def Input_Fields(cls):
        return [FdObjLoc.NAME,
                FdObjLocPolar.NAME]

    @classmethod
    def Output_Fields(cls):
        return [FdProbSR.NAME]

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
        data_ops = cls.compute_ops(mapinfo, augment_radius=augment_radius,
                                   augment_dfactor=augment_dfactor,
                                   fill_neg=fill_neg, rotate_amount=rotate_amount,
                                   add_pr_noise=add_pr_noise)
        fields = [(FdAbsObjLoc, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdObjLoc, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdObjLocPolar, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdLmSym, tuple()),
                  (FdMapName, tuple()),
                  (FdProbSR, tuple())]
        dataset = SpatialRelationDataset.build(keyword, map_names, data_dirpath,
                                               fields=fields,
                                               data_ops=data_ops)
        _info_dataset = {keyword: {"fields": fields, "ops": data_ops}}

        if antonym_as_neg:
            if keyword == "front":
                antonym = "behind"
            elif keyword == "behind":
                antonym = "front"
            elif keyword == "left":
                antonym = "right"
            elif keyword == "right":
                antonym = "left"

            # Apply the same operators on the negative dataset,
            # and additionally flip the probability, and balance the data.
            op_flip = (OpFlipProb, tuple())
            neg_data_ops = data_ops + [op_flip]
            if balance:
                op_balance = (OpBalance, (len(dataset.df),))
                neg_data_ops.append(op_balance)
            neg_dataset = SpatialRelationDataset.build(
                antonym, map_names, data_dirpath,
                fields=fields,
                data_ops=neg_data_ops)
            dataset = dataset.append(neg_dataset)
            # for frame of reference, don't do min/max normalization. Instead,
            # use the class's own normalization function.
            if normalizers is not None:
                dataset.normalizers = normalizers
            _info_dataset[antonym] = {"fields":fields, "ops":neg_data_ops}
            _info_dataset["_normalizers_"] = dataset.normalizers
        return dataset, _info_dataset


    @classmethod
    def make_input(cls, data_sample, dataset, mapinfo, as_batch=True, abs_obj_loc=None, **kwargs):
        if abs_obj_loc is None:
            inpt = torch.cat([data_sample[FdObjLoc.NAME].float(),
                              data_sample[FdObjLocPolar.NAME].float()])
        else:
            # Need to compute relative object location from the absolute one,
            # to use as input to the network.
            x, y = abs_obj_loc
            landmark_symbol = data_sample[FdLmSym.NAME]
            lmctr = mapinfo.center_of_mass(landmark_symbol, data_sample[FdMapName.NAME])
            rel_obj_loc = np.array([x-lmctr[0], y-lmctr[1]])
            rel_obj_loc_polar = to_polar(rel_obj_loc)
            rel_obj_loc = dataset.normalize(FdObjLoc.NAME, rel_obj_loc)
            rel_obj_loc_polar = dataset.normalize(FdObjLocPolar.NAME, rel_obj_loc)
            inpt = torch.cat([torch.tensor(rel_obj_loc).float(),
                              torch.tensor(rel_obj_loc_polar).float()])
        if as_batch:
            return inpt.reshape(1,-1)
        else:
            return inpt

    @classmethod
    def Eval(cls, keyword, model, dataset, device, save_dirpath,
             suffix="group", **kwargs):
        cls.Eval_OutputPr(keyword, model, dataset, device, save_dirpath,
                          suffix=suffix, **kwargs)


    @classmethod
    def Plot(cls, keyword, model, dataset, device, save_dirpath,
             suffix="group", **kwargs):
        cls.Plot_OutputPr(keyword, model, dataset, device, save_dirpath,
                          suffix=suffix, **kwargs)
