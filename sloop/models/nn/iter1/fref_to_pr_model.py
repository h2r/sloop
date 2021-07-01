# Given point, frame of reference, and map,
# output probability of spatial relation


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from spatial_foref.models.nn.base_model import BaseModel
from spatial_foref.models.nn.loss_function import FoRefLoss, clamp_angle
from spatial_foref.models.nn.plotting import *
from spatial_foref.models.nn.metrics import *
from spatial_foref.models.nn.iter1.common import *
import numpy as np
import json
from pprint import pprint


class ContextPrModel(BaseModel):
    NAME="context_pr"

    def __init__(self, keyword, learning_rate=1e-4, map_dims=(21,21)):
        super(ContextPrModel, self).__init__()
        self.keyword = keyword
        self.map_dims = map_dims
        self.map_layer = MapImgCNN(map_dims=map_dims)
        self.layer_map_compress =\
            nn.Sequential(nn.Linear(CONV3_PLAIN*map_dims[0]*map_dims[1], 32),
                          nn.ReLU(),
                          nn.Linear(32, 32),
                          nn.ReLU(),
                          nn.Linear(32, 1))
        self.layer_foref = nn.Sequential(nn.Linear(3, 32),
                                         nn.ReLU(),
                                         nn.Linear(32, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 18))
        self.layer_point = nn.Sequential(nn.Linear(2, 16),
                                         nn.ReLU(),
                                         nn.Linear(16, 32),
                                         nn.ReLU(),
                                         nn.Linear(32, 64))
        self.layer_final = nn.Sequential(nn.Linear(1 + 18 + 64, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, 64),
                                         nn.ReLU(),                                         
                                         nn.Linear(64, 20),
                                         nn.ReLU(),                                         
                                         nn.Linear(20, 1))
        self.input_size = map_dims[0]*map_dims[1]
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)


    def forward(self, x):
        x_map = x[:, :-5]
        x_point = x[:, -5:-3]
        x_foref = x[:, -3:]
        
        x_map = self.map_layer(x_map)
        x_map = self.layer_map_compress(x_map)

        x_foref = self.layer_foref(x_foref)

        x_point = self.layer_point(x_point)

        out = torch.cat([x_map, x_point, x_foref], 1)
        out = self.layer_final(out)
        return out

    @classmethod
    def Input_Fields(cls):
        return [FdCtxImg.NAME,
                FdAbsObjLoc.NAME,
                FdFoRefOrigin.NAME,
                FdFoRefAngle.NAME]

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
        data_ops = cls.compute_ops(mapinfo, keyword=keyword,
                                   augment_radius=augment_radius,
                                   augment_dfactor=augment_dfactor,
                                   fill_neg=fill_neg, rotate_amount=rotate_amount,
                                   add_pr_noise=add_pr_noise)
        fields = [(FdAbsObjLoc, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdObjLoc, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdCtxImg, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdCtxEgoImg, (mapinfo,), {"desired_dims": desired_dims}),                                    
                  (FdFoRefOrigin, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdFoRefAngle, tuple()),                  
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
    def Train(cls, model, trainset, device,
              **kwargs):
        # First, train the frame of reference module
        print("Training the frame of reference module")
        criterion = nn.MSELoss(reduction="sum")
        return super(ContextPrModel, cls).Train(model, trainset, device,
                                                criterion=criterion,
                                                **kwargs)    

    @classmethod
    def make_input(cls, data_sample, dataset, mapinfo,
                   as_batch=True, abs_obj_loc=None, **kwargs):
        if abs_obj_loc is None:
            inpt = torch.cat([data_sample[FdCtxImg.NAME].float(),
                              data_sample[FdAbsObjLoc.NAME].float(),
                              data_sample[FdFoRefOrigin.NAME].float(),
                              data_sample[FdFoRefAngle.NAME].float()])
        else:
            # Need to compute relative object location from the absolute one,
            # to use as input to the network.
            x, y = abs_obj_loc
            abs_obj_loc = dataset.normalize(FdAbsObjLoc.NAME,
                                            np.array([x,y]))
            inpt = torch.cat([data_sample[FdCtxImg.NAME].float(),
                              torch.tensor(abs_obj_loc).float(),
                              data_sample[FdFoRefOrigin.NAME].float(),
                              data_sample[FdFoRefAngle.NAME].float()])
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
