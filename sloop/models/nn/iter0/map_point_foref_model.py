# A model that looks like:
#
# Map --> CNN --> FOR
#          |         \
#          -----------> FCN --> Pr
#                    /
#       point -------
#

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sloop.datasets.dataloader import *
from sloop.models.nn.base_model import BaseModel
from sloop.models.nn.loss_function import FoRefLoss, clamp_angle
from sloop.models.nn.plotting import *
from sloop.models.nn.metrics import *
import json
from pprint import pprint

CONV1_PLAIN = 6
CONV2_PLAIN = 16
CONV1_KERNEL = 3
CONV2_KERNEL = 3
MAP_FEATURE_SIZE = 84
FOREF_L1 = 64
FINAL_FCN_L1 = 128
FINAL_FCN_L2 = 64

class MapPointForefModel(BaseModel):
    """Given map, a point, produce a probability of the predicate.
    An intermediate step of this network is to produce a frame of reference."""
    NAME="map_point_foref"
    def __init__(self, keyword, learning_rate=1e-4, map_dims=(21,21)):
        super(MapPointForefModel, self).__init__()
        self.keyword = keyword
        self.map_dims = map_dims

        self.conv1 = nn.Conv2d(1, CONV1_PLAIN,
                               kernel_size=CONV1_KERNEL)
        self.pool = nn.MaxPool2d(CONV1_KERNEL)
        self.conv2 = nn.Conv2d(CONV1_PLAIN, CONV2_PLAIN,
                               kernel_size=CONV2_KERNEL)
        self.fc1 = nn.Linear(CONV2_PLAIN*(CONV2_KERNEL**2), 120)  # TODO: This linear input doesn't work for map (21,21)
        self.fc2 = nn.Linear(120, MAP_FEATURE_SIZE)
        self.fc3 = nn.Linear(MAP_FEATURE_SIZE, FOREF_L1)
        self.fcn_foref = nn.Linear(FOREF_L1, 3)

        self.fc4 = nn.Linear(MAP_FEATURE_SIZE + 3 + 2, FINAL_FCN_L1)
        self.fc5 = nn.Linear(FINAL_FCN_L1, FINAL_FCN_L2)
        self.fcn_pr = nn.Linear(FINAL_FCN_L2, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

    def forward(self, x, map_only=False, return_foref=False):
        """
        If map_only is False, then `x` contains both the map and the point.
        Therefore we are evaluating the network for the final probability output.
        Otherwise, x is just the image, and we're evaluating the network
        for the frame of reference.

        If map_only is False, and return_foref is True, then we will
        return both the probability and the frame of reference.
        """
        if map_only:
            x_map = x
        else:
            point = x[:, -2:]
            x_map = x[:, :-2]

        x_map = x_map.view(-1, 1, self.map_dims[0], self.map_dims[1])
        x_map = self.pool(F.relu(self.conv1(x_map)))
        x_map = self.pool(F.relu(self.conv2(x_map)))
        x_map = x_map.view(x_map.shape[0], -1)
        x_map = F.relu(self.fc1(x_map))
        x_map = F.relu(self.fc2(x_map))
        map_feature = x_map
        foref = F.relu(self.fc3(map_feature))
        foref = self.fcn_foref(foref)

        if map_only:
            return foref
        else:
            out = torch.cat([map_feature, foref, point], 1)
            out = F.relu(self.fc4(out))
            out = F.relu(self.fc5(out))
            pr = self.fcn_pr(out)
            if return_foref:
                return pr, foref
            else:
                return pr

    @classmethod
    def Input_Fields(cls):
        return [FdMapImg.NAME]

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
                  (FdMapImg, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdFoRefOrigin, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdFoRefAngle, tuple()),
                  (FdLmSym, tuple()),
                  (FdMapName, tuple()),
                  (FdProbSR, tuple())]
        dataset = SpatialRelationDataset.build(keyword, map_names, data_dirpath,
                                               fields=fields,
                                               data_ops=data_ops)
        _info_dataset = {keyword: {"fields": fields, "ops": data_ops}}
        return dataset, _info_dataset

    @classmethod
    def make_input(cls, data_sample, dataset, mapinfo,
                   as_batch=True, **kwargs):
        return data_sample[FdMapImg.NAME].float()

    @classmethod
    def Train(cls, model, trainset, device,
              **kwargs):
        # First, train the frame of reference module
        print("Training the frame of reference module")
        criterion = FoRefLoss(trainset, reduction="sum", device=device)
        input_fields = [FdMapImg.NAME]
        output_fields = [FdFoRefOrigin.NAME,
                         FdFoRefAngle.NAME]
        all_train_losses, all_val_losses = {}, {}
        train_losses, val_losses, _, _ =\
            super(MapPointForefModel, cls).Train(model, trainset, device,
                                                 criterion=criterion,
                                                 input_fields=input_fields,
                                                 output_fields=output_fields,
                                                 model_args={"map_only": True},
                                                 **kwargs)
        all_train_losses["map_to_fref"] = train_losses
        all_val_losses["map_to_fref"] = val_losses

        # Next, train the probability prediction module
        print("Training the probability prediction")
        criterion = nn.MSELoss(reduction="sum")
        input_fields = [FdMapImg.NAME,
                        FdAbsObjLoc.NAME]
        output_fields = [FdProbSR.NAME]
        train_losses, val_losses, trainset, valset =\
            super(MapPointForefModel, cls).Train(model, trainset, device,
                                                 criterion=criterion,
                                                 input_fields=input_fields,
                                                 output_fields=output_fields,
                                                 model_args={"map_only": False},
                                                 **kwargs)
        all_train_losses["map_to_pr"] = train_losses
        all_val_losses["map_to_pr"] = val_losses
        return all_train_losses, all_val_losses, trainset, valset


    @classmethod
    def make_input(cls, data_sample, dataset, mapinfo,
                   as_batch=True, abs_obj_loc=None, **kwargs):
        if abs_obj_loc is None:
            inpt = torch.cat([data_sample[FdMapImg.NAME].float(),
                              data_sample[FdAbsObjLoc.NAME].float()])
        else:
            # Need to compute relative object location from the absolute one,
            # to use as input to the network.
            x, y = abs_obj_loc
            abs_obj_loc = dataset.normalize(FdAbsObjLoc.NAME,
                                            np.array([x,y]))
            inpt = torch.cat([data_sample[FdMapImg.NAME].float(),
                              torch.tensor(abs_obj_loc).float()])
        if as_batch:
            return inpt.reshape(1,-1)
        else:
            return inpt


    @classmethod
    def Eval(cls, keyword, model, dataset, device, save_dirpath,
             suffix="group", **kwargs):
        """Evaluation of models that output spatial relation probability"""
        mapinfo = MapInfoDataset()
        for map_name in dataset.map_names:
            mapinfo.load_by_name(map_name.strip())

        metrics_dir = os.path.join(save_dirpath, "metrics", suffix)
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        results = {"perplex_true": [],  # The perplexity of a distribution for the true object location
                   "perplex_pred": [],  # The perplexity of the predicted heatmap
                   "kl_div": [],     # The kl divergence between true and predicted distributions
                   "distance": []}  # The distance between most likely object location and true object location

        map_dims = kwargs.get("map_dims", (21,21))
        all_locations = [(x,y)
                         for x in range(map_dims[0])
                         for y in range(map_dims[1])]
        variance = [[1, 0], [0,1]]
        for i in range(len(dataset)):
            data_sample = dataset[i]  # The returned data_sample should be normalized
            objloc = dataset.rescale(FdAbsObjLoc.NAME,
                                     data_sample[FdAbsObjLoc.NAME])
            true_dist = normal_pdf_2d(objloc, variance, all_locations)
            heatmap = cls.compute_heatmap(model, data_sample,
                                          dataset, mapinfo, device=device,
                                          map_dims=map_dims)
            pred_dist = {(x,y): heatmap[x,y]
                         for x,y in all_locations}
            # Convert the dictionary distributions into sequences, with matching
            # elements at each index.
            seqs, vals = dists_to_seqs([true_dist, pred_dist], avoid_zero=True)
            # Compute metrics and record
            perplex_true = perplexity(seqs[0])
            perplex_pred = perplexity(seqs[1])
            kl_div = kl_divergence(seqs[0], q=seqs[1])
            results["perplex_true"].append(perplex_true)
            results["perplex_pred"].append(perplex_pred)
            results["kl_div"].append(kl_div)

            objloc_pred = max(pred_dist, key=lambda x: pred_dist[x])
            dist = euclidean_dist(objloc_pred, objloc)
            results["distance"].append(dist)

            sys.stdout.write("Computing heatmaps & metrics...[%d/%d]\r" % (i+1, len(dataset)))

        results = compute_mean_ci(results)
        with open(os.path.join(metrics_dir, "information_metrics.json"), "w") as f:
            json.dump(json_safe(results), f, indent=4, sort_keys=True)

        print("Summary results:")
        pprint(results["__summary__"])


    @classmethod
    def Plot(cls, keyword, model, dataset, device, save_dirpath,
             suffix="plot", **kwargs):
        """Plot some examples for the model whose output is spatial relation probability"""
        mapinfo = MapInfoDataset()
        for map_name in dataset.map_names:
            mapinfo.load_by_name(map_name.strip())
        amount = kwargs.get("plot_amount", 10)
        plotsdir = os.path.join(save_dirpath, "plots", suffix)
        if not os.path.exists(plotsdir):
            os.makedirs(plotsdir)

        indices = list(np.arange(len(dataset)))
        random.Random(100).shuffle(indices)
        all_inputs = []
        for kk in range(amount):
            i = indices[kk]
            data_sample = dataset[i]
            inpt = cls.make_input(data_sample, dataset, mapinfo,
                                  as_batch=True)
            all_inputs.append(inpt)
        all_inputs = torch.cat(all_inputs, 0)
        prediction_pr, prediction_foref = model(all_inputs.to(device), return_foref=True)

        # Make plots
        map_dims = kwargs.get("map_dims", None)
        for i in range(len(prediction_pr)):
            data_sample = dataset[i]
            pred_prob = dataset.rescale(FdProbSR.NAME, prediction_pr[i].item())
            true_prob = dataset.rescale(FdProbSR.NAME, data_sample[FdProbSR.NAME])
            heatmap = cls.compute_heatmap(model, data_sample,
                                          dataset, mapinfo, device=device,
                                          map_dims=map_dims)

            # Plot map
            ax = plt.gca()
            if FdBdgImg.NAME in data_sample:
                map_img_normalized = data_sample[FdBdgImg.NAME].reshape(heatmap.shape)
                mapimg = dataset.rescale(FdBdgImg.NAME, map_img_normalized)
                plot_map(ax, mapimg.numpy().transpose())
            elif FdMapImg.NAME in data_sample:
                map_img_normalized = data_sample[FdMapImg.NAME].reshape(heatmap.shape)
                mapimg = dataset.rescale(FdMapImg.NAME, map_img_normalized)
                plot_map(ax, mapimg.numpy().transpose())

            # Plot heatmap
            plot_map(ax, heatmap, alpha=0.6)

            # Plot object location
            objloc = dataset.rescale(FdAbsObjLoc.NAME, data_sample[FdAbsObjLoc.NAME])
            ax.scatter([objloc[0].item()], [objloc[1].item()], s=100, c="cyan")

            # Plot frame of reference
            if FdFoRefOrigin.NAME in data_sample:
                foref_true = read_foref(dataset,
                                        [*data_sample[FdFoRefOrigin.NAME],
                                         data_sample[FdFoRefAngle.NAME]])
                plot_foref(foref_true, ax, c1="magenta", c2="lime")

            # Plot predicted frame of reference
            foref_pred = read_foref(dataset,
                                    prediction_foref[i])
            plot_foref(foref_pred, ax, c1="coral", c2="green")

            plt.title("%s : %.3f" % (keyword, pred_prob))
            plt.savefig(os.path.join(plotsdir, "%s-%s-%d.png" % (keyword, suffix, i+1)))
            plt.clf()
            sys.stdout.write("Plotting ...[%d/%d]\r" % (i+1, len(prediction_pr)))
