import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from spatial_foref.datasets.dataloader import *
from spatial_foref.models.nn.plotting import *
from spatial_foref.models.nn.metrics import *
from spatial_foref.models.nn.iter5.common import *
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
import random


# ##################### PrToFrefModel ############################
class FrefToPrModel(nn.Module):
    NAME = "fref_to_pr"
    def __init__(self, keyword, learning_rate=1e-4, map_dims=(21,21)):
        super(FrefToPrModel, self).__init__()
        self.keyword = keyword
        self.layer_half_abs = HalfModel(map_dims=map_dims)
        self.layer_half_ego = HalfModel(map_dims=map_dims)
        self.layer_combine = nn.Sequential(nn.Linear(HALF_FEAT_SIZE2*2 + 3, FULL_FEAT_SIZE),
                                           nn.ReLU(),                                            
                                           nn.Linear(FULL_FEAT_SIZE, FULL_FEAT_SIZE2),
                                           nn.ReLU(),
                                           nn.Linear(FULL_FEAT_SIZE2, 1))  # tentative
        self.input_size = self.layer_half_abs.input_size * 2
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

    def forward(self, x):
        x_foref = x[:, :3]
        x_abs = x[:, 3:self.layer_half_abs.input_size+3]
        x_ego = x[:, self.layer_half_abs.input_size+3:]
        out_abs = self.layer_half_abs(x_abs)
        out_ego = self.layer_half_ego(x_ego)
        out = torch.cat([x_foref, out_abs, out_ego], 1)
        out = self.layer_combine(out)
        return out

    @classmethod
    def Input_Fields(cls):
        return [FdFoRefOrigin.NAME,
                FdFoRefAngle.NAME,
                FdBdgImg.NAME,
                FdObjLoc.NAME,
                FdBdgEgoImg.NAME,
                FdAbsObjLoc.NAME]

    @classmethod
    def Output_Fields(cls):
        return [FdProbSR.NAME]

    @classmethod
    def Train(cls, model, trainset, device,
              val_ratio=0.2, num_epochs=500, batch_size=10, shuffle=True,
              save_dirpath=None, loss_threshold=1e-4, early_stopping=False,
              valset=None):
        
        """
        model (nn.Module): The network
        trainset (Dataset)
        val_ratio (float) ratio of the trainset to use as validation set
        loss_threshold (float): The threshold of training loss change.
        early_stopping (bool): True if terminate when validation loss increases,
            by looking at the average over the previous 10 epochs.
        """
        assert type(trainset) == SpatialRelationDataset
        criterion = nn.MSELoss(reduction="sum")

        if valset is None:
            print("Splitting training set by 1:%.2f to get validation set." % val_ratio)
            trainset, valset = trainset.split(val_ratio)
        print("Train set size: %d" % len(trainset))
        print("Validation set size: %d" % len(valset))        
        
        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=shuffle)    

        train_losses = []
        val_losses = []
        print(device)
        for epoch in range(num_epochs):
            running_train_loss = 0.0

            batch_num = 0
            for batch in train_loader:  # iterate by batches
                # Must transform to float because that's what pytorch supports.
                input_data = []
                for fd_name in cls.Input_Fields():
                    input_data.append(batch[fd_name].float())
                train_inputs = torch.cat(input_data, 1).float()

                output_data = []
                for fd_name in cls.Output_Fields():
                    output_data.append(batch[fd_name].float())
                train_labels = torch.cat(output_data, 1).float()
                model.train()  # Training mode
                prediction = model(train_inputs.to(device))
                train_loss = criterion(prediction, train_labels.to(device))
                model.zero_grad()
                train_loss.backward()
                model.optimizer.step()

                running_train_loss += train_loss.item()
                batch_num += 1

            # Record losses per epoch; Both training and validation losses
            print('[%d] loss: %.5f' %
                  (epoch + 1, running_train_loss / batch_num))

            train_losses.append(running_train_loss / batch_num)
            # Is the loss converging?
            train_done = False
            window = 20
            if loss_threshold is not None and epoch % window == 0:
                if len(train_losses) >= window * 2:
                    t_now = np.mean(train_losses[-window:])
                    t_prev = np.mean(train_losses[-2*window:-window])                    
                    loss_diff = abs(t_now - t_prev)
                    if loss_diff < loss_threshold:
                        train_done = True
                        print("Training loss converged.")

            # Compute validation loss
            running_val_loss = 0.0
            batch_num = 0
            for batch in val_loader:
                input_data = []
                for fd_name in cls.Input_Fields():
                    input_data.append(batch[fd_name].float())
                val_inputs = torch.cat(input_data, 1).float()

                output_data = []
                for fd_name in cls.Output_Fields():
                    output_data.append(batch[fd_name].float())
                val_labels = torch.cat(output_data, 1).float()
                prediction = model(val_inputs.to(device))
                val_loss = criterion(prediction, val_labels.to(device))
                running_val_loss += val_loss.item()
                batch_num += 1
            val_losses.append(running_val_loss / batch_num)

            if early_stopping:
                window = 20
                if epoch % window == 0:
                    if len(val_losses) >= window*2:
                        v_now = np.mean(val_losses[-window:])
                        v_prev = np.mean(val_losses[-2*window:-window])
                        if v_now > v_prev:
                            # Validation loss increased. Stop
                            print("Validation loss incrased (window size = %d). Stop." % window)
                            train_done = True
            
            if train_done:
                break

        if save_dirpath is not None:
            if not os.path.exists(save_dirpath):
                os.makedirs(save_dirpath)
            torch.save(model, os.path.join(save_dirpath, model.keyword + "_model.pt"))
        return train_losses, val_losses, trainset, valset

    
    @classmethod
    def get_data(cls, keyword, data_dirpath, map_names,
                 augment_radius=0,
                 augment_dfactor=None,
                 fill_neg=False,
                 rotate_amount=0,
                 balance=True,
                 add_pr_noise=0.15,
                 normalizers=None,
                 desired_dims=None):
        """
        normalizers: If given, then the loaded dataset will use this set of normalizers.
            Otherwise, normalizers will be computed.
        """
        
        mapinfo = MapInfoDataset()
        for map_name in map_names:
            mapinfo.load_by_name(map_name.strip())

        _info_dataset = {keyword: {"fields":[], "ops":[]}}  # Use to record data collection information            
        data_ops = []            
        if augment_radius > 0:
            op = (OpAugPositive,
                  (mapinfo, augment_radius),
                  {"dfactor": augment_dfactor})
            data_ops.append(op)
        if fill_neg:
            op = (OpFillNeg,
                  (mapinfo, keyword))
            data_ops.append(op)

        op_add_noise = (OpProbAddNoise,
                         tuple(),
                         {"noise": add_pr_noise})
        data_ops.append(op_add_noise)

        if rotate_amount > 0:
            op = (OpRandomRotate, (mapinfo,), {"amount": rotate_amount})
            data_ops.append(op)

        fields = [(FdObjLoc, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdAbsObjLoc, (mapinfo,), {"desired_dims": desired_dims}),
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
        _info_dataset[keyword]["fields"] = fields
        _info_dataset[keyword]["ops"] = data_ops        

        # Make a negative dataset using the antonym
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
        op_balance = (OpBalance, (len(dataset.df),))
        neg_data_ops = data_ops + [op_flip, op_balance]
        neg_dataset = SpatialRelationDataset.build(
            antonym, map_names, data_dirpath,
            fields=fields,
            data_ops=neg_data_ops)
        dataset = dataset.append(neg_dataset)
        # for frame of reference, don't do min/max normalization. Instead,
        # use the class's own normalization function.
        if normalizers is not None:
            dataset.normalizers = normalizers
        _info_dataset[antonym] = {"fields":[], "ops":[]}
        _info_dataset[antonym]["fields"] = fields
        _info_dataset[antonym]["ops"] = neg_data_ops
        _info_dataset["_normalizers_"] = dataset.normalizers
        return dataset, _info_dataset


    @classmethod
    def compute_heatmap(cls, model, data_sample,
                        dataset, mapinfo, device="cpu", map_dims=None):
        """
        data_sample is a row in the dataset. It contains fields
        such as frame of reference, and the map image. We aim to
        compute a heatmap on top of these information.
        """
        landmark_symbol = data_sample[FdLmSym.NAME]
        map_name = mapinfo.map_name_of(landmark_symbol)
        if map_dims is None:
            map_dims = mapinfo.map_dims(map_name)
        heatmap = np.zeros(map_dims, dtype=np.float64)
        
        lmctr = mapinfo.center_of_mass(landmark_symbol, data_sample[FdMapName.NAME])
        
        all_inputs = []
        for idx in range(map_dims[0]*map_dims[1]):
            x = idx // map_dims[1]
            y = idx - x * map_dims[1]
            abs_obj_loc = dataset.normalize(FdAbsObjLoc.NAME,
                                            np.array([x,y]))
            rel_obj_loc = dataset.normalize(FdObjLoc.NAME,
                                            np.array([x-lmctr[0], y-lmctr[1]]))
            # Concatenate features for a data sample into a single row tensor.
            inpt = torch.cat([data_sample[FdFoRefOrigin.NAME].float(),
                              data_sample[FdFoRefAngle.NAME].float(),
                              torch.tensor(rel_obj_loc).float(),
                              data_sample[FdBdgEgoImg.NAME].float(),
                              torch.tensor(abs_obj_loc).float(),
                              data_sample[FdBdgImg.NAME].float()])
            all_inputs.append(inpt.reshape(1,-1))
        # We are concatenating the tensors by rows.            
        all_inputs = torch.cat(all_inputs, 0)
        prediction = model(all_inputs.to(device))

        total_prob = 0.0
        for idx in range(map_dims[0]*map_dims[1]):
            x = idx // map_dims[1]
            y = idx - x * map_dims[1]
            prob = prediction[idx].item()
            heatmap[x,y] = prob
            total_prob += prob
        heatmap[x,y] /= total_prob
        return heatmap


    @classmethod
    def Eval(cls, keyword, model, dataset, device, save_dirpath,
             suffix="group", **kwargs):
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
            data_sample = dataset[i]
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
            seqs, vals = dists_to_seqs([true_dist, pred_dist])
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
        """Plot some examples"""
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
            inpt = torch.cat([data_sample[FdFoRefOrigin.NAME].float(),
                              data_sample[FdFoRefAngle.NAME].float(),
                              data_sample[FdObjLoc.NAME].float(),
                              data_sample[FdBdgEgoImg.NAME].float(),
                              data_sample[FdAbsObjLoc.NAME].float(),
                              data_sample[FdBdgImg.NAME].float()])
            all_inputs.append(inpt.reshape(1,-1))
        all_inputs = torch.cat(all_inputs, 0)
        prediction = model(all_inputs.to(device))

        # Make plots
        map_dims = kwargs.get("map_dims", None)        
        for i in range(len(prediction)):
            data_sample = dataset[i]
            pred_prob = dataset.rescale(FdProbSR.NAME, prediction[i].item())
            true_prob = dataset.rescale(FdProbSR.NAME, data_sample[FdProbSR.NAME])
            heatmap = cls.compute_heatmap(model, data_sample,
                                          dataset, mapinfo, device=device,
                                          map_dims=map_dims)
            map_img_normalized = data_sample[FdBdgImg.NAME].reshape(heatmap.shape)
            mapimg = dataset.rescale(FdBdgImg.NAME, map_img_normalized)
            objloc = dataset.rescale(FdAbsObjLoc.NAME, data_sample[FdAbsObjLoc.NAME])
            foref_true = read_foref(dataset,
                                    [*data_sample[FdFoRefOrigin.NAME],
                                     data_sample[FdFoRefAngle.NAME]])
            # Plot heatmap
            ax = plt.gca()
            plot_map(ax, mapimg.numpy().transpose())
            plot_map(ax, heatmap, alpha=0.6)
            # Plot the object location
            ax.scatter([objloc[0].item()], [objloc[1].item()], s=100, c="cyan")
            # Plot the frame of ref (true)
            plot_foref(foref_true, ax, c1="magenta", c2="lime")
            plt.title("%s : %.3f" % (keyword, pred_prob))
            plt.savefig(os.path.join(plotsdir, "%s-%s-%d.png" % (keyword, suffix, i+1)))
            plt.clf()
            sys.stdout.write("Plotting ...[%d/%d]\r" % (i+1, len(prediction)))
