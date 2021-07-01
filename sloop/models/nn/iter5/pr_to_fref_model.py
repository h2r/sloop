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
from spatial_foref.models.nn.loss_function import FoRefLoss, clamp_angle
from spatial_foref.utils import json_safe
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pprint import pprint

##################### PrToFrefModel ############################
class PrToFrefModel(nn.Module):
    NAME = "pr_to_fref"
    def __init__(self, keyword, learning_rate=1e-4, map_dims=(21,21)):
        super(PrToFrefModel, self).__init__()
        self.keyword = keyword
        self.layer_half_abs = HalfModel(map_dims=map_dims)
        self.layer_half_ego = HalfModel(map_dims=map_dims)
        self.layer_combine = nn.Sequential(nn.Linear(HALF_FEAT_SIZE2*2 + 1, FULL_FEAT_SIZE),
                                           nn.ReLU(),                                            
                                           nn.Linear(FULL_FEAT_SIZE, FULL_FEAT_SIZE2),
                                           nn.ReLU(),
                                           nn.Linear(FULL_FEAT_SIZE2, 3))  # tentative
        self.input_size = self.layer_half_abs.input_size * 2
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

    def forward(self, x):
        x_pr = x[:, :1]
        x_abs = x[:, 1:self.layer_half_abs.input_size+1]
        x_ego = x[:, self.layer_half_abs.input_size+1:]
        out_abs = self.layer_half_abs(x_abs)
        out_ego = self.layer_half_ego(x_ego)
        out = torch.cat([x_pr, out_abs, out_ego], 1)
        out = self.layer_combine(out)
        return out

    @classmethod
    def Input_Fields(cls):
        return [FdProbSR.NAME,
                FdBdgImg.NAME,
                FdObjLoc.NAME,
                FdBdgEgoImg.NAME,
                FdAbsObjLoc.NAME]

    @classmethod
    def Output_Fields(cls):
        return [FdFoRefOrigin.NAME,
                FdFoRefAngle.NAME]

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

        if valset is None:
            print("Splitting training set by 1:%.2f to get validation set." % val_ratio)
            trainset, valset = trainset.split(val_ratio)
        assert trainset.normalizers == valset.normalizers, "Train / validation sets have different normalizers."
        print("Train set size: %d" % len(trainset))
        print("Validation set size: %d" % len(valset))        
        
        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=shuffle)

        criterion = FoRefLoss(trainset, reduction="sum", device=device)

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
    def Eval(cls, keyword, model, dataset, device,
             save_dirpath, suffix="metrics", **kwargs):
        print("Eval for class %s is done together through Plot." % cls.__name__)
            
    @classmethod
    def Plot(cls, keyword, model, dataset, device,
             save_dirpath, suffix="plot", **kwargs):

        batch_size = 1
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

        # Number of examples to generate plots for. The
        # rest would only be used to evaluate the metrics.
        plot_amount = kwargs.get("plot_amount", 10)
        map_dims = kwargs.get("map_dims", (21,21))

        plots_dir = os.path.join(save_dirpath, "plots", suffix)
        if not os.path.exists(os.path.join(plots_dir)):
            os.makedirs(os.path.join(plots_dir))
        metrics_dir = os.path.join(save_dirpath, "metrics", suffix)
        if not os.path.exists(os.path.join(metrics_dir)):
            os.makedirs(os.path.join(metrics_dir))            

        # Saving the metrics
        results = {"pos_neg_angle_diff": [],
                   "true_pred_angle_diff": [],
                   "pos_neg_origin_diff": [],
                   "true_pred_origin_diff": []}
            
        for i, batch in enumerate(data_loader):
            # Get prediction
            prob = dataset.normalize(FdProbSR.NAME, 0.99)
            train_inputs = torch.cat([
                torch.tensor([[prob]]).float(),
                batch[FdBdgImg.NAME].float(),
                batch[FdObjLoc.NAME].float(),
                batch[FdBdgEgoImg.NAME].float(),
                batch[FdAbsObjLoc.NAME].float(),
            ], 1).float()
            prediction_pos = model(train_inputs.to(device))[0]
            foref_pos = read_foref(dataset, prediction_pos)

            prob = dataset.normalize(FdProbSR.NAME, 0.01)
            train_inputs = torch.cat([
                torch.tensor([[prob]]).float(),
                batch[FdBdgImg.NAME].float(),
                batch[FdObjLoc.NAME].float(),
                batch[FdBdgEgoImg.NAME].float(),
                batch[FdAbsObjLoc.NAME].float(),
            ], 1).float()
            prediction_neg = model(train_inputs.to(device))[0]
            foref_neg = read_foref(dataset, prediction_neg)

            train_inputs = torch.cat([
                batch[FdProbSR.NAME].float(),
                batch[FdBdgImg.NAME].float(),
                batch[FdObjLoc.NAME].float(),
                batch[FdBdgEgoImg.NAME].float(),
                batch[FdAbsObjLoc.NAME].float(),
            ], 1).float()
            prediction_data = model(train_inputs.to(device))[0]
            foref_data = read_foref(dataset, prediction_data)

            foref_true = read_foref(dataset, torch.cat([batch[FdFoRefOrigin.NAME],
                                                        batch[FdFoRefAngle.NAME]], 1)[0])
            
            # Record results
            results["pos_neg_angle_diff"].append(
                math.degrees(clamp_angle(abs(foref_pos[2] - foref_neg[2]))))
            results["pos_neg_origin_diff"].append(
                euclidean_dist(foref_pos[:2], foref_neg[:2]))
                
            results["true_pred_angle_diff"].append(
                math.degrees(clamp_angle(abs(foref_true[2] - foref_data[2]))))
            results["true_pred_origin_diff"].append(
                euclidean_dist(foref_true[:2], foref_data[:2]))

            # Create example plots
            if i < plot_amount:
                mapimg = batch[FdBdgImg.NAME][0].numpy().reshape(map_dims)
                objloc = dataset.rescale(FdAbsObjLoc.NAME, batch[FdAbsObjLoc.NAME][0])

                fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 5))
                colors = {"positive": ("crimson", "green"),
                          "negative": ("magenta", "lime"),
                          "prediction": ("coral", "yellowgreen"),
                          "true": ("deepskyblue", "darkkhaki")}

                plot_multiple(mapimg,
                              dataset,
                              {"positive": foref_pos,
                               "negative": foref_neg},
                              objloc,
                              colors, axes[0],
                              nmap_dims=map_dims)

                plot_multiple(mapimg,
                              dataset,
                              {"prediction": foref_data,
                               "true": foref_true},
                              objloc,
                              colors, axes[1],
                              map_dims=map_dimsa)

                prob = float(batch[FdProbSR.NAME][0])
                prob = dataset.rescale(FdProbSR.NAME, prob)
                plt.title("%s prob: %.3f" % (keyword, prob))
                plt.savefig(os.path.join(plots_dir, "%s-%s-%d.png" % (keyword, suffix, i+1)))
                plt.clf()
            sys.stdout.write("Computing metrics and plotting ...[%d/%d]\r" % (i+1, len(data_loader)))

        # Plot 1d plots -- These are summary plots
        plot_1d(results["pos_neg_angle_diff"], "Angle differences between positive and negative FoRs")
        plt.savefig(os.path.join(plots_dir, "%s-%s-pos_neg_angle_diff.png" % (keyword, suffix)))
        plt.clf()
        
        plot_1d(results["pos_neg_origin_diff"], "Distances between positive and negative FoRs")
        plt.savefig(os.path.join(plots_dir, "%s-%s-pos_neg_origin_diff.png" % (keyword, suffix)))
        plt.clf()        
        
        plot_1d(results["true_pred_angle_diff"], "Angle differences between true and prdicted FoRs")
        plt.savefig(os.path.join(plots_dir, "%s-%s-true_pred_angle_diff.png" % (keyword, suffix)))
        plt.clf()        
        
        plot_1d(results["true_pred_origin_diff"], "Distances between true and predicted FoRs")
        plt.savefig(os.path.join(plots_dir, "%s-%s-true_pred_origin_diff.png" % (keyword, suffix)))
        plt.clf()

        titles = {
            "pos_neg_angle_diff": "Angle differences between positive and negative FoRs",
            "pos_neg_origin_diff": "Distances between positive and negative FoRs",
            "true_pred_angle_diff": "Angle differences between true and predicted FoRs",
            "true_pred_origin_diff": "Distances between true and predicted FoRs",            
        }

        # Save metrics
        results["__summary__"] = {}
        for catg in results:
            if catg.startswith("__"):
                continue
            plot_1d(results[catg], titles[catg])
            plt.savefig(os.path.join(plots_dir, "%s-%s-%s.png" % (keyword, suffix, catg)))
            plt.clf()
            
            mean, ci = mean_ci_normal(results[catg], confidence_interval=0.95)
            results["__summary__"][catg] = {
                "mean": mean,
                "ci-95": ci                
            }
        with open(os.path.join(metrics_dir, "foref_deviation.json"), "w") as f:
            json.dump(json_safe(results), f, indent=4, sort_keys=True)

        print("Summary results:")
        pprint(results["__summary__"])
        plt.close()
    
    
