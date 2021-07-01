# Make box plot
import numpy as np
import os
import json
from pprint import pprint
import matplotlib.pyplot as plt

def read_all_foref_deviation(filepath):
    with open(filepath) as f:
        r = json.load(f)
        all_diffs = r["true_pred_angle_diff"]
    return list(map(float, all_diffs))

def read_mean_foref_deviation(filepath):
    with open(filepath) as f:
        r = json.load(f)
        return float(r["__summary__"]["true_pred_angle_diff"]["mean"])

name_to_abrv = {
    "ego-ctx-foref-angle": "EGO-CTX",
    "ego-bdg-foref-angle": "EGO",
    "ctx-foref-angle": "CTX",
    "random": "Random",
    "annotator": "Human"
}
name_order = [
    "random", "ctx-foref-angle", "ego-bdg-foref-angle", "ego-ctx-foref-angle", "annotator"
]

def joint_boxplot(predicate, data_kind, result_file):
    # Gather all results, by approach.
    results = {}
    for root, dirs, files in os.walk("./%s" % predicate):
        # root: top-level directory (recursive)
        # dirs: direct subdirectories of root
        # files: files directly under root
        rootdir = os.path.basename(root)
        if not (rootdir.startswith("iter")\
                and len(rootdir.split("_")) == 3):
            continue
        print("Collecting %s" % rootdir)

        itername = rootdir.split("_")[0]
        timestamp = rootdir.split("_")[-1]
        baseline_name = "_".join(rootdir.split("_")[1:-1])
        model_name = baseline_name.split(":")[0]

        result_path = os.path.join(root,
                                   "metrics", data_kind, result_file)
        all_diffs = read_all_foref_deviation(result_path)
        if model_name not in results:
            results[model_name] = []
        results[model_name].extend(all_diffs)

    pprint(results)
    fig, ax = plt.subplots()
    data = []
    tick_labels = []
    locs = np.arange(1,len(name_order)+1)
    for model_name in name_order:
        print(model_name, len(results[model_name]))
        data.append(results[model_name])
        tick_labels.append(name_to_abrv[model_name])
    ax.boxplot(data)
    ax.set_xticks(locs)
    ax.set_xticklabels(tick_labels)
    plt.show()

def map_mean_boxplot(predicate, data_kind, result_file, ax):
    # First, gather the mean of every map. Then, do box plot for them.
    # (This seems to be a more reasonable way. We already have the mean
    #  for each map, and we just want to know how good the approach is
    #  across these maps).
    results = {}
    for root, dirs, files in os.walk("./%s" % predicate):
        # root: top-level directory (recursive)
        # dirs: direct subdirectories of root
        # files: files directly under root
        rootdir = os.path.basename(root)
        if not (rootdir.startswith("iter")\
                and len(rootdir.split("_")) == 3):
            continue
        print("Collecting %s" % rootdir)

        itername = rootdir.split("_")[0]
        timestamp = rootdir.split("_")[-1]
        baseline_name = "_".join(rootdir.split("_")[1:-1])
        model_name = baseline_name.split(":")[0]
        map_name = baseline_name.split(":")[2]

        result_path = os.path.join(root,
                                   "metrics", data_kind, result_file)
        mean_val = read_mean_foref_deviation(result_path)
        if model_name not in results:
            results[model_name] = []
        results[model_name].append(mean_val)

    # Make box plot
    pprint(results)
    data = []
    tick_labels = []
    locs = np.arange(1,len(name_order)+1)
    for model_name in name_order:
        print(model_name, len(results[model_name]))
        data.append(results[model_name])
        tick_labels.append(name_to_abrv[model_name])
    ax.boxplot(data, showmeans=True, meanline=True)
    ax.set_xticks(locs)
    ax.set_xticklabels(tick_labels, rotation=17)
    ax.set_title(predicate)
    ax.set_ylim(0, 120)

def explicit_plot(predicate, data_kind, result_file):
    """Since there are only five maps, might as well
    just plot these points directly."""

if __name__ == "__main__":
    predicate = "left"
    data_kind = "test"
    result_file = "foref_deviation.json"
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3), sharey=True)
    for i, predicate in enumerate(["front", "left"]):
        ax = axes[i]
        map_mean_boxplot(predicate, data_kind, result_file, ax)
    axes[0].set_ylabel("Angular Deviation (degrees)")
    fig.tight_layout()
    plt.savefig("together-plot.png", dpi=300)        

