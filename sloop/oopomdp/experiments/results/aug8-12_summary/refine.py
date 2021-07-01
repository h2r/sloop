# Refine result tables
from spatial_foref.oopomdp.experiments.states_result import StatesResult
from spatial_foref.oopomdp.experiments.plotting import *
from spatial_foref.oopomdp.experiments.pd_utils import *
import pandas as pd
import os


def refine_detections(exp_path, baselines=None):
    """Detections result"""
    print("Processing detection results...")
    df = pd.read_csv(os.path.join(exp_path, "detections_success_steps.csv"))
    StatesResult.plot_success_count_vs_step_limit(
        df, exp_path,
        has_map=False,
        baselines=baselines)

def refine_prior_quality(exp_path):
    print("Processing prior quality results...")
    df = pd.read_csv(os.path.join(exp_path, "prior_quality.csv"), index_col=0)
    subs = {}
    for column in df.columns:
        if column in prior_to_label:
            subs[column] = prior_to_label[column]
    df = df.rename(columns=subs).transpose()
    df["mean (.95ci)"] = ["%.2f (%.2f)" % (df["mean"][i], df["ci"][i])
                          for i in range(len(df))]
    df.to_csv(os.path.join(exp_path, "prior_quality_refined.csv"),
              columns=["mean (.95ci)"], index=True)
    print(" --- Latex Table for Prior Qulaity ---")
    print(df.transpose().to_latex(index=True))

def refine_reward_discounted(exp_path):
    print("Processing reward discounted results...")
    df = pd.read_csv(os.path.join(exp_path, "reward_summary.csv"), index_col=0)
    df["disc_reward (ci)"] = ["%.2f (%.2f)" % (df["disc_reward-avg"][i], df["disc_reward-ci95"][i])
                              for i in range(len(df))]
    for prior_type in prior_to_label:
        df["prior_type"] = df["prior_type"].replace([prior_type], [prior_to_label[prior_type]])
    df.to_csv(os.path.join(exp_path, "reward_summary_refined.csv"),
              columns=["sensor_range", "prior_type", "disc_reward (ci)", "num_detected-count"], index=True)
    print(" --- Latex Table for Discounted Reward ---")
    print(df[["sensor_range", "prior_type", "disc_reward (ci)"]].transpose().to_latex(index=False))

def refine_reward_lang_numrels(exp_path, save=True):
    print("Processing reward lang numrels results...")
    df = pd.read_csv(os.path.join(exp_path, "rewards_lang-numrels_summary.csv"), index_col=0)
    df = df.rename(columns={"disc_reward-count": "count"})
    df["disc_reward (ci)"] = ["%.2f (%.2f)" % (df["disc_reward-avg"][i], df["disc_reward-ci95"][i])
                              for i in range(len(df))]
    for prior_type in prior_to_label:
        df["prior_type"] = df["prior_type"].replace([prior_type], [prior_to_label[prior_type]])

    tuples = [#("", "sensor_range"),
              ("", "num_rels")]
    for prior_type in sorted(df["prior_type"].unique()):
        tuples.extend([(prior_type, "avg (ci)"),
                       # (prior_type, "success"),
                       (prior_type, "count")])
    index = pd.MultiIndex.from_tuples(tuples, names=['prior_type', 'item'])

    rows = []
    for sensor_range in {3}:
        for num_rels in df["num_rels"].unique():
            row = [num_rels]
            for prior_type in sorted(df["prior_type"].unique()):
                filter_rules = (df["num_rels"] == num_rels)\
                    & (df["prior_type"] == prior_type) & (df["sensor_range"] == sensor_range)

                try:
                    avgci = df.loc[filter_rules].iloc[0]["disc_reward (ci)"]
                                   # & (df["num_detected"] == 1)].iloc[0]["disc_reward (ci)"]
                except IndexError:
                    avgci = "nan"

                # try:
                #     num_detected = df.loc[filter_rules\
                #                           & (df["num_detected"] == 1)].iloc[0]["count"]
                # except IndexError:
                #     num_detected = 0

                # try:
                #     num_missed = df.loc[filter_rules\
                #                         & (df["num_detected"] == 0)].iloc[0]["count"]
                # except IndexError:
                #     num_missed = 0

                try:
                    count = df.loc[filter_rules].iloc[0]["count"]
                except IndexError:
                    count = 0

                # rate = num_detected / max(1,(num_detected + num_missed))
                row.extend([avgci, count])#, "%.2f%%" % (rate*100), num_detected + num_missed])
            rows.append(row)
    df = pd.DataFrame(rows, columns=index)
    if save:
        df.to_csv(os.path.join(exp_path, "rewards_lang-numrels_summary_refined.csv"))
    return df


def refine_reward_lang_predicate(exp_path, fname="rewards_lang-predicates_summary.csv", save=True):
    print("Processing reward lang predicate results...")
    df = pd.read_csv(os.path.join(exp_path, fname), index_col=0)
    df = df.rename(columns={"disc_reward-count": "count"})
    df["disc_reward (ci)"] = ["%.2f (%.2f)" % (df["disc_reward-avg"][i], df["disc_reward-ci95"][i])
                              for i in range(len(df))]
    for prior_type in prior_to_label:
        df["prior_type"] = df["prior_type"].replace([prior_type], [prior_to_label[prior_type]])

    tuples = [#("", "sensor_range"),
              ("", "predicate")]
    for prior_type in sorted(df["prior_type"].unique()):
        tuples.extend([(prior_type, "avg (ci)"),
                       # (prior_type, "success"),
                       (prior_type, "count")])
    index = pd.MultiIndex.from_tuples(tuples, names=['prior_type', 'item'])

    rows = []
    for sensor_range in {3}:
        for predicate in df["predicate"].unique():
            row = [#sensor_range,
                   predicate]
            for prior_type in sorted(df["prior_type"].unique()):
                filter_rules = (df["predicate"] == predicate)\
                    & (df["prior_type"] == prior_type) & (df["sensor_range"] == sensor_range)
                try:
                    avgci = df.loc[filter_rules].iloc[0]["disc_reward (ci)"]
                                   # & (df["num_detected"] == 1)].iloc[0]["disc_reward (ci)"]
                except IndexError:
                    avgci = "nan"

                # try:
                #     num_detected = df.loc[(df["predicate"] == predicate)\
                #                            & (df["prior_type"] == prior_type)\
                #                            & (df["num_detected"] == 1)].iloc[0]["count"]
                # except IndexError:
                #     num_detected = 0

                # try:
                #     num_missed = df.loc[(df["predicate"] == predicate)\
                #                         & (df["prior_type"] == prior_type)\
                #                         & (df["num_detected"] == 0)].iloc[0]["count"]
                # except IndexError:
                #     num_missed = 0
                try:
                    count = df.loc[filter_rules].iloc[0]["count"]
                except IndexError:
                    count = 0

                # rate = num_detected / max(1,(num_detected + num_missed))
                row.extend([avgci, count])#"%.2f%%" % (rate*100), num_detected + num_missed])
            rows.append(row)
    df = pd.DataFrame(rows, columns=index)
    if save:
        df.to_csv(os.path.join(exp_path, "%s_refined.csv" % os.path.splitext(fname)[0]))
    return df


if __name__ == "__main__":
    # exp_path = "/home/kaiyuzh/repo/spatial-foref/spatial_foref/oopomdp/experiments/results/aug8-12_summary/all-joint"
    exp_path = "/home/kaiyuzh/repo/spatial-foref/spatial_foref/oopomdp/experiments/results/april-1-2021/all-joint"

    # # Candidate baslines
    # "informed",
    # "informed#5",
    # "informed#15",
    # "uniform",
    # "keyword*",
    # "keyword",
    # "rule#based#ego#ctx",
    # "rule#based#ego#ctx#auto"
    refine_detections(exp_path)
    refine_prior_quality(exp_path)
    refine_reward_discounted(exp_path)
    refine_reward_lang_numrels(exp_path)
    refine_reward_lang_predicate(exp_path)
    refine_reward_lang_predicate(exp_path, fname="rewards_lang-predicates_summary_good-forefs.csv")
    refine_reward_lang_predicate(exp_path, fname="rewards_lang-predicates_summary_bad-forefs.csv")
