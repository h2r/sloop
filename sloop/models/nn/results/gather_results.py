import json
import argparse
import os
import re
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

class Result:
    NAME = "result"
    @classmethod
    def read(cls, filepath):
        raise NotImplementedError
    
    @classmethod
    def aggregate_results(cls, results, output_dir):
        """results is a dictionary mapping from baseline name to result.
        This will then save relevant things to the output_dir for this result."""
        raise NotImplementedError

    @classmethod
    def file_to_type(cls, filename):
        FILE_TO_TYPE = {
            "infomation_metrics.json": InfoMetrics,
            "information_metrics.json": InfoMetrics,
            "rule_based_information_metrics.json": InfoMetrics,
            "rule_based_infomation_metrics.json": InfoMetrics,
            "foref_deviation.json": ForefDeviation
        }
        return FILE_TO_TYPE[filename]

    

class InfoMetrics(Result):
    NAME = "info_metrics"
    @classmethod
    def read(cls, filepath):
        with open(filepath) as f:
            result = json.load(f)
        return result["__summary__"]

    @classmethod
    def aggregate_results(cls, results, output_dir):
        """results is a dictionary mapping {data_kind -> {baseline -> result}}
        This will then save relevant things"""
        # Save a data frame.
        for data_kind in results:
            rr = {"method": [],
                  "keyword": [],
                  "test_map": [],
                  "distance":[],
                  "distance_ci":[],
                  "kl_div": [],
                  "kl_div_ci": [],
                  "perplex_pred": [],
                  "perplex_pred_ci": [],
                  "perplex_true": [],
                  "perplex_true_ci": []}
            for baseline in results[data_kind]:
                method = baseline.split(":")[0]
                keyword = baseline.split(":")[1]
                test_map = baseline.split(":")[2]
                rr["method"].append(method)
                rr["keyword"].append(keyword)
                rr["test_map"].append(test_map) 
                
                result = results[data_kind][baseline]
                for item in result:
                    rr[item].append(result[item]["mean"])
                    rr["%s_ci" % item].append(result[item]["ci-95"])
            df = pd.DataFrame(rr)
            df.to_csv(os.path.join(output_dir, "%s-%s-results.csv" % (data_kind, cls.NAME)))


class ForefDeviation(Result):
    NAME = "foref_deviation"
    @classmethod
    def read(cls, filepath):
        with open(filepath) as f:
            result = json.load(f)
        return result["__summary__"]

    @classmethod
    def aggregate_results(cls, results, output_dir):
        """results is a dictionary mapping {data_kind -> {baseline -> result}}
        This will then save relevant things"""
        # Save a data frame.
        for data_kind in results:
            rr = {"method": [],
                  "keyword": [],
                  "test_map": [],
                  "true_pred_angle_diff": [],
                  "true_pred_angle_diff_ci":[],
                  "true_pred_origin_diff": [],
                  "true_pred_origin_diff_ci":[]}                  
            for baseline in results[data_kind]:
                method = baseline.split(":")[0]
                keyword = baseline.split(":")[1]
                test_map = baseline.split(":")[2]
                rr["method"].append(method)
                rr["keyword"].append(keyword)
                rr["test_map"].append(test_map)                 
                
                result = results[data_kind][baseline]
                for item in result:
                    rr[item].append(result[item]["mean"])
                    rr["%s_ci" % item].append(result[item]["ci-95"])
            df = pd.DataFrame(rr)
            df.to_csv(os.path.join(output_dir, "%s-%s-results.csv" % (data_kind, cls.NAME)))


def main():
    parser = argparse.ArgumentParser(description="Aggregate results of baselines")
    parser.add_argument("results_dir", type=str,
                        help="Path to directory with results."
                        "Each result should be stored in a directory iter#_baseline_timestamp")
    parser.add_argument("result_file", type=str,
                        help="The filename of the metric result you want. E.g. information_metrics.json")
    parser.add_argument("output_dir", type=str,
                        help="Directory to save the gathered results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    for root, dirs, files in os.walk(args.results_dir):
        # root: top-level directory (recursive)
        # dirs: direct subdirectories of root
        # files: files directly under root
        rootdir = os.path.basename(root)
        if not (rootdir.startswith("iter")\
                and len(rootdir.split("_")) == 3):
            print("Skipped %s" % rootdir)
            continue

        itername = rootdir.split("_")[0]
        timestamp = rootdir.split("_")[-1]
        baseline_name = "_".join(rootdir.split("_")[1:-1])
        print("Reading %s (%s)" % (rootdir, baseline_name))

        for data_kind in os.listdir(os.path.join(root, "metrics")):
            # data_kind could be train/test/val.
            if data_kind not in {"train", "test", "val"}:
                print("Unknown data kind %s" % data_kind)
                continue
            result_path = os.path.join(root,
                                       "metrics", data_kind, args.result_file)
            ResultCls = Result.file_to_type(args.result_file)
            result = ResultCls.read(result_path)

            if ResultCls not in results:
                results[ResultCls] = {}
            if data_kind not in results[ResultCls]:
                results[ResultCls][data_kind] = {}
            results[ResultCls][data_kind][baseline_name] = result
            
    for result_cls in results:
        # Gather and save results
        print("Aggregated results for %s" % result_cls.NAME)
        result_cls.aggregate_results(results[result_cls], args.output_dir)
        

if __name__ == "__main__":
    main()
