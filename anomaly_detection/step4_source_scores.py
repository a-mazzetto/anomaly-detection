"""Combine source scores"""
# %%
import json
import argparse
import numpy as np
from pvalues.combiners import min_pvalue_combiner

def source_scores(user_args=None):
    parser = argparse.ArgumentParser(description='Parameter estimation')
    parser.add_argument('settings', type=str, nargs='+', help='File with settings')
    args = parser.parse_args(user_args)
    with open(args.settings[0], "r", encoding="utf-8") as file:
        settings = json.load(file)

    input_file = settings["phase3"]["dest_file"]
    output_file = settings["phase4"]["dest_file"]

    current_source = ""
    current_source_scores = []
    current_source_times = []

    with open(output_file, "w", encoding="utf-8") as out_file:
        with open(input_file, "r", encoding="utf-8") as file:
            for line in file:
                source, _, score, time = line.strip().split("\t")
                if source != current_source:
                    if len(current_source) > 0:
                        current_source_score = min_pvalue_combiner(np.array(current_source_scores))
                        time_at_min = current_source_times[np.argmin(current_source_scores)]
                        out_file.write("\t".join([current_source, str(current_source_score), str(time_at_min)]) + "\n")
                    current_source = source
                    current_source_scores = []
                    current_source_times = []
                current_source_scores.append(float(score))
                current_source_times.append(float(time))
        current_source_score = min_pvalue_combiner(np.array(current_source_scores))
        time_at_min = current_source_times[np.argmin(current_source_scores)]
        out_file.write("\t".join([current_source, str(current_source_score), str(time_at_min)]) + "\n")

    # Output file to be sorted by score

if __name__ == "__main__":
    source_scores()
