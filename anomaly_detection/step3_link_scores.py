"""Calculate p-value of destination"""
# %%
import json
import argparse
import numpy as np
from pvalues.combiners import min_pvalue_combiner
from .utils import switched_open

def link_scores(user_args = None):
    parser = argparse.ArgumentParser(description='Parameter estimation')
    parser.add_argument('settings', type=str, nargs='+', help='File with settings')
    args = parser.parse_args(user_args)
    with switched_open(args.settings[0], "r") as file:
        settings = json.load(file)

    input_file = settings["phase2"]["dest_file"]
    output_file = settings["phase3"]["dest_file"]

    current_link = ""
    current_link_scores = []
    current_link_times = []

    with switched_open(output_file, "w") as out_file:
        with switched_open(input_file, "r") as file:
            for line in file:
                link, time, _, _, _, score = line.strip().split("\t")
                if link != current_link:
                    if len(current_link) > 0:
                        source, dest = current_link.split("_")
                        link_score = min_pvalue_combiner(np.array(current_link_scores))
                        time_at_min = current_link_times[np.argmin(current_link_scores)]
                        out_file.write("\t".join([source, dest, str(link_score), str(time_at_min)]) + "\n")
                    current_link = link
                    current_link_scores = []
                    current_link_times = []
                current_link_scores.append(float(score))
                current_link_times.append(float(time))
        source, dest = current_link.split("_")
        link_score = min_pvalue_combiner(np.array(current_link_scores))
        time_at_min = current_link_times[np.argmin(current_link_scores)]
        out_file.write("\t".join([source, dest, str(link_score), str(time_at_min)]) + "\n")

    # Output file already sorted

if __name__ == "__main__":
    link_scores()
