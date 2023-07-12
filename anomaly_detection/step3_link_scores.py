"""Calculate p-value of destination"""
# %%
import json
import argparse
from pvalues.combiners import min_pvalue_combiner

parser = argparse.ArgumentParser(description='Parameter estimation')
parser.add_argument('settings', type=str, nargs='+', help='File with settings')
args = parser.parse_args()
with open(args.settings[0], "r", encoding="utf-8") as file:
    settings = json.load(file)

input_file = settings["phase2"]["dest_file"]
output_file = settings["phase3"]["dest_file"]

current_link = ""
current_link_scores = []

with open(output_file, "w", encoding="utf-8") as out_file:
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            link, _, _, _, _, score = line.strip().split("\t")
            if link != current_link:
                source, dest = link.split("_")
                if len(current_link) > 0:
                    link_score = min_pvalue_combiner(*current_link_scores)
                    out_file.write("\t".join([source, dest, str(link_score)]) + "\n")
                current_link = link
                current_link_scores = []
            current_link_scores.append(float(score))
    link_score = min_pvalue_combiner(*current_link_scores)
    out_file.write("\t".join([source, dest, str(link_score)]) + "\n")

# Output file already sorted

# %%
