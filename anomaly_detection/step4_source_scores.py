"""Combine source scores"""
# %%
import json
import argparse
from pvalues.combiners import fisher_pvalues_combiner

parser = argparse.ArgumentParser(description='Parameter estimation')
parser.add_argument('settings', type=str, nargs='+', help='File with settings')
args = parser.parse_args()
with open(args.settings[0], "r", encoding="utf-8") as file:
    settings = json.load(file)

input_file = settings["phase3"]["dest_file"]
output_file = settings["phase4"]["dest_file"]

current_source = ""
current_source_scores = []

with open(output_file, "w", encoding="utf-8") as out_file:
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            source, _, score = line.strip().split("\t")
            if source != current_source:
                if len(current_source) > 0:
                    source_score = fisher_pvalues_combiner(*current_source_scores)
                    out_file.write("\t".join([source, str(source_score)]) + "\n")
                current_source = source
                current_source_scores = []
            current_source_scores.append(float(score))
    source_score = fisher_pvalues_combiner(*current_source_scores)
    out_file.write("\t".join([source, str(source_score)]) + "\n")

# Output file to be sorted by score

# %%
