"""Combine source scores"""
# %%
import os
import subprocess
from input_parameters import *
from pvalues.combiners import fisher_pvalues_combiner

current_source = ""
current_source_scores = []

with open(SOURCE_SCORE_FILEPATH, "w", encoding="utf-8") as out_file:
    with open(LINK_SCORE_FILEPATH, "r", encoding="utf-8") as file:
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
