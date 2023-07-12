"""Calculate p-value of destination"""
# %%
import os
import subprocess
from input_parameters import *
from pvalues.combiners import min_pvalue_combiner

current_link = ""
current_link_scores = []

with open(LINK_SCORE_FILEPATH, "w", encoding="utf-8") as out_file:
    with open(SOURCE_GIVEN_DEST_PVALUES_FILEPATH_PY, "r", encoding="utf-8") as file:
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
