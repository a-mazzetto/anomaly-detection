"""Calculate p-value of destination"""
# %%
import os
import json
import argparse
import subprocess
from processes.pitman_yor_pvalue import PitmanYorPValue
from pvalues.combiners import fisher_pvalues_combiner

parser = argparse.ArgumentParser(description='Parameter estimation')
parser.add_argument('settings', type=str, nargs='+', help='File with settings')
args = parser.parse_args()
with open(args.settings[0], "r", encoding="utf-8") as file:
    settings = json.load(file)

input_file = settings["phase1"]["dest_file"]
output_file = settings["phase2"]["dest_file"]
params_file = settings["phase0"]["x_y_params_file"]
n_nodes = settings["info"]["n_nodes"]

def look_for_params(dest_query):
    with open(params_file, "r", encoding="utf-8") as file:
        for line in file:
            dest, alpha, d = line.strip().split("\t")
            if dest == dest_query:
                return alpha, d
    raise ValueError(f"Row relative to {dest_query} not found")

current_dest = ""

with open(output_file, "w", encoding="utf-8") as out_file:
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            dest, time, source, anomaly, y_pvalue = line.strip().split("\t")
            if dest != current_dest:
                current_dest = dest
                alpha, d = look_for_params(dest)
                py_pvalue = PitmanYorPValue(alpha=float(alpha), d=float(d), n_nodes=n_nodes)
            x_given_y_pvalue = py_pvalue.pvalue_and_update(source)
            # Combine p-values
            score = fisher_pvalues_combiner(float(y_pvalue), x_given_y_pvalue)
            out_file.write("\t".join(["_".join((source, dest)), time, anomaly, y_pvalue, str(x_given_y_pvalue), str(score)]) + "\n")

# Sort file
completed = subprocess.run(["powershell", "-Command",
    f"Get-Content {output_file} | Sort-Object | Set-Content -Path {output_file}"],
    capture_output=True)
if completed.returncode != 0:
    print("An error occured: %s", completed.stderr)
else:
    print("Command executed successfully!")

# %%
