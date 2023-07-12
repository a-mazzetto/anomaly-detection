"""Calculate p-value of destination"""
# %%
import os
import json
import argparse
import subprocess
from processes.pitman_yor_pvalue import PitmanYorPValue

parser = argparse.ArgumentParser(description='Parameter estimation')
parser.add_argument('settings', type=str, nargs='+', help='File with settings')
args = parser.parse_args()
with open(args.settings[0], "r", encoding="utf-8") as file:
    settings = json.load(file)

input_file = settings["input"]["filepath"]
output_file = settings["phase1"]["dest_file"]
params_file = settings["phase0"]["y_params_file"]
n_nodes = settings["info"]["n_nodes"]

with open(params_file, "r", encoding="utf-8") as file:
    count = 0
    for line in file:
        if count > 1:
            raise AssertionError("This file should have one line")
        alpha, d = line.strip().split("\t")
        alpha = float(alpha)
        d = float(d)
        count += 1

py_pvalue = PitmanYorPValue(alpha=alpha, d=d, n_nodes=n_nodes)

with open(output_file, "w", encoding="utf-8") as out_file:
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            time, source, dest, anomaly = line.strip().split("\t")
            pvalue = py_pvalue.pvalue_and_update(dest)
            out_file.write("\t".join([dest, time, source, anomaly, str(pvalue)]) + "\n")

# Sort file
completed = subprocess.run(["powershell", "-Command",
    f"Get-Content {output_file} | Sort-Object | Set-Content -Path {output_file}"],
    capture_output=True)
if completed.returncode != 0:
    print("An error occured: %s", completed.stderr)
else:
    print("Command executed successfully!")
# %%
