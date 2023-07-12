"""Calculate p-value of destination"""
# %%
import os
import subprocess
from input_parameters import *
from processes.pitman_yor_pvalue import PitmanYorPValue

y_params_filepath = os.path.join(RESULTS_FOLDER, Y_PARAMETERS_FILENAME)
with open(y_params_filepath, "r", encoding="utf-8") as file:
    count = 0
    for line in file:
        if count > 1:
            raise AssertionError("This file should have one line")
        alpha, d = line.strip().split("\t")
        alpha = float(alpha)
        d = float(d)
        count += 1

py_pvalue = PitmanYorPValue(alpha=alpha, d=d, n_nodes=N_NODES)

with open(DESTINATION_PVALUES_FILEPATH, "w", encoding="utf-8") as out_file:
    with open(FILE_PATH, "r", encoding="utf-8") as file:
        for line in file:
            time, source, dest, anomaly = line.strip().split("\t")
            pvalue = py_pvalue.pvalue_and_update(dest)
            out_file.write("\t".join([dest, time, source, anomaly, str(pvalue)]) + "\n")

# Sort file
completed = subprocess.run(["powershell", "-Command",
    f"Get-Content {DESTINATION_PVALUES_FILEPATH} | Sort-Object | Set-Content -Path {DESTINATION_PVALUES_FILEPATH}"],
    capture_output=True)
if completed.returncode != 0:
    print("An error occured: %s", completed.stderr)
else:
    print("Command executed successfully!")
# %%
