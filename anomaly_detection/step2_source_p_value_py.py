"""Calculate p-value of destination"""
# %%
import os
import subprocess
from input_parameters import *
from processes.pitman_yor_pvalue import PitmanYorPValue
from pvalues.combiners import fisher_pvalues_combiner

x_given_y_params_filepath = os.path.join(RESULTS_FOLDER, X_GIVEN_Y_PARAMETERS_FILENAME)

def look_for_params(dest_query):
    with open(x_given_y_params_filepath, "r", encoding="utf-8") as file:
        for line in file:
            dest, alpha, d = line.strip().split("\t")
            if dest == dest_query:
                return alpha, d
    raise ValueError(f"Row relative to {dest_query} not found")

current_dest = ""

with open(SOURCE_GIVEN_DEST_PVALUES_FILEPATH_PY, "w", encoding="utf-8") as out_file:
    with open(DESTINATION_PVALUES_FILEPATH, "r", encoding="utf-8") as file:
        for line in file:
            dest, time, source, anomaly, y_pvalue = line.strip().split("\t")
            if dest != current_dest:
                current_dest = dest
                alpha, d = look_for_params(dest)
                py_pvalue = PitmanYorPValue(alpha=float(alpha), d=float(d), n_nodes=N_NODES)
            x_given_y_pvalue = py_pvalue.pvalue_and_update(source)
            # Combine p-values
            score = fisher_pvalues_combiner(float(y_pvalue), x_given_y_pvalue)
            out_file.write("\t".join(["_".join((source, dest)), time, anomaly, y_pvalue, str(x_given_y_pvalue), str(score)]) + "\n")

# Sort file
completed = subprocess.run(["powershell", "-Command",
    f"Get-Content {SOURCE_GIVEN_DEST_PVALUES_FILEPATH_PY} | Sort-Object | Set-Content -Path {SOURCE_GIVEN_DEST_PVALUES_FILEPATH_PY}"],
    capture_output=True)
if completed.returncode != 0:
    print("An error occured: %s", completed.stderr)
else:
    print("Command executed successfully!")

# %%
