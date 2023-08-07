"""Calculate p-value of destination"""
# %%
import os
import json
import argparse
import subprocess
from warnings import showwarning
import numpy as np
from processes.pitman_yor_pvalue import PitmanYorPValue, StreamingPitmanYorPValue
from processes.ddcrp_pvalue import DDCRPPValue
from pvalues.combiners import fisher_pvalues_combiner
from .utils import switched_open

def source_conditional_process(user_args=None):
    parser = argparse.ArgumentParser(description='Parameter estimation')
    parser.add_argument('settings', type=str, nargs='+', help='File with settings')
    args = parser.parse_args(user_args)
    with switched_open(args.settings[0], "r") as file:
        settings = json.load(file)

    input_file = settings["phase1"]["dest_file"]
    output_file = settings["phase2"]["dest_file"]
    params_file = settings["phase0"]["x_y_params_file"]
    n_nodes = settings["info"]["n_nodes"]
    tstart = settings["phase1"]["tstart"]

    process_type = settings["info"]["type"]
    twindow = settings["info"]["stream_time_window"]

    def look_for_params(dest_query):
        with switched_open(params_file, "r") as file:
            for line in file:
                _input = line.strip().split("\t")
                dest = _input[0]
                if dest == dest_query:
                    alpha = _input[1]
                    d = _input[2] if len(_input) > 2 else None
                    if d is not None:
                        return float(alpha), float(d)
                    else:
                        return float(alpha)
                elif dest == "average":
                    avg_alpha = _input[1]
                    avg_d = _input[2] if len(_input) > 2 else None
        showwarning(f"Row relative to {dest_query} not found, using average", UserWarning, filename="look_for_params", lineno="")
        if avg_d is not None:
            return float(avg_alpha), float(avg_d)
        else:
            return float(avg_alpha)

    current_dest = ""

    with switched_open(output_file, "w") as out_file:
        with switched_open(input_file, "r") as file:
            for line in file:
                dest, time, source, anomaly, y_pvalue, t_pvalue = line.strip().split("\t")
                if dest != current_dest:
                    current_dest = dest
                    if process_type == "DP":
                        alpha = look_for_params(dest)
                        py_pvalue = PitmanYorPValue(alpha=alpha, d=0, n_nodes=n_nodes)
                    elif process_type in ("PY", "POISSON_PY"):
                        alpha, d = look_for_params(dest)
                        py_pvalue = PitmanYorPValue(alpha=alpha, d=d, n_nodes=n_nodes)
                    elif process_type == "DDCRP":
                        alpha = look_for_params(dest)
                        py_pvalue = DDCRPPValue(alpha=alpha, beta=twindow, n_nodes=n_nodes)
                    elif process_type == "STREAM_PY":
                        alpha, d = look_for_params(dest)
                        py_pvalue = StreamingPitmanYorPValue(twindow=twindow, alpha=alpha, d=d, n_nodes=n_nodes)
                    else:
                        raise ValueError("Unknown process type")
                if process_type in ("DP", "PY", "POISSON_PY"):
                    x_given_y_pvalue = py_pvalue.pvalue_and_update(source)
                else:
                    x_given_y_pvalue = py_pvalue.pvalue_and_update(source, float(time))
                # Output only after training period
                if float(time) > tstart:
                    # Combine p-values
                    if t_pvalue != "None":
                        score = fisher_pvalues_combiner(np.array([float(y_pvalue), x_given_y_pvalue, float(t_pvalue)]))
                    else:
                        score = fisher_pvalues_combiner(np.array([float(y_pvalue), x_given_y_pvalue]))
                    out_file.write("\t".join(["_".join((source, dest)), time, anomaly, y_pvalue, str(x_given_y_pvalue), str(score)]) + "\n")

    # Sort file
    completed = subprocess.run(["powershell", "-Command",
        f"Get-Content {output_file} | Sort-Object | Set-Content -Path {output_file}"],
        capture_output=True)
    if completed.returncode != 0:
        print("An error occured: %s", completed.stderr)
    else:
        print("Command executed successfully!")

if __name__ == "__main__":
    source_conditional_process()
