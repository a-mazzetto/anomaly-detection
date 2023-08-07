"""Calculate p-value of destination"""
# %%
import os
import json
import argparse
import subprocess
from parameter_estimation.parameter_estimation import PoissonProcessRateEstimation
from processes.pitman_yor_pvalue import PitmanYorPValue, StreamingPitmanYorPValue
from processes.ddcrp_pvalue import DDCRPPValue
from processes.poisson_pvalue import PitmanYorMarkedPPPValue
from .utils import switched_open

def destination_process(user_args=None):
    parser = argparse.ArgumentParser(description='Parameter estimation')
    parser.add_argument('settings', type=str, nargs='+', help='File with settings')
    args = parser.parse_args(user_args)
    with switched_open(args.settings[0], "r") as file:
        settings = json.load(file)

    input_file = settings["input"]["filepath"]
    output_file = settings["phase1"]["dest_file"]
    params_file = settings["phase0"]["y_params_file"]
    process_type = settings["info"]["type"]
    twindow = settings["info"]["stream_time_window"]
    forgetting_factor = settings["info"]["forgetting_factor"]
    n_nodes = settings["info"]["n_nodes"]
    training_tstart = settings["phase0"]["tstart"]
    tstart = settings["phase1"]["tstart"]
    tend = settings["phase1"]["tend"]

    with switched_open(params_file, "r") as file:
        count = 0
        for line in file:
            if count > 1:
                raise AssertionError("This file should have one line")
            params = line.strip().split("\t")
            alpha = float(params[0])
            d = float(params[1]) if len(params) > 1 else None
            count += 1

    if process_type == "DP":
        py_pvalue = PitmanYorPValue(alpha=alpha, d=0, n_nodes=n_nodes)
    elif process_type == "PY":
        py_pvalue = PitmanYorPValue(alpha=alpha, d=d, n_nodes=n_nodes)
    elif process_type == "DDCRP":
        py_pvalue = DDCRPPValue(alpha=alpha, beta=twindow, n_nodes=n_nodes)
    elif process_type == "STREAM_PY":
        py_pvalue = StreamingPitmanYorPValue(twindow=twindow, alpha=alpha, d=d, n_nodes=n_nodes)
    elif process_type == "POISSON_PY":
        poisson_rate = PoissonProcessRateEstimation(
            forgetting_factor=forgetting_factor, t_start=min(training_tstart, tstart))
        py_pvalue_s = PitmanYorMarkedPPPValue(alpha=alpha, d=d, n_nodes=n_nodes)
    else:
        raise ValueError("Unknown process type")

    with switched_open(output_file, "w") as out_file:
        with switched_open(input_file, "r") as file:
            for line in file:
                time, source, dest, anomaly = line.strip().split("\t")
                time = float(time)
                # Calculations are necessary also in the training phase, and will be disregarded in the next step,
                # after calculating source p-values
                if time > min(training_tstart, tstart):  
                    if process_type in ("DP", "PY"):
                        pvalue = py_pvalue.pvalue_and_update(dest)
                        poisson_pvalue = None
                    elif process_type == "POISSON_PY":
                        poisson_rate.update(time)
                        pvalue, poisson_pvalue = py_pvalue_s.pvalue_and_update(time, poisson_rate.rate_est, dest)
                    else:
                        pvalue = py_pvalue.pvalue_and_update(dest, time)
                        poisson_pvalue = None
                    out_file.write("\t".join([dest, str(time), source, anomaly, str(pvalue), str(poisson_pvalue)]) + "\n")
                if time > tend:
                    break

    # Sort file
    completed = subprocess.run(["powershell", "-Command",
        f"Get-Content {output_file} | Sort-Object | Set-Content -Path {output_file}"],
        capture_output=True)
    if completed.returncode != 0:
        print("An error occured: %s", completed.stderr)
    else:
        print("Command executed successfully!")

if __name__ == "__main__":
    destination_process()
