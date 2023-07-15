"""Calculate p-value of destination"""
# %%
import os
import json
import argparse
import subprocess
from processes.pitman_yor_pvalue import PitmanYorPValue, StreamingPitmanYorPValue
from processes.ddcrp_pvalue import DDCRPPValue

def destination_process(user_args=None):
    parser = argparse.ArgumentParser(description='Parameter estimation')
    parser.add_argument('settings', type=str, nargs='+', help='File with settings')
    args = parser.parse_args(user_args)
    with open(args.settings[0], "r", encoding="utf-8") as file:
        settings = json.load(file)

    input_file = settings["input"]["filepath"]
    output_file = settings["phase1"]["dest_file"]
    params_file = settings["phase0"]["y_params_file"]
    process_type = settings["info"]["type"]
    twindow = settings["info"]["stream_time_window"]
    n_nodes = settings["info"]["n_nodes"]
    tstart = settings["phase1"]["tstart"]
    tend = settings["phase1"]["tend"]

    with open(params_file, "r", encoding="utf-8") as file:
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
    elif process_type == "POISSON+PY":
        raise NotImplementedError("Still to be implemented")
    else:
        raise ValueError("Unknown process type")

    with open(output_file, "w", encoding="utf-8") as out_file:
        with open(input_file, "r", encoding="utf-8") as file:
            for line in file:
                time, source, dest, anomaly = line.strip().split("\t")
                if float(time) > tstart:
                    if process_type in ("DP", "PY", "POISSON+PY"):
                        pvalue = py_pvalue.pvalue_and_update(dest)
                    else:
                        pvalue = py_pvalue.pvalue_and_update(dest, float(time))
                    out_file.write("\t".join([dest, time, source, anomaly, str(pvalue)]) + "\n")
                if float(time) > tend:
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
