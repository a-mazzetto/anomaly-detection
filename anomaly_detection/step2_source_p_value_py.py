"""Calculate p-value of destination"""
# %%
import os
import json
import argparse
import subprocess
from processes.pitman_yor_pvalue import PitmanYorPValue
from processes.ddcrp_pvalue import DDCRPPValue
from pvalues.combiners import fisher_pvalues_combiner

def source_conditional_process(user_args=None):
    parser = argparse.ArgumentParser(description='Parameter estimation')
    parser.add_argument('settings', type=str, nargs='+', help='File with settings')
    args = parser.parse_args(user_args)
    with open(args.settings[0], "r", encoding="utf-8") as file:
        settings = json.load(file)

    input_file = settings["phase1"]["dest_file"]
    output_file = settings["phase2"]["dest_file"]
    params_file = settings["phase0"]["x_y_params_file"]
    n_nodes = settings["info"]["n_nodes"]

    is_ddcrp = settings["info"]["ddcrp"]
    ddcrp_beta = settings["info"]["beta_ddcrp"]

    def look_for_params(dest_query):
        with open(params_file, "r", encoding="utf-8") as file:
            for line in file:
                _input = line.strip().split("\t")
                dest = _input[0]
                if dest == dest_query:
                    alpha = _input[1]
                    d = _input[2] if len(_input) > 2 else None
                    if d is not None:
                        return alpha, d
                    else:
                        return alpha
        raise ValueError(f"Row relative to {dest_query} not found")

    current_dest = ""

    with open(output_file, "w", encoding="utf-8") as out_file:
        with open(input_file, "r", encoding="utf-8") as file:
            for line in file:
                dest, time, source, anomaly, y_pvalue = line.strip().split("\t")
                if dest != current_dest:
                    current_dest = dest
                    if is_ddcrp:
                        alpha = look_for_params(dest)
                        py_pvalue = DDCRPPValue(alpha=float(alpha), beta=float(ddcrp_beta), n_nodes=n_nodes)
                    else:
                        alpha, d = look_for_params(dest)
                        py_pvalue = PitmanYorPValue(alpha=float(alpha), d=float(d), n_nodes=n_nodes)
                if is_ddcrp:
                    x_given_y_pvalue = py_pvalue.pvalue_and_update(source, float(time))
                else:
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

if __name__ == "__main__":
    source_conditional_process()
