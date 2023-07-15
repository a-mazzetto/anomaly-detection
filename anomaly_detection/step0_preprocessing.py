"""Script to estimate parameters from dataset"""
# %%
import os
import json
import argparse
from collections import Counter
import numpy as np
from scipy.stats import beta
import subprocess
import matplotlib.pylab as plt
from parameter_estimation.parameter_estimation import pitman_yor_est_pars, dirichlet_est_pars

def preprocessing(user_args=None):
    parser = argparse.ArgumentParser(description='Parameter estimation')
    parser.add_argument('settings', type=str, nargs='+', help='File with settings')
    args = parser.parse_args(user_args)
    with open(args.settings[0], "r", encoding="utf-8") as file:
        settings = json.load(file)

    input_file = settings["input"]["filepath"]
    output1_file = settings["phase0"]["y_params_file"]
    output2_file = settings["phase0"]["x_y_params_file"]
    n_nodes = settings["info"]["n_nodes"]
    result_folder = settings["output"]["root"]
    process_type = settings["info"]["type"]
    tstart = settings["phase0"]["tstart"]
    tend = settings["phase0"]["tend"]
    threshold = settings["phase0"]["threshold"]

    destination_counter = Counter()
    source_counters = {}
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            time, source, dest, _ = line.strip().split("\t")
            time = float(time)
            if time >= tstart and time <= tend:
                if dest not in destination_counter:
                    destination_counter[dest] = 1
                    source_counters[dest] = Counter()
                    source_counters[dest][source] = 1
                else:
                    destination_counter[dest] += 1
                    if source not in source_counters[dest]:
                        source_counters[dest][source] = 1
                    else:
                        source_counters[dest][source] += 1
            if time > tend:
                break

    # Estimate destination process parameters
    if process_type in ("DP", "DDCRP"):
        destination_process_params = dirichlet_est_pars(
            meas_kn=len(destination_counter),
            n=sum(destination_counter.values()),
            n_nodes=n_nodes)
    elif process_type in ("PY", "STREAM_PY", "POISSON+PY"):
        destination_process_params = pitman_yor_est_pars(
            meas_kn=len(destination_counter),
            meas_h1n=sum(np.isclose(list(destination_counter.values()), 1)),
            n=sum(destination_counter.values()),
            n_nodes=n_nodes)
    else:
        raise ValueError("Unknown process type")
    destination_process_params = [destination_process_params] if \
        isinstance(destination_process_params, float) else destination_process_params

    # Estimate source processes parameters
    dest_list = list(source_counters.keys())
    dest_alpha = np.nan * np.ones(len(dest_list))
    dest_d = np.nan * np.ones_like(dest_alpha)
    for i, dest in enumerate(dest_list):
        counter = source_counters[dest]
        if sum(counter.values()) > threshold:
            if process_type in ("DP", "DDCRP"):
                dest_alpha[i] = dirichlet_est_pars(
                    meas_kn=len(counter),
                    n=sum(counter.values()),
                    n_nodes=n_nodes)
            elif process_type in ("PY", "STREAM_PY", "POISSON+PY"):
                dest_alpha[i], dest_d[i] = pitman_yor_est_pars(
                    meas_kn=len(counter),
                    meas_h1n=sum(np.isclose(list(counter.values()), 1)),
                    n=sum(counter.values()),
                    n_nodes=n_nodes)
            else:
                raise ValueError("Unknown process type")

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(dest_alpha, density=True)
    ax[0].set_xlabel(r"$\alpha$")
    ax[0].vlines(x=2.0, ymin=0, ymax=0.6,
                linestyle="--", color="red")
    ax[0].vlines(x=7.0, ymin=0, ymax=0.3,
                linestyle="--", color="red")
    ax[0].vlines(x=12.0, ymin=0, ymax=0.1,
                linestyle="--", color="red")
    if not np.all(np.isnan(dest_d)):
        ax[1].hist(dest_d, density=True)
        x_plot = np.linspace(0, 1, 100)
        ax[1].plot(x_plot, beta(a=2.0, b=5.0).pdf(x_plot), linestyle="--", color="red")
        ax[1].set_xlabel("$d$")
    fig.savefig(os.path.join(result_folder, "preprocessing.pdf"))

    # Fill NAN
    dest_alpha[np.isnan(dest_alpha)] = np.nanmedian(dest_alpha)
    if not np.all(np.isnan(dest_d)):
        dest_d[np.isnan(dest_d)] = np.nanmedian(dest_d)

    # Save data to file
    with open(output1_file, "w", encoding="utf-8") as file:
        file.write("\t".join([str(i) for i in destination_process_params]) + "\n")

    with open(output2_file, "w", encoding="utf-8") as file:
        if not np.all(np.isnan(dest_d)):
            for line in zip(dest_list, dest_alpha.astype(str), dest_d.astype(str)):
                file.write("\t".join(line) + "\n")
            file.write("\t".join(["average", np.nanmedian(dest_alpha).astype(str),
                                  np.nanmedian(dest_d).astype(str)]) + "\n")
        else:
            for line in zip(dest_list, dest_alpha.astype(str)):
                file.write("\t".join(line) + "\n")
            file.write("\t".join(["average", np.nanmedian(dest_alpha).astype(str)]) + "\n")

    # Sort file
    completed = subprocess.run(["powershell", "-Command",
        f"Get-Content {output2_file} | Sort-Object | Set-Content -Path {output2_file}"],
        capture_output=True)
    if completed.returncode != 0:
        print("An error occured: %s", completed.stderr)
    else:
        print("Command executed successfully!")

if __name__ == "__main__":
    preprocessing()
