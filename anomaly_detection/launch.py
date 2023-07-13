"""Anomaly detection parametrization"""
import os

def create_settings(
        input_file,
        output_folder,
        n_nodes,
        param_est_interval = [0, 1e8],
        param_est_threshold = 30,
        ddcrp=False,
        beta_ddcrp=None):
    """Create settings for anomaly detection job"""
    settings = {}
    settings["input"] = {}
    settings["input"]["filepath"] = input_file
    settings["input"]["filename"] = ".".join(
        os.path.basename(input_file).split(".")[:-1])
    settings["output"] = {}
    settings["output"]["root"] = output_folder
    if not os.path.exists(settings["output"]["root"]):
        os.mkdir(settings["output"]["root"])
    settings["info"] = {}
    settings["info"]["n_nodes"] = n_nodes
    settings["info"]["ddcrp"] = ddcrp
    settings["info"]["beta_ddcrp"] = beta_ddcrp
    settings["phase0"] = {}
    settings["phase0"]["tstart"] = param_est_interval[0]
    settings["phase0"]["tend"] = param_est_interval[1]
    settings["phase0"]["threshold"] = param_est_threshold
    settings["phase0"]["y_params_file"] = os.path.join(settings["output"]["root"], "phase0_y_params.txt")
    settings["phase0"]["x_y_params_file"] = os.path.join(settings["output"]["root"], "phase0_x_y_params.txt")
    settings["phase1"] = {}
    settings["phase1"]["dest_file"] = os.path.join(settings["output"]["root"], "phase1_dest_pval.txt")
    settings["phase2"] = {}
    settings["phase2"]["dest_file"] = os.path.join(settings["output"]["root"], "phase2_source_pval.txt")
    settings["phase3"] = {}
    settings["phase3"]["dest_file"] = os.path.join(settings["output"]["root"], "phase3_link_score.txt")
    settings["phase4"] = {}
    settings["phase4"]["dest_file"] = os.path.join(settings["output"]["root"], "phase_4_source_score.txt")
    return settings

if __name__ == "__main__":
    import json
    from data_generation.constants import NNODES
    from anomaly_detection.step0_preprocessing import preprocessing
    from anomaly_detection.step1_destination_p_value import destination_process
    from anomaly_detection.step2_source_p_value_py import source_conditional_process
    from anomaly_detection.step3_link_scores import link_scores
    from anomaly_detection.step4_source_scores import source_scores

    settings = create_settings(".//data//auth.txt", ".//data//auth//", NNODES)
    settings_filename = ".//data//auth//settings.json"
    with open(settings_filename, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)

    preprocessing([f"{settings_filename}"])
    destination_process([f"{settings_filename}"])
    source_conditional_process([f"{settings_filename}"])
    link_scores([f"{settings_filename}"])
    source_scores([f"{settings_filename}"])

    settings_ddcrp = create_settings(".//data//auth.txt", ".//data//auth_ddcrp//", NNODES, ddcrp=True, beta_ddcrp=12 * 60 * 60)
    settings_filename_ddcrp = ".//data//auth_ddcrp//settings.json"
    with open(settings_filename_ddcrp, "w", encoding="utf-8") as file:
        json.dump(settings_ddcrp, file, indent=4)

    preprocessing([f"{settings_filename_ddcrp}"])
    destination_process([f"{settings_filename_ddcrp}"])
    source_conditional_process([f"{settings_filename_ddcrp}"])
    link_scores([f"{settings_filename_ddcrp}"])
    source_scores([f"{settings_filename_ddcrp}"])
