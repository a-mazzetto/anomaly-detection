"""Anomaly detection parametrization"""
import os

def create_settings(
        input_file,
        output_folder,
        n_nodes,
        process_type,
        param_est_interval = [0, 1e8],
        param_est_threshold = 30,
        test_interval = [0, 1e8],
        stream_time_window = None,
        forgetting_factor = 0.99):
    """Create settings for anomaly detection job.
    Process type can be `DP`, `PY`, `DDCRP`, `STREAM_PY` or `POISSON_PY`"""
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
    settings["info"]["type"] = process_type
    settings["info"]["stream_time_window"] = stream_time_window
    settings["info"]["forgetting_factor"] = forgetting_factor
    settings["phase0"] = {}
    settings["phase0"]["tstart"] = param_est_interval[0]
    settings["phase0"]["tend"] = param_est_interval[1]
    settings["phase0"]["threshold"] = param_est_threshold
    settings["phase0"]["y_params_file"] = os.path.join(settings["output"]["root"], "phase0_y_params.txt")
    settings["phase0"]["x_y_params_file"] = os.path.join(settings["output"]["root"], "phase0_x_y_params.txt")
    settings["phase0"]["pois_params_file"] = os.path.join(settings["output"]["root"], "phase0_pois_params.txt")
    settings["phase1"] = {}
    settings["phase1"]["tstart"] = test_interval[0]
    settings["phase1"]["tend"] = test_interval[1]
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

    USE_REAL_PARAMETERS = False

    ##################
    ### PITMAN-YOR ###
    ##################
    settings = create_settings("./data/dataset_002/auth.txt", "./data/dataset_002/PY/", NNODES, process_type="PY",
                               param_est_interval=[0, 24 * 3600], test_interval=[24 * 3600, 1e8])
    settings_filename = "./data/dataset_002/PY/settings.json"
    with open(settings_filename, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)

    preprocessing([f"{settings_filename}"])
    if USE_REAL_PARAMETERS:
        pass
    destination_process([f"{settings_filename}"])
    source_conditional_process([f"{settings_filename}"])
    link_scores([f"{settings_filename}"])
    source_scores([f"{settings_filename}"])

    ##################
    ### DIRICHLET ###
    ##################
    settings_dp = create_settings("./data/dataset_002/auth.txt", "./data/dataset_002/DP/", NNODES, process_type="DP",
                                  param_est_interval=[0, 24 * 3600], test_interval=[24 * 3600, 1e8])
    settings_filename_dp = "./data/dataset_002/DP/settings.json"
    with open(settings_filename_dp, "w", encoding="utf-8") as file:
        json.dump(settings_dp, file, indent=4)

    preprocessing([f"{settings_filename_dp}"])
    if USE_REAL_PARAMETERS:
        pass
    destination_process([f"{settings_filename_dp}"])
    source_conditional_process([f"{settings_filename_dp}"])
    link_scores([f"{settings_filename_dp}"])
    source_scores([f"{settings_filename_dp}"])

    #############
    ### DDCRP ###
    #############
    settings_ddcrp = create_settings("./data/dataset_002/auth.txt", "./data/dataset_002/DDCRP/", NNODES, process_type="DDCRP",
                                     param_est_interval=[0, 24 * 3600], test_interval=[24 *3600, 1e8],
                                     stream_time_window=12 * 3600,)
    settings_filename_ddcrp = "./data/dataset_002/DDCRP/settings.json"
    with open(settings_filename_ddcrp, "w", encoding="utf-8") as file:
        json.dump(settings_ddcrp, file, indent=4)

    preprocessing([f"{settings_filename_ddcrp}"])
    if USE_REAL_PARAMETERS:
        pass
    destination_process([f"{settings_filename_ddcrp}"])
    source_conditional_process([f"{settings_filename_ddcrp}"])
    link_scores([f"{settings_filename_ddcrp}"])
    source_scores([f"{settings_filename_ddcrp}"])

    #################
    ### STREAM-PY ###
    #################
    settings_stream_py = create_settings("./data/dataset_002/auth.txt", "./data/dataset_002/STREAMPY/", NNODES, process_type="STREAM_PY",
                                         param_est_interval=[0, 24 * 3600], test_interval=[24 *3600, 1e8],
                                         stream_time_window=12 * 3600,)
    settings_filename_stream_py = "./data/dataset_002/STREAMPY/settings.json"
    with open(settings_filename_stream_py, "w", encoding="utf-8") as file:
        json.dump(settings_stream_py, file, indent=4)

    preprocessing([f"{settings_filename_stream_py}"])
    if USE_REAL_PARAMETERS:
        pass
    destination_process([f"{settings_filename_stream_py}"])
    source_conditional_process([f"{settings_filename_stream_py}"])
    link_scores([f"{settings_filename_stream_py}"])
    source_scores([f"{settings_filename_stream_py}"])

    #################
    ### POISSON-PY ###
    #################
    settings_stream_pois_py = create_settings("./data/dataset_002/auth.txt", "./data/dataset_002/POISPY/", NNODES, process_type="POISSON_PY",
                                         param_est_interval=[0, 24 * 3600], test_interval=[24 *3600, 1e8],
                                         stream_time_window=12 * 3600,)
    settings_filename_pois_py = "./data/dataset_002/POISPY/settings.json"
    with open(settings_filename_pois_py, "w", encoding="utf-8") as file:
        json.dump(settings_stream_pois_py, file, indent=4)

    preprocessing([f"{settings_filename_pois_py}"])
    if USE_REAL_PARAMETERS:
        pass
    destination_process([f"{settings_filename_pois_py}"])
    source_conditional_process([f"{settings_filename_pois_py}"])
    link_scores([f"{settings_filename_pois_py}"])
    source_scores([f"{settings_filename_pois_py}"])
