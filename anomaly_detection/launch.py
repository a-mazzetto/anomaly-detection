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
        forgetting_factor = 0.99,
        gzip=False):
    """Create settings for anomaly detection job.
    Process type can be `DP`, `PY`, `DDCRP`, `STREAM_PY` or `POISSON_PY`"""
    ext = ".txt.gz" if gzip else ".txt"
    if os.name == "nt" and gzip:
        raise ValueError("gzip option not supported on Windows")
    settings = {}
    settings["input"] = {}
    settings["input"]["filepath"] = input_file
    settings["input"]["filename"] = ".".join(
        os.path.basename(input_file).split(".")[:-1])
    settings["input"]["root"] = os.path.dirname(input_file)
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
    settings["phase0"]["y_params_file"] = os.path.join(settings["output"]["root"], f"phase0_y_params{ext}")
    settings["phase0"]["x_y_params_file"] = os.path.join(settings["output"]["root"], f"phase0_x_y_params{ext}")
    settings["phase1"] = {}
    settings["phase1"]["tstart"] = test_interval[0]
    settings["phase1"]["tend"] = test_interval[1]
    settings["phase1"]["dest_file"] = os.path.join(settings["output"]["root"], f"phase1_dest_pval{ext}")
    settings["phase2"] = {}
    settings["phase2"]["dest_file"] = os.path.join(settings["output"]["root"], f"phase2_source_pval{ext}")
    settings["phase3"] = {}
    settings["phase3"]["dest_file"] = os.path.join(settings["output"]["root"], f"phase3_link_score{ext}")
    settings["phase4"] = {}
    settings["phase4"]["dest_file"] = os.path.join(settings["output"]["root"], f"phase_4_source_score{ext}")
    return settings

if __name__ == "__main__":
    from copy import deepcopy
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
    settings_train = create_settings("./data/dataset_002/auth.txt", "./data/dataset_002/PY/", NNODES, process_type="PY",
                                     param_est_interval=[0, 24 * 3600], test_interval=[24 * 3600, 1e8])
    settings_train_filename = "./data/dataset_002/PY/settings_train.json"
    with open(settings_train_filename, "w", encoding="utf-8") as file:
        json.dump(settings_train, file, indent=4)

    settings_test = deepcopy(settings_train)
    settings_test_filename = "./data/dataset_002/PY/settings_test.json"
    if USE_REAL_PARAMETERS:
        settings_test["phase0"]["y_params_file"] = os.path.join(
            os.path.dirname(settings_test["input"]["filepath"]), "phase0_y_params.txt")
        settings_test["phase0"]["x_y_params_file"] = os.path.join(
            os.path.dirname(settings_test["input"]["filepath"]), "phase0_x_y_params.txt")
    with open(settings_test_filename, "w", encoding="utf-8") as file:
        json.dump(settings_test, file, indent=4)

    preprocessing([f"{settings_train_filename}"])
    destination_process([f"{settings_test_filename}"])
    source_conditional_process([f"{settings_test_filename}"])
    link_scores([f"{settings_test_filename}"])
    source_scores([f"{settings_test_filename}"])

    ##################
    ### DIRICHLET ###
    ##################
    settings_dp = create_settings("./data/dataset_002/auth.txt", "./data/dataset_002/DP/", NNODES, process_type="DP",
                                  param_est_interval=[0, 24 * 3600], test_interval=[24 * 3600, 1e8])
    settings_filename_dp = "./data/dataset_002/DP/settings.json"
    with open(settings_filename_dp, "w", encoding="utf-8") as file:
        json.dump(settings_dp, file, indent=4)

    preprocessing([f"{settings_filename_dp}"])
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

    settings_ddcrp_test = deepcopy(settings_ddcrp)
    settings_filename_ddcrp_test = "./data/dataset_002/PY/settings_test.json"
    if USE_REAL_PARAMETERS:
        settings_ddcrp_test["phase0"]["y_params_file"] = os.path.join(
            os.path.dirname(settings_ddcrp_test["input"]["filepath"]), "phase0_y_params.txt")
        settings_ddcrp_test["phase0"]["x_y_params_file"] = os.path.join(
            os.path.dirname(settings_ddcrp_test["input"]["filepath"]), "phase0_x_y_params.txt")
    with open(settings_filename_ddcrp_test, "w", encoding="utf-8") as file:
        json.dump(settings_ddcrp_test, file, indent=4)

    preprocessing([f"{settings_filename_ddcrp}"])
    destination_process([f"{settings_filename_ddcrp_test}"])
    source_conditional_process([f"{settings_filename_ddcrp_test}"])
    link_scores([f"{settings_filename_ddcrp_test}"])
    source_scores([f"{settings_filename_ddcrp_test}"])

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
    destination_process([f"{settings_filename_pois_py}"])
    source_conditional_process([f"{settings_filename_pois_py}"])
    link_scores([f"{settings_filename_pois_py}"])
    source_scores([f"{settings_filename_pois_py}"])
