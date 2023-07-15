"""Test data generation"""
import os
import json
import pytest
from anomaly_detection.launch import create_settings
from utils import create_results_folder, get_baseline_folder, compare_source_score_files
from anomaly_detection.step0_preprocessing import preprocessing
from anomaly_detection.step1_destination_p_value import destination_process
from anomaly_detection.step2_source_p_value_py import source_conditional_process
from anomaly_detection.step3_link_scores import link_scores
from anomaly_detection.step4_source_scores import source_scores

SEED = 0

@pytest.mark.parametrize("test_name, reference_file, process_type",
                         [
                            ("anomaly_detection_py", "phase4.txt", "PY"),
                            ("anomaly_detection_ddcrp", "phase4.txt", "DDCRP"),
                         ]
)
def test_anomaly_detection_py(test_name, reference_file, process_type):
    """Function to test data generation"""
    settings = create_settings(
        input_file=os.path.join(get_baseline_folder(test_name), "auth.txt"),
        output_folder=create_results_folder(test_name),
        n_nodes=100,
        process_type=process_type,
        stream_time_window=1 * 60 * 60
    )
    settings_filename = os.path.join(create_results_folder(test_name), "settings.json")
    with open(settings_filename, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)

    preprocessing([f"{settings_filename}"])
    destination_process([f"{settings_filename}"])
    source_conditional_process([f"{settings_filename}"])
    link_scores([f"{settings_filename}"])
    source_scores([f"{settings_filename}"])

    reference_file = os.path.join(get_baseline_folder(test_name), reference_file)
    compare_source_score_files(settings["phase4"]["dest_file"], reference_file)
