"""Check script"""
# %%
import json
from data_generation.constants import NNODES
from anomaly_detection.step0_preprocessing import preprocessing
from anomaly_detection.step1_destination_p_value import destination_process
from anomaly_detection.step2_source_p_value_py import source_conditional_process
from anomaly_detection.step3_link_scores import link_scores
from anomaly_detection.step4_source_scores import source_scores
from launch import create_settings

settings = create_settings("./data/dataset_001/dataset_001.txt", "./data/dataset_001", NNODES, process_type="PY")
settings_file = "./data/dataset_001/settings.json"
with open(settings_file, "w", encoding="utf-8") as file:
    json.dump(settings, file, indent=4)

# No pre-processing step, the parameters are provided
destination_process([f"{settings_file}"])
source_conditional_process([f"{settings_file}"])
link_scores([f"{settings_file}"])
source_scores([f"{settings_file}"])

# %%
