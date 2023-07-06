"""Generated datasets 001. Most simple dataset where all parameters are constant"""
from data_generation import constants
from data_generation.data_generation import generate_dataset

FILE_NAME = "data/dataset_001.txt"
SEED = 0

_ = generate_dataset(
    max_time=constants.MAX_TIME,
    period=constants.PERIOD,
    n_nodes=constants.NNODES,
    destination_intensity=7.0,
    destination_discount=0.25,
    source_intensities=2.0,
    source_discounts=0.5,
    node_names=constants.computer_names,
    seed=SEED,
    dicretize_time=True,
    file_name=FILE_NAME)
