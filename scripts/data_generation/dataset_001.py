"""Generated datasets 001. Most simple dataset where all parameters are constant"""
from data_generation import constants
from data_generation.data_generation import generate_dataset

FILE_NAME = "./data/dataset_001/dataset_001.txt"
SEED = 0

_, params = generate_dataset(
    max_time=4 * constants.MAX_TIME,
    period=4 * constants.PERIOD,
    n_nodes=constants.NNODES,
    destination_intensity=7.0,
    destination_discount=0.25,
    source_intensities=2.0,
    source_discounts=0.5,
    node_names=constants.computer_names,
    seed=SEED,
    discretize_time=True,
    file_name=FILE_NAME)

with open("./data/dataset_001/phase0_y_params.txt", "w", encoding="utf-8") as _f:
    _f.write("\t".join([str(7.0), str(0.25)]) + "\n")
with open("./data/dataset_001/phase0_x_y_params.txt", "w", encoding="utf-8") as _f:
    for i in params:
        _f.write("\t".join([str(j) for j in i]) + "\n")
