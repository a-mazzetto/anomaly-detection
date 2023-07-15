"""Generated datasets 002. Dataset where parameters are random and with lateral movement"""
# %%
import os
from data_generation import constants
from data_generation.data_generation import generate_dataset, generate_ddcrp_dataset, save_dataset
from data_generation.lateral_movement import generate_lateral_movement, LateralMovementType
from data_generation.dataset_operations import join_datasets_and_sort
# %%
SEED = 0
DDCRP = False
if DDCRP:
    FILE_NAME = "data/auth_ddcrp.txt"
else:
    FILE_NAME = "data/auth.txt"
DESTINATION_INTENSITY = 7.0
DESTINATION_DISCOUNT = 0.25

def random_source_intensities(gen, size):
    """Most computers (60%) see routinary activity"""
    return gen.choice(a=[2., 7., 12.], p=[0.6, 0.3, 0.1], size=size)

def random_source_discounts(gen, size):
    return gen.beta(a=2.0, b=5.0, size=size)

# %%
if DDCRP:
    dataset, params = generate_ddcrp_dataset(
        max_time=4 * constants.MAX_TIME,
        period=4 * constants.PERIOD,
        n_nodes=constants.NNODES,
        destination_intensity=DESTINATION_INTENSITY,
        destination_discount=DESTINATION_DISCOUNT,
        source_intensities=random_source_intensities,
        source_decay=43200,
        node_names=constants.computer_names,
        seed=SEED,
        discretize_time=False,
        file_name=None
    )
else:
    dataset, params = generate_dataset(
        max_time=4 * constants.MAX_TIME,
        period=4 * constants.PERIOD,
        n_nodes=constants.NNODES,
        destination_intensity=DESTINATION_INTENSITY,
        destination_discount=DESTINATION_DISCOUNT,
        source_intensities=random_source_intensities,
        source_discounts=random_source_discounts,
        node_names=constants.computer_names,
        seed=SEED,
        discretize_time=False,
        file_name=None)

# Create a lateral movement starting at 3/4 of maximum time, after observing the graph for
# one hour and that propagates for maximum 3 hours
min_time = float(dataset[0][0])
max_time = float(dataset[-1][0])
lm_time = 3 * (max_time - min_time) / 4
learning_period = [lm_time - 3600, lm_time]
movement_period = [lm_time, lm_time + 3 * 3600]
edge_list = [(i[1], i[2]) for i in dataset if i[0] >= learning_period[0] and\
             i[0] <= learning_period[1]]
lm_dataset = generate_lateral_movement(
    edge_list=edge_list,
    time_interval=movement_period,
    rate=10.0,
    typology=LateralMovementType(rw_steps=2, rw_num=20, rw_reset=False),
    target_type="low-traffic",
    gen=SEED)

full_dataset = join_datasets_and_sort(dataset, lm_dataset)
save_dataset(full_dataset, file_name=FILE_NAME)

# Save also process parameters
root = os.path.dirname(FILE_NAME)
with open(os.path.join(root, "phase0_y_params.txt"), "w", encoding="utf-8") as _f:
    _f.write("\t".join([str(DESTINATION_INTENSITY), str(DESTINATION_DISCOUNT)]) + "\n")
with open(os.path.join(root, "phase0_x_y_params.txt"), "w", encoding="utf-8") as _f:
    for i in params:
        _f.write("\t".join([str(j) for j in i]) + "\n")

# %%
