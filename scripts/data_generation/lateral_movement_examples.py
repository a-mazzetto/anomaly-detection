"""Examples of lateral movement application"""

# %% Generate data
from data_generation.data_generation import generate_dataset

NNODES  = 100
computer_names = [f'C{i}' for i in range(NNODES)]
# For example, simulate 12h
MAX_TIME = 12 * 60 * 60
# For example, 3h period
PERIOD = 3 * 60 * 60

def source_intensities(gen, size):
    return gen.choice(a=[2., 7., 12.], p=[0.2, 0.3, 0.5], size=size)

dataset = generate_dataset(
    max_time=MAX_TIME,
    period=PERIOD,
    n_nodes=NNODES,
    destination_intensity=7.0,
    destination_discount=0.25,
    source_intensities=source_intensities,
    source_discounts=0.5,
    node_names=computer_names,
    seed=0,
    discretize_time=True,
    file_name=None)
dataset = dataset[0]
edge_list = [(i[1], i[2]) for i in dataset]

# %% Generate Lateral Movement
from data_generation.lateral_movement import generate_lateral_movement, LateralMovementType

# Improperly using the whole dataset to learn the graph
lm_dataset = generate_lateral_movement(
        edge_list=edge_list,
        time_interval=[20, 60],
        rate=10.0,
        typology=LateralMovementType(n_attempts=20),
        target_type="low-traffic",
        gen=0)

# Improperly using the whole dataset to learn the graph
lm_dataset_1 = generate_lateral_movement(
        edge_list=edge_list,
        time_interval=[20, 60],
        rate=10.0,
        typology=LateralMovementType(rw_steps=5, rw_num=5, rw_reset=False),
        target_type="low-traffic",
        gen=0)

# %% Join dataset
from data_generation.dataset_operations import join_datasets_and_sort

join_datasets_and_sort(dataset, lm_dataset_1)
# %%
