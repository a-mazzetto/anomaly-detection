"""Generated datasets 002. Dataset where parameters are random and with lateral movement"""
from data_generation import constants
from data_generation.data_generation import generate_dataset, generate_ddcrp_dataset, save_dataset
from data_generation.lateral_movement import generate_lateral_movement, LateralMovementType
from data_generation.dataset_operations import join_datasets_and_sort

SEED = 0
DDCRP = True
if DDCRP:
    FILE_NAME = "data/auth_ddcrp.txt"
else:
    FILE_NAME = "data/auth.txt"

def random_source_intensities(gen, size):
    """Most computers (60%) see routinary activity"""
    return gen.choice(a=[2., 7., 12.], p=[0.6, 0.3, 0.1], size=size)

def random_source_discounts(gen, size):
    return gen.beta(a=2.0, b=5.0, size=size)

if DDCRP:
    dataset, _ = generate_ddcrp_dataset(
        max_time=4 * constants.MAX_TIME,
        period=constants.PERIOD,
        n_nodes=constants.NNODES,
        destination_intensity=7.0,
        destination_discount=0.25,
        source_intensities=random_source_intensities,
        source_decay=43200,
        node_names=constants.computer_names,
        seed=SEED,
        discretize_time=False,
        file_name=None
    )
else:
    dataset, _, _ = generate_dataset(
        max_time=4 * constants.MAX_TIME,
        period=constants.PERIOD,
        n_nodes=constants.NNODES,
        destination_intensity=7.0,
        destination_discount=0.25,
        source_intensities=random_source_intensities,
        source_discounts=random_source_discounts,
        node_names=constants.computer_names,
        seed=SEED,
        discretize_time=False,
        file_name=None)

# Create a lateral movement starting at mid-time, after observing the graph for
# one hour and that propagates for maximum one hour
mid_time = (float(dataset[-1][0]) - float(dataset[0][0])) / 2
learning_period = [mid_time - 3600, mid_time]
movement_period = [mid_time, mid_time + 3600]
edge_list = [(i[1], i[2]) for i in dataset if i[0] >= learning_period[0] and\
             i[0] <= learning_period[1]]
lm_dataset = generate_lateral_movement(
    edge_list=edge_list,
    time_interval=movement_period,
    rate=10.0,
    typology=LateralMovementType(rw_steps=10, rw_num=10, rw_reset=False),
    target_type="low-traffic",
    gen=SEED)

full_dataset = join_datasets_and_sort(dataset, lm_dataset)
save_dataset(full_dataset, file_name=FILE_NAME)
