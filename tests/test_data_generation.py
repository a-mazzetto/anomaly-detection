"""Test data generation"""
import os
import pytest
from data_generation import constants
from data_generation.data_generation import generate_dataset, generate_ddcrp_dataset, save_dataset
from data_generation.dataset_operations import join_datasets_and_sort
from utils import create_results_folder, get_baseline_folder, compare_datasets, load_dataset
from data_generation.lateral_movement import generate_lateral_movement, LateralMovementType

SEED = 0

def random_source_intensities(gen, size):
    return gen.choice(a=[2., 7., 12.], p=[0.2, 0.3, 0.5], size=size)

def random_source_discounts(gen, size):
    return gen.beta(a=2.0, b=5.0, size=size)

@pytest.mark.parametrize("test_name, source_intensities, source_discounts",
                         [
                             ("data_gen_constant_source_params", 2.0, 0.5),
                             ("data_gen_random_source_params",
                              random_source_intensities,
                              random_source_discounts)
                         ]
)
def test_data_generation(test_name, source_intensities, source_discounts):
    """Function to test data generation"""
    file_name = f'{test_name}.txt'
    baseline_file = os.path.join(get_baseline_folder(test_name), file_name)
    results_file = os.path.join(create_results_folder(test_name), file_name)
    _ = generate_dataset(
        max_time=constants.MAX_TIME,
        period=constants.PERIOD,
        n_nodes=constants.NNODES,
        destination_intensity=7.0,
        destination_discount=0.25,
        source_intensities=source_intensities,
        source_discounts=source_discounts,
        node_names=constants.computer_names,
        seed=SEED,
        discretize_time=False,
        file_name=results_file)
    compare_datasets(baseline_file, results_file)

@pytest.mark.parametrize("test_name, source_intensities",
                         [
                             ("data_gen_ddcrp_const_source_params", 2.0),
                             ("data_gen_ddcrp_random_source_params",
                              random_source_intensities)
                         ]
)
def test_ddcrp_data_generation(test_name, source_intensities):
    """Function to test data generation"""
    file_name = f'{test_name}.txt'
    baseline_file = os.path.join(get_baseline_folder(test_name), file_name)
    results_file = os.path.join(create_results_folder(test_name), file_name)
    _ = generate_ddcrp_dataset(
        max_time=constants.MAX_TIME,
        period=constants.PERIOD,
        n_nodes=constants.NNODES,
        destination_intensity=7.0,
        destination_discount=0.25,
        source_intensities=source_intensities,
        source_decay=constants.PERIOD,
        node_names=constants.computer_names,
        seed=SEED,
        discretize_time=False,
        file_name=results_file)
    compare_datasets(baseline_file, results_file)

@pytest.mark.parametrize("test_name, lm_type",
                         [
                             ("data_gen_random_walk_lateral", "random_walk"),
                             ("data_gen_longhest_path_lateral", "longhest_path")
                         ]
)
def test_lateral_movement_generation(test_name, lm_type):
    """Function to test lateral movement generation"""
    file_name = f'{test_name}.txt'
    baseline_file = os.path.join(get_baseline_folder(test_name), file_name)
    results_file = os.path.join(create_results_folder(test_name), file_name)
    # Load dataset on whcih to superimpose lateral movement
    normal_dataset = load_dataset(os.path.join(get_baseline_folder(test_name), "base_dataset.txt"))
    max_time = float(normal_dataset[-1][0])
    mid_point = len(normal_dataset) // 2
    mid_time = float(normal_dataset[mid_point][0])
    graph_history_start = mid_point - min(1000, len(normal_dataset[:mid_point]))
    edge_list = [(i[1], i[2]) for i in normal_dataset[graph_history_start:mid_point]]
    if lm_type == "longhest_path":
        lm_dataset = generate_lateral_movement(
            edge_list=edge_list,
            time_interval=[mid_time, min(mid_time + 10, max_time)],
            rate=10.0,
            typology=LateralMovementType(n_attempts=20),
            target_type="low-traffic",
            gen=0)
    elif lm_type == "random_walk":
        lm_dataset = generate_lateral_movement(
            edge_list=edge_list,
            time_interval=[mid_time, min(mid_time + 10, max_time)],
            rate=10.0,
            typology=LateralMovementType(rw_steps=5, rw_num=5, rw_reset=False),
            target_type="low-traffic",
            gen=0)
    else:
        NotImplementedError("Option not available")
    full_dataset = join_datasets_and_sort(normal_dataset, lm_dataset)
    save_dataset(full_dataset, file_name=results_file)
    compare_datasets(baseline_file, results_file)
