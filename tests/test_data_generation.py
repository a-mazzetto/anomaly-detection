"""Test data generation"""
import os
import pytest
from data_generation import constants
from data_generation.data_generation import generate_dataset
from utils import create_results_folder, get_baseline_folder, compare_datasets

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
