"""Testing Utilities"""
import os
import numpy as np
import pandas as pd

def get_baseline_folder(folder_name):
    return os.path.join(os.getcwd(), 'tests', 'baseline', folder_name)

def create_results_folder(folder_name):
    root_folder = os.path.join(os.getcwd(), 'tests', 'results')
    if not os.path.exists(root_folder):
        os.makedirs(root_folder, exist_ok=False)
    folder_path = os.path.join(root_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def compare_datasets(baseline_file, results_file):
    """Function to compare resulting datasets"""
    baseline = pd.read_table(baseline_file, names=['time', 'source', 'destination', 'anomaly'], sep='\t')
    baseline.time = baseline.time.astype(float)
    baseline.source = baseline.source.str.strip()
    baseline.destination = baseline.destination.str.strip()
    baseline.anomaly = baseline.anomaly.astype(int)
    result = pd.read_table(results_file, names=['time', 'source', 'destination', 'anomaly'], sep='\t')
    result.time = result.time.astype(float)
    result.source = result.source.str.strip()
    result.destination = result.destination.str.strip()
    result.anomaly = baseline.anomaly.astype(int)
    assert np.allclose(baseline.time, result.time), 'Mismatch with respect to baseline in the time column'
    assert all(baseline.destination == result.destination), "Mismatch in destination sequence"
    assert all(baseline.source == result.source), "Mismatch in source sequence"
    assert all(baseline.anomaly == result.anomaly), "Mismatch in anomaly flag"

def load_dataset(file):
    """Load dataset"""
    dataset = pd.read_table(file, names=['time', 'source', 'destination', 'anomaly'], sep='\t')
    dataset.time = dataset.time.astype(float)
    dataset.source = dataset.source.str.strip()
    dataset.destination = dataset.destination.str.strip()
    dataset.anomaly = dataset.anomaly.astype(int)
    return [(row.time, row.source, row.destination, row.anomaly) for _, row in dataset.iterrows()]

def compare_source_score_files(test, ref):
    """Compare text files"""
    test_df = pd.read_table(test, index_col=0, names=["score"])
    test_df.score = test_df.score.astype(float)
    base_df = pd.read_table(ref, index_col=0, names=["score"])
    base_df.score = base_df.score.astype(float)
    assert np.all(test_df.index == base_df.index), "Index of output different from reference"
    assert np.testing.assert_allclose(test_df.score, base_df.score), "Scores different from baseline"
