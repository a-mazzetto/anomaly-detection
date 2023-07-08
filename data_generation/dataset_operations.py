"""Operations on datasets"""
import numpy as np

def join_datasets_and_sort(dataset_0, dataset_1):
    """Assuming the first column is time"""
    dataset = dataset_0 + dataset_1
    times = np.array([i[0] for i in dataset]).astype(float)
    reordering = np.argsort(times)
    return [dataset[i] for i in reordering]
