"""Generate dataset for test of GNN convolution layers"""
# %% Imports
import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
import matplotlib.pyplot as plt
from data_generation import constants
from data_generation.data_generation import generate_dataset
from data_generation.dataset_operations import simple_aggregate_dataset_into_graphs

def create_dataset(save_path, aggregation_interval=15 * 60):

    def random_source_intensities(gen, size):
        """Most computers (60%) see routinary activity"""
        return gen.choice(a=[2., 7., 12.], p=[0.6, 0.3, 0.1], size=size)

    print("Creating data...")
    first_group_data = []

    for i in range(100):
        print(f"Generating {i}th generation")
        dataset, _ = generate_dataset(
            max_time=18 * constants.MAX_TIME,
            period=2 * constants.PERIOD,
            n_nodes=constants.NNODES,
            destination_intensity=7.0,
            destination_discount=0.25,
            source_intensities=random_source_intensities,
            source_discounts=0.25,
            node_names=constants.NNODES,
            seed=None,
            discretize_time=False,
            file_name=None)

        first_group_data.extend(simple_aggregate_dataset_into_graphs(dataset, aggregation_interval)[0])

    np.save(os.path.join(save_path, "first_group_data.npy"), np.array(first_group_data, dtype=object), allow_pickle=True)

class OneGraphFamilyDataset(InMemoryDataset):
    def __init__(self, root="./data/OneGraphFamilyDataset", transform=None, pre_transform=None, pre_filter=None, use_features=True):
        """
        Custom parameters:
        :param bool features: use features
        """
        self.use_features = use_features
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['first_group_data.npy']

    @property
    def processed_file_names(self):
        if self.use_features:
            return ['one_graph_family_dataset_feat.pt']
        else:
            return ['one_graph_family_dataset.pt']

    def download(self):
        """Create raw dataset and save to `raw_dir`"""
        print("Creating dataset for the first time requires several minutes")
        create_dataset(self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        data_group = np.load(self.raw_paths[0], allow_pickle=True)

        if self.pre_filter is not None:
            raise NotImplementedError("No pre-filtering method")

        if self.pre_transform is not None:
            raise NotImplementedError("No pre-transform method")

        data_list = []
        for links in data_group:
            outdegree = np.ndarray(shape=(0,))
            indegree = np.ndarray(shape=(0,))
            if self.use_features:
                for i in range(links.max() + 1):
                    outdegree = np.append(outdegree, np.sum(links[:, 0] == i))
                    indegree = np.append(indegree, np.sum(links[:, 1] == i))
                    features = torch.tensor(np.vstack((indegree, outdegree)).T, dtype=torch.float)
            else:
                features = torch.ones((links.max() + 1, 1))
            links_matrix = torch.tensor(links, dtype=torch.int64).t().contiguous()
            geom_data = Data(x=features, edge_index=links_matrix)
            assert geom_data.validate(), "Issues with Data creation"
            data_list.append(geom_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":

    dataset = OneGraphFamilyDataset(use_features=True)
