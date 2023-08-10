"""Generate dataset for testing of VGRNN"""
# %% Imports
import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch
from data_generation import constants
from data_generation.data_generation import generate_dataset
from data_generation.dataset_operations import simple_aggregate_dataset_into_graphs

def create_training_dataset(save_path, aggregation_interval=15 * 60, dyn_steps=4):

    def random_source_intensities(gen, size):
        """Most computers (60%) see routinary activity"""
        return gen.choice(a=[2., 7., 12.], p=[0.6, 0.3, 0.1], size=size)

    print("Creating data for training...")
    dynamic_graph = []

    for _ in range(5):
        dataset, _ = generate_dataset(
            max_time=2 * constants.MAX_TIME,
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
        agg_dataset = simple_aggregate_dataset_into_graphs(
            dataset, aggregation_interval, rename=False)[0]

        dynamic_graph.extend([agg_dataset[i:i+dyn_steps] for \
                              i in range(len(agg_dataset)- (dyn_steps - 1))])

    np.save(os.path.join(save_path, "training_dynamic_graphs.npy"), np.array(dynamic_graph, dtype=object), allow_pickle=True)

class VGRNNTestDataset(InMemoryDataset):
    def __init__(self, root="./data/VGRNNTestDataset", transform=None, pre_transform=None, pre_filter=None, use_features=True):
        """
        Creates a dataset which is batches of Batched dynamic graphs. To obtain the adjacency matrix use
        to_dense_adj(batch.edge_index, batch=batch.batch)

        Custom parameters:
        :param bool features: use features
        """
        self.use_features = use_features
        super().__init__(root, transform, pre_transform, pre_filter)
        data = torch.load(self.processed_paths[0])
        batch_number = np.loadtxt(self.processed_paths[1])
        batches = []
        for i in np.unique(batch_number):
            batch_match = np.where(batch_number == i)[0]
            batches.append(Batch.from_data_list(data[batch_match[0]:(batch_match[-1] + 1)]))
        self.data, self.slices = self.collate(batches)

    @property
    def raw_file_names(self):
        return ['training_dynamic_graphs.npy']

    @property
    def processed_file_names(self):
        if self.use_features:
            return ['vgrnn_training_dataset_feat.pt', 'batch_numper_feat.txt']
        else:
            return ['vgrnn_training_dataset.pt', 'batch_number.txt']

    def download(self):
        """Create raw dataset and save to `raw_dir`"""
        print("Creating dataset for the first time requires several minutes")
        create_training_dataset(self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        data_groups = np.load(self.raw_paths[0], allow_pickle=True)

        if self.pre_filter is not None:
            raise NotImplementedError("No pre-filtering method")

        if self.pre_transform is not None:
            raise NotImplementedError("No pre-transform method")

        processed_dyn_graph = []
        batch_number = np.ndarray(shape=(0,))
        for n_batch, dyn_graph in enumerate(data_groups):
            for static_graph_links in dyn_graph:
                # Guarantee same dimension for all
                max_nodes = constants.NNODES
                outdegree = np.ndarray(shape=(0,))
                indegree = np.ndarray(shape=(0,))
                if self.use_features:
                    for i in range(max_nodes):
                        outdegree = np.append(outdegree, np.sum(static_graph_links[:, 0] == i))
                        indegree = np.append(indegree, np.sum(static_graph_links[:, 1] == i))
                        features = torch.tensor(np.vstack((indegree, outdegree)).T, dtype=torch.float)
                else:
                    features = torch.ones((max_nodes, 1))
                links_matrix = torch.tensor(static_graph_links, dtype=torch.int64).t().contiguous()
                geom_data = Data(x=features, edge_index=links_matrix)
                processed_dyn_graph.append(geom_data)
                batch_number = np.append(batch_number, n_batch)

        torch.save(processed_dyn_graph, self.processed_paths[0])
        np.savetxt(self.processed_paths[1], batch_number.astype(int))

if __name__ == "__main__":

    dataset = VGRNNTestDataset(use_features=False)

# %%
