from datasets.common import PointCloudDataset
import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import Sampler
import time


class MiniMarketDataset(PointCloudDataset):
    def __init__(self, config, set='training', use_potentials=True, h5_path=None):

        super().__init__('MiniMarket')
        
        # Store config
        self.set = set
        self.use_potentials = use_potentials
        self.label_to_names = {0: 'background', 1: 'object'}
        self.init_labels()  # same as S3DIS (populates label_values, ignored_labels, num_classes)

        # Dataset metadata
        config.dataset_task = 'cloud_segmentation'

        # Set epoch_n based on set
        if self.set == 'training':
            self.epoch_n = config.epoch_steps * config.batch_num
        else:
            self.epoch_n = config.validation_size * config.batch_num


        self.config = config
        self.h5_path = Path(h5_path)

        # Load data
        with h5py.File(self.h5_path, 'r') as f:
            self.input_points = f['seg_points'][:]   # (N, P, 3)
            self.input_colors = f['seg_colors'][:]   # (N, P, 3)
            self.input_labels = f['seg_labels'][:]   # (N, P, 2)

        self.num_models = self.input_points.shape[0]
        self.epoch_n = self.num_models

        # For binary segmentation
        self.label_to_names = {0: 'background', 1: 'object'}
        self.ignored_labels = np.array([])  # if none
        self.num_classes = 2
        config.num_classes = self.num_classes
        config.dataset_task = 'segmentation'

    def __len__(self):
        return self.num_models

    def __getitem__(self, idx):
        points = self.input_points[idx].astype(np.float32)  # (640, 3)
        colors = self.input_colors[idx].astype(np.float32)  # (640, 3)
        labels_oh = self.input_labels[idx]                 # (640, 2)

        # Convert one-hot to class index (0 or 1)
        labels = np.argmin(labels_oh, axis=1).astype(np.int64)  # (640,)

        # Apply augmentation if needed (e.g., rotation, jitter)
        # points, colors = self.augmentation(points, colors)  # define if needed

        features = colors  # or np.ones_like(points[:, :1]) for constant input

        # KPConv expects: points, features, labels, stack_lengths
        stack_lengths = [points.shape[0]]
        input_list = self.segmentation_inputs(points, features, labels,stack_lengths)

        return input_list




class MiniMarketSampler(Sampler):
    """Simplified sampler for MiniMarket segmentation dataset."""

    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.N = dataset.epoch_n

    def __iter__(self):
        return iter(np.random.permutation(len(self.dataset))[:self.N])

    def __len__(self):
        return self.N
    
    def calibration(self, dataloader, verbose=False):
        print("MiniMarket: Dummy calibration (fixed batch_limit)")
        self.dataset.batch_limit = torch.tensor([self.dataset.config.batch_num * 640], dtype=torch.float32)
        self.dataset.batch_limit.share_memory_()
        self.dataset.neighborhood_limits = [32, 32, 32, 32, 32]  # Dummy values; override if needed



class MiniMarketCustomBatch:
    """Custom batch definition for MiniMarket segmentation dataset with KPConv"""

    def __init__(self, input_list):
        # Remove extra batch dimension
        input_list = input_list[0]

        # Number of KPConv layers (each has points, neighbors, pools, lengths)
        L = (len(input_list) - 5) // 4

        # Convert numpy arrays to tensors
        ind = 0
        self.points = [torch.from_numpy(arr) for arr in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(arr) for arr in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(arr) for arr in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(arr) for arr in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])  # shape (total_points,) with class index
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.model_inds = torch.from_numpy(input_list[ind])

    def pin_memory(self):
        self.points = [t.pin_memory() for t in self.points]
        self.neighbors = [t.pin_memory() for t in self.neighbors]
        self.pools = [t.pin_memory() for t in self.pools]
        self.lengths = [t.pin_memory() for t in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.model_inds = self.model_inds.pin_memory()
        return self

    def to(self, device):
        self.points = [t.to(device) for t in self.points]
        self.neighbors = [t.to(device) for t in self.neighbors]
        self.pools = [t.to(device) for t in self.pools]
        self.lengths = [t.to(device) for t in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.model_inds = self.model_inds.to(device)
        return self

    def unstack_points(self, layer=None):
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError(f"Unknown element name: {element_name}")

        all_outputs = []
        for layer_idx, elem_layer in enumerate(elements):
            if layer is not None and layer != layer_idx:
                continue

            i0 = 0
            split_list = []
            lengths = self.lengths[layer_idx + 1] if element_name == 'pools' else self.lengths[layer_idx]

            for b_i, length in enumerate(lengths):
                elem = elem_layer[i0:i0 + length]
                if element_name in ['neighbors', 'pools']:
                    mask = elem >= self.points[layer_idx].shape[0]
                    elem[mask] = -1
                    if element_name == 'neighbors':
                        elem[elem >= 0] -= i0
                    else:  # pooling
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_idx][:b_i])
                i0 += length
                split_list.append(elem.numpy() if to_numpy else elem)

            if layer is not None:
                return split_list
            all_outputs.append(split_list)

        return all_outputs


def MiniMarketCollate(batch_data):
    return MiniMarketCustomBatch(batch_data)