#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling S3DIS dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import numpy as np
import pickle
import torch
import math
import warnings
from multiprocessing import Lock


# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *

from datasets.common import grid_subsampling
from utils.config import bcolors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class S3DISDataset(PointCloudDataset):
    """
    PyTorch-compatible dataset class for handling the Stanford Large-Scale 3D Indoor Spaces (S3DIS) dataset.

    This dataset class is designed for cloud segmentation tasks. It loads and preprocesses the full point clouds
    into memory, supports potential-based and random sampling strategies, and is compatible with multiprocessing
    during training.

    Attributes:
        label_to_names (dict): Maps class indices to semantic labels.
        ignored_labels (np.array): Labels to be ignored during training (can be empty).
        path (str): Root path to the dataset on disk.
        dataset_task (str): The type of task, e.g., 'cloud_segmentation'.
        set (str): One of ['training', 'validation', 'test', 'ERF'].
        use_potentials (bool): Whether to use potential-based sampling.
        epoch_n (int): Number of models used per epoch.
        files (List[str]): List of .ply files corresponding to the chosen split.
        input_trees, input_colors, input_labels: Data containers for point cloud information.
        pot_trees: KD-Trees used for potential sampling.
        potentials, min_potentials, argmin_potentials: Structures for potential-based sampling.
        batch_limit (torch.Tensor): Max number of points per batch.
        epoch_inds, epoch_i: Epoch sampling management structures.
        worker_lock (Lock): Lock object for controlling parallel sampling.
    """
    def __init__(self, config, set='training', use_potentials=True, load_data=True):
        """
        Initialize the S3DIS dataset object.

        Args:
            config (object): Configuration object containing dataset parameters.
            set (str): One of ['training', 'validation', 'test', 'ERF'] determining the dataset split.
            use_potentials (bool): Whether to use potentials for sampling strategy.
            load_data (bool): If False, skips loading the point cloud data.
        """
        PointCloudDataset.__init__(self, 'S3DIS')

        ############
        # Parameters
        ############

        # Dict from labels to names
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'chair',
                               8: 'table',
                               9: 'bookcase',
                               10: 'sofa',
                               11: 'board',
                               12: 'clutter'}

        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # Dataset folder
        self.path = 'D:\Datasets\Stanford3dDataset_v1.2'

        # Type of task conducted on this dataset
        self.dataset_task = 'cloud_segmentation'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes - len(self.ignored_labels)
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        # Path of the training files
        self.train_path = 'original_ply'

        # List of files to process
        ply_path = join(self.path, self.train_path)

        # Proportion of validation scenes
        self.cloud_names = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
        self.all_splits = [0, 1, 2, 3, 4, 5]
        self.validation_split = 4

        # Number of models used per epoch
        if self.set == 'training':
            self.epoch_n = config.epoch_steps * config.batch_num
        elif self.set in ['validation', 'test', 'ERF']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for S3DIS data: ', self.set)

        # Stop data is not needed
        if not load_data:
            return

        ###################
        # Prepare ply files
        ###################

        self.prepare_S3DIS_ply()

        ################
        # Load ply files
        ################
        
        # List of training files
        self.files = []
        for i, f in enumerate(self.cloud_names):
            if self.set == 'training':
                if self.all_splits[i] != self.validation_split:
                    self.files += [join(ply_path, f + '.ply')]
            elif self.set in ['validation', 'test', 'ERF']:
                if self.all_splits[i] == self.validation_split:
                    self.files += [join(ply_path, f + '.ply')]
            else:
                raise ValueError('Unknown set for S3DIS data: ', self.set)

        if self.set == 'training':
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if self.all_splits[i] != self.validation_split]
        elif self.set in ['validation', 'test', 'ERF']:
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)
                                if self.all_splits[i] == self.validation_split]

        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')


        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []

        # Start loading
        self.load_subsampled_clouds()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(np.array(self.argmin_potentials, dtype=np.int64))
            self.min_potentials = torch.from_numpy(np.array(self.min_potentials, dtype=np.float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            self.epoch_inds = torch.from_numpy(np.zeros((2, self.epoch_n), dtype=np.int64))
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = Lock()

        # For ERF visualization, we want only one cloud per batch and no randomness
        if self.set == 'ERF':
            self.batch_limit = torch.tensor([1], dtype=torch.float32)
            self.batch_limit.share_memory_()
            np.random.seed(42)

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.cloud_names)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        if self.use_potentials:
            return self.potential_item(batch_i)
        else:
            return self.random_item(batch_i)

    def potential_item(self, batch_i, debug_workers=False):
        """
        Generate one training or validation batch based on point potentials for S3DIS segmentation.

        This method selects input spheres (regions) from the dataset by choosing center points with the
        lowest accumulated sampling 'potential'. It uses a spatial KD-tree to retrieve neighboring points
        within a radius of each center. Potentials are updated after each sampling to ensure spatial 
        coverage and avoid oversampling dense areas. The process continues until the batch point count
        exceeds a configured threshold.

        Parameters
        ----------
        batch_i : int
            Index of the batch (not directly used but can be used for tracking or debugging).
        
        debug_workers : bool, optional
            If True, prints visual indicators of worker states for debugging multiprocessing behavior.
            (Default: False)

        Returns
        -------
        tuple
            A tuple containing the following lists, each representing part of the batch:
            - p_list : list of np.ndarray
                Centered input points (shape [N_i, 3]) for each input region.
            - f_list : list of np.ndarray
                Features of each point (e.g., RGB + height) (shape [N_i, F]).
            - l_list : list of np.ndarray
                Ground truth labels per point in each region.
            - pi_list : list of np.ndarray
                Indices of selected points in the original point cloud.
            - i_list : list of int
                Indices of center points used to define each region.
            - ci_list : list of int
                Cloud indices from which each region was sampled.
            - s_list : list of float
                Scaling factors applied during augmentation for each region.
            - R_list : list of np.ndarray
                Rotation matrices applied during augmentation.

        Raises
        ------
        ValueError
            If repeated attempts (100×batch size) to sample non-empty spheres fail, suggesting that the
            point cloud may be too sparse or incorrectly processed.
        """
        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0
        failed_attempts = 0

        info = get_worker_info()
        if info is not None:
            wid = info.id
        else:
            wid = None

        while True:

            t += [time.time()]

            if debug_workers:
                message = ''
                for wi in range(info.num_workers):
                    if wi == wid:
                        message += ' {:}X{:} '.format(bcolors.FAIL, bcolors.ENDC)
                    elif self.worker_waiting[wi] == 0:
                        message += '   '
                    elif self.worker_waiting[wi] == 1:
                        message += ' | '
                    elif self.worker_waiting[wi] == 2:
                        message += ' o '
                print(message)
                self.worker_waiting[wid] = 0

            with self.worker_lock:

                if debug_workers:
                    message = ''
                    for wi in range(info.num_workers):
                        if wi == wid:
                            message += ' {:}v{:} '.format(bcolors.OKGREEN, bcolors.ENDC)
                        elif self.worker_waiting[wi] == 0:
                            message += '   '
                        elif self.worker_waiting[wi] == 1:
                            message += ' | '
                        elif self.worker_waiting[wi] == 2:
                            message += ' o '
                    print(message)
                    self.worker_waiting[wid] = 1

                # Get potential minimum
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])

                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)

                # Center point of input region
                center_point = np.copy(pot_points[point_ind, :].reshape(1, -1))

                # Add a small noise to center point
                if self.set != 'ERF':
                    center_point += np.clip(np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape),
                                            -self.config.in_radius / 2,
                                            self.config.in_radius / 2)

                # Indices of points in input region
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)

                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]

                # Update potentials (Tukey weights)
                if self.set != 'ERF':
                    tukeys = np.square(1 - d2s / np.square(self.config.in_radius))
                    tukeys[d2s > np.square(self.config.in_radius)] = 0
                    self.potentials[cloud_ind][pot_inds] += tukeys
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)


            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            t += [time.time()]

            # Number collected
            n = input_inds.shape[0]

            # Safe check for empty spheres
            if n < 2:
                failed_attempts += 1
                if failed_attempts > 100 * self.config.batch_num:
                    raise ValueError('It seems this dataset only containes empty input spheres')
                t += [time.time()]
                t += [time.time()]
                continue

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            t += [time.time()]

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if n > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    n = input_inds.shape[0]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        if debug_workers:
            message = ''
            for wi in range(info.num_workers):
                if wi == wid:
                    message += ' {:}0{:} '.format(bcolors.OKBLUE, bcolors.ENDC)
                elif self.worker_waiting[wi] == 0:
                    message += '   '
                elif self.worker_waiting[wi] == 1:
                    message += ' | '
                elif self.worker_waiting[wi] == 2:
                    message += ' o '
            print(message)
            self.worker_waiting[wid] = 2

        t += [time.time()]

        # Display timings
        debugT = False
        if debugT:
            print('\n************************\n')
            print('Timings:')
            ti = 0
            N = 5
            mess = 'Init ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Pots ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Sphere .... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Collect ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Augment ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += N * (len(stack_lengths) - 1) + 1
            print('concat .... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('input ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('stack ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('\n************************\n')
        return input_list

    def random_item(self, batch_i):
        """
        Generate one training or validation batch using predefined random sampling indices.

        This method retrieves input spheres (regions of points) based on pre-shuffled epoch indices.
        For each center point, it gathers neighboring points within a given radius using a KD-tree,
        applies optional data and color augmentations, and assembles them into a batch. Sampling 
        continues until the number of collected points exceeds the configured batch limit.

        Parameters
        ----------
        batch_i : int
            Index of the batch (not directly used but may serve for debugging or tracking).

        Returns
        -------
        tuple
            A tuple containing the following lists, each representing a component of the batch:
            - p_list : list of np.ndarray
                Centered input points for each region (shape [N_i, 3]).
            - f_list : list of np.ndarray
                Input features for each point (e.g., color + height) (shape [N_i, F]).
            - l_list : list of np.ndarray
                Ground truth labels for each point in the region.
            - pi_list : list of np.ndarray
                Indices of selected points in the original point cloud.
            - i_list : list of int
                Indices of the selected center points in the point clouds.
            - ci_list : list of int
                Indices of the clouds from which regions are sampled.
            - s_list : list of float
                Scale factors applied during data augmentation.
            - R_list : list of np.ndarray
                Rotation matrices applied during data augmentation.

        Raises
        ------
        ValueError
            If repeated attempts (100×batch size) fail to sample non-empty regions, possibly due to 
            sparsity or preprocessing issues in the dataset.
        """
        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0
        failed_attempts = 0

        while True:

            with self.worker_lock:

                # Get potential minimum
                cloud_ind = int(self.epoch_inds[0, self.epoch_i])
                point_ind = int(self.epoch_inds[1, self.epoch_i])

                # Update epoch indice
                self.epoch_i += 1
                if self.epoch_i >= int(self.epoch_inds.shape[1]):
                    self.epoch_i -= int(self.epoch_inds.shape[1])
                

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Center point of input region
            center_point = np.copy(points[point_ind, :].reshape(1, -1))

            # Add a small noise to center point
            if self.set != 'ERF':
                center_point += np.clip(np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape),
                                        -self.config.in_radius / 2,
                                        self.config.in_radius / 2)

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            # Number collected
            n = input_inds.shape[0]
            
            # Safe check for empty spheres
            if n < 2:
                failed_attempts += 1
                if failed_attempts > 100 * self.config.batch_num:
                    raise ValueError('It seems this dataset only containes empty input spheres')
                continue

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if n > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    n = input_inds.shape[0]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        return input_list

    def prepare_S3DIS_ply(self):
        """
        Aggregates and converts raw room-level annotated point cloud data from the S3DIS dataset into unified PLY files per area.

        This method processes each area (e.g., 'Area_1', 'Area_2', ...) by:
        1. Iterating through its subfolders (rooms).
        2. Reading all `.txt` annotation files that contain XYZRGB and class label information.
        3. Mapping each object to a semantic class using `self.name_to_label`.
        4. Stacking all room-level points, colors, and class labels into single arrays.
        5. Writing the aggregated data into a `.ply` file with fields ['x', 'y', 'z', 'red', 'green', 'blue', 'class'].

        The generated `.ply` files are saved to:
            {self.path}/{self.train_path}/{Area_X}.ply

        This is a one-time preprocessing step and is skipped for any area where the `.ply` file already exists.

        Notes:
            - If the raw `.txt` files contain malformed or non-ASCII characters (a known S3DIS issue), the method attempts to fix them automatically with regex.
            - 'stairs' is mapped to the 'clutter' class to ensure compatibility with the label schema.

        Raises:
            ValueError: If an object label is unknown and cannot be mapped to a valid class.
            Warning: If non-standard characters are found in the `.txt` files and require correction.
        """
        print('\nPreparing ply files')
        t0 = time.time()

        # Folder for the ply files
        ply_path = join(self.path, self.train_path)
        if not exists(ply_path):
            makedirs(ply_path)

        for cloud_name in self.cloud_names:

            # Pass if the cloud has already been computed
            cloud_file = join(ply_path, cloud_name + '.ply')
            if exists(cloud_file):
                continue

            # Get rooms of the current cloud
            cloud_folder = join(self.path, cloud_name)
            room_folders = [join(cloud_folder, room) for room in listdir(cloud_folder) if isdir(join(cloud_folder, room))]

            # Initiate containers
            cloud_points = np.empty((0, 3), dtype=np.float32)
            cloud_colors = np.empty((0, 3), dtype=np.uint8)
            cloud_classes = np.empty((0, 1), dtype=np.int32)

            # Loop over rooms
            for i, room_folder in enumerate(room_folders):

                print('Cloud %s - Room %d/%d : %s' % (cloud_name, i+1, len(room_folders), room_folder.split('/')[-1]))

                for object_name in listdir(join(room_folder, 'Annotations')):

                    if object_name[-4:] == '.txt':

                        # Text file containing point of the object
                        object_file = join(room_folder, 'Annotations', object_name)

                        # Object class and ID
                        tmp = object_name[:-4].split('_')[0]
                        if tmp in self.name_to_label:
                            object_class = self.name_to_label[tmp]
                        elif tmp in ['stairs']:
                            object_class = self.name_to_label['clutter']
                        else:
                            raise ValueError('Unknown object name: ' + str(tmp))

                        # Read object points and colors
                        try:
                            object_data = np.loadtxt(object_file, dtype=np.float32)
                        except ValueError:
                            # Correct bug in S3DIS dataset
                            warnings.warn('Bugs in: ' + object_file + '. Try fixing...')
                            import re
                            pattern = r'[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F-\x9F]'
                            with open(object_file, 'r') as f:
                                lines = f.readlines()
                            for l_i, line in enumerate(lines):
                                if re.search(pattern, line):
                                    print('Line %d contains non-ascii characters. Fixing...' % l_i)
                                    ref_n = len(lines[l_i - 1].strip().split())
                                    cur_n = len(lines[l_i].strip().split())
                                    lines[l_i] = re.sub(pattern, '' if cur_n == ref_n else ' ', line)
                            with open(object_file, 'w') as f:
                                f.writelines(lines)
                            object_data = np.loadtxt(object_file, dtype=np.float32)

                        # Stack all data
                        cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
                        cloud_colors = np.vstack((cloud_colors, object_data[:, 3:6].astype(np.uint8)))
                        object_classes = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
                        cloud_classes = np.vstack((cloud_classes, object_classes))

            # Save as ply
            write_ply(cloud_file,
                      (cloud_points, cloud_colors, cloud_classes),
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        print('Done in {:.1f}s'.format(time.time() - t0))
        return

    def load_subsampled_clouds(self):
        """
        Load or generate subsampled point cloud data and associated KD-Trees for the S3DIS dataset.

        This method performs the following tasks:
        1. For each scene (PLY file) in the dataset:
        - Loads precomputed subsampled clouds and KD-Trees if available.
        - Otherwise, reads the raw PLY file, applies voxel-based grid subsampling to reduce point count,
            builds a KD-Tree on the subsampled cloud, and saves both the tree and subsampled data to disk.
        2. Stores point positions, colors, and labels into `self.input_trees`, `self.input_colors`, and `self.input_labels`.
        3. If `use_potentials` is enabled:
        - Computes an additional coarsely subsampled KD-Tree for each cloud to be used in potential-based sampling.
        - These coarse KD-Trees are stored in `self.pot_trees`.

        The subsampling is controlled by `self.config.first_subsampling_dl` (fine) and `self.config.in_radius / 10` (coarse).

        Raises:
            FileNotFoundError: If a required PLY file is missing and cannot be read.
            ValueError: If subsampling parameters are misconfigured.

        Side Effects:
            - Creates directories and writes `.pkl` and `.ply` files under `input_{dl}/` if not already present.
            - Populates in-memory data structures needed for fast sampling and batch creation.

        Notes:
            - Uses `grid_subsampling` for downsampling.
            - Uses `KDTree` from `sklearn.neighbors` for fast nearest-neighbor search.
        """
        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))
            print("KDTree_file:", KDTree_file)
            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_labels = data['class']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                labels = data['class']

                # Subsample cloud
                sub_points, sub_colors, sub_labels = grid_subsampling(points,
                                                                      features=colors,
                                                                      labels=labels,
                                                                      sampleDl=dl)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)
                #search_tree = nnfln.KDTree(n_neighbors=1, metric='L2', leaf_size=10)
                #search_tree.fit(sub_points)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_colors, sub_labels],
                          ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]

            size = sub_colors.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            for i, file_path in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

                # Check if inputs have already been computed
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]
                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = data['class']

                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    #dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class S3DISSampler(Sampler):
    """
    Sampler for the S3DIS dataset used in point cloud segmentation.

    This sampler determines how to select regions (spheres) from the dataset to feed into the network for training or validation.
    It ensures that regions are sampled in a way that balances class representation and covers the dataset efficiently.

    The behavior depends on the `use_potentials` flag:
    - If `use_potentials=True`: region centers are chosen based on potentials (handled elsewhere).
    - If `use_potentials=False`: this sampler selects fixed numbers of regions per class, aiming for balanced sampling.

    Parameters
    ----------
    dataset : S3DISDataset
        The dataset to sample from. The sampler holds a reference to it (no deep copy).

    Attributes
    ----------
    dataset : S3DISDataset
        Reference to the dataset for sampling labels and point cloud metadata.

    N : int
        Number of steps (batches) per epoch, set depending on whether this is training or validation.
    """

    def __init__(self, dataset: S3DISDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return

    def __iter__(self):
        """
        Generate the indices for each training or validation batch.

        If `use_potentials` is False, this method:
        - Randomly selects a set of point indices across all clouds.
        - Ensures balanced representation by choosing approximately the same number of samples per class.
        - Stores the selected indices in `dataset.epoch_inds`.

        Then, yields one index per step, which is used by the dataset to fetch the corresponding region (sphere).

        Yields
        ------
        int
            A batch index (used internally to index `epoch_inds`).
        """

        if not self.dataset.use_potentials:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int64)

            # Number of sphere centers taken per class in each cloud
            num_centers = self.N * self.dataset.config.batch_num
            random_pick_n = int(np.ceil(num_centers / self.dataset.config.num_classes))

            # Choose random points of each class for each cloud
            epoch_indices = np.zeros((2, 0), dtype=np.int64)
            for label_ind, label in enumerate(self.dataset.label_values):
                if label not in self.dataset.ignored_labels:

                    # Gather indices of the points with this label in all the input clouds 
                    all_label_indices = []
                    for cloud_ind, cloud_labels in enumerate(self.dataset.input_labels):
                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        all_label_indices.append(np.vstack((np.full(label_indices.shape, cloud_ind, dtype=np.int64), label_indices)))

                    # Stack them: [2, N1+N2+...]
                    all_label_indices = np.hstack(all_label_indices)

                    # Select a a random number amongst them
                    N_inds = all_label_indices.shape[1]
                    if N_inds < random_pick_n:
                        chosen_label_inds = np.zeros((2, 0), dtype=np.int64)
                        while chosen_label_inds.shape[1] < random_pick_n:
                            chosen_label_inds = np.hstack((chosen_label_inds, all_label_indices[:, np.random.permutation(N_inds)]))
                        warnings.warn('When choosing random epoch indices (use_potentials=False), \
                                       class {:d}: {:s} only had {:d} available points, while we \
                                       needed {:d}. Repeating indices in the same epoch'.format(label,
                                                                                                self.dataset.label_names[label_ind],
                                                                                                N_inds,
                                                                                                random_pick_n))

                    elif N_inds < 50 * random_pick_n:
                        rand_inds = np.random.choice(N_inds, size=random_pick_n, replace=False)
                        chosen_label_inds = all_label_indices[:, rand_inds]

                    else:
                        chosen_label_inds = np.zeros((2, 0), dtype=np.int64)
                        while chosen_label_inds.shape[1] < random_pick_n:
                            rand_inds = np.unique(np.random.choice(N_inds, size=2*random_pick_n, replace=True))
                            chosen_label_inds = np.hstack((chosen_label_inds, all_label_indices[:, rand_inds]))
                        chosen_label_inds = chosen_label_inds[:, :random_pick_n]

                    # Stack for each label
                    all_epoch_inds = np.hstack((all_epoch_inds, chosen_label_inds))

            # Random permutation of the indices
            random_order = np.random.permutation(all_epoch_inds.shape[1])[:num_centers]
            all_epoch_inds = all_epoch_inds[:, random_order].astype(np.int64)

            # Update epoch inds
            self.dataset.epoch_inds += torch.from_numpy(all_epoch_inds)

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    def fast_calib(self):
        """
        Quickly calibrate the batch size (`batch_limit`) to match the desired average number of samples per batch.

        This method performs a lightweight calibration loop using a Proportional (P) controller to adjust the 
        number of points allowed per batch (`dataset.batch_limit`). It is designed for faster convergence than 
        the full calibration routine, and is especially useful in datasets like Semantic3D where:
        
        - Point cloud densities vary significantly.
        - Early samples may all come from dense regions, leading to misleading batch size estimates.
        - Potentials have not yet been updated across the dataset, increasing the chance of selecting biased samples.

        The method ensures that the average batch size converges toward the configured target 
        (`dataset.config.batch_num`), using a low-pass filter and error smoothing to detect convergence.

        Returns
        -------
        None

        Side Effects
        ------------
        - Modifies `self.dataset.batch_limit` in-place to achieve stable batch size behavior.
        - Displays progress information during calibration.
        - Stops when the estimated batch size (`estim_b`) converges to the target.

        Notes
        -----
        - This calibration is approximate but much faster than the full `calibration()` method.
        - Best used at the beginning of training, especially on large-scale datasets with uneven point distributions.
        """

        # Estimated average batch size and target value
        estim_b = 0
        target_b = self.dataset.config.batch_num

        # Calibration parameters
        low_pass_T = 10
        Kp = 100.0
        finer = False
        breaking = False

        # Convergence parameters
        smooth_errors = []
        converge_threshold = 0.1

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(2)

        for epoch in range(10):
            for i, test in enumerate(self):

                # New time
                t = t[-1:]
                t += [time.time()]

                # batch length
                b = len(test)

                # Update estim_b (low pass filter)
                estim_b += (b - estim_b) / low_pass_T

                # Estimate error (noisy)
                error = target_b - b

                # Save smooth errors for convergene check
                smooth_errors.append(target_b - estim_b)
                if len(smooth_errors) > 10:
                    smooth_errors = smooth_errors[1:]

                # Update batch limit with P controller
                self.dataset.batch_limit += Kp * error

                # finer low pass filter when closing in
                if not finer and np.abs(estim_b - target_b) < 1:
                    low_pass_T = 100
                    finer = True

                # Convergence
                if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                    breaking = True
                    break

                # Average timing
                t += [time.time()]
                mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d},  //  {:.1f}ms {:.1f}ms'
                    print(message.format(i,
                                         estim_b,
                                         int(self.dataset.batch_limit),
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1]))

            if breaking:
                break

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Perform automatic calibration of batch size and neighborhood limits for point cloud processing.

        This method tunes two key parameters needed for efficient and stable training of KPConv-based networks:

        1. Batch Calibration:
        - Adjusts the `batch_limit` (maximum number of points per batch) using a PID control loop.
        - The goal is to ensure that the average number of stacked point clouds (i.e., spheres/regions) 
            in each batch reaches a target value set by `config.batch_num`.

        2. Neighborhood Calibration:
        - Computes per-layer `neighborhood_limits`, which restrict the number of neighbors considered
            during convolution at each layer.
        - These limits are set so that at least `untouched_ratio` (default: 90%) of neighborhoods are 
            unaffected (i.e., not clipped).

        Cached values are reused if available unless `force_redo=True` is specified.

        Parameters
        ----------
        dataloader : DataLoader
            PyTorch DataLoader used to iterate over the training dataset during calibration.
        
        untouched_ratio : float, optional (default: 0.9)
            The percentage of neighborhoods that should remain untouched by the neighbor limit cutoff.

        verbose : bool, optional (default: False)
            If True, prints detailed status information during the calibration process.

        force_redo : bool, optional (default: False)
            If True, ignores cached calibration results and recalibrates from scratch.

        Returns
        -------
        None

        Side Effects
        ------------
        - Sets `self.dataset.batch_limit` to a calibrated value.
        - Sets `self.dataset.neighborhood_limits`, a list of per-layer limits for neighborhood sizes.
        - Stores calibration results in 'batch_limits.pkl' and 'neighbors_limits.pkl' for reuse.
        - May raise an exception or show plots if convergence is not reached (for debugging).

        Raises
        ------
        ValueError
            If calibration fails to converge or cannot collect valid data (e.g., all input spheres are empty).

        Notes
        -----
        - This step is essential before training to ensure stable GPU memory usage and efficient neighborhood lookup.
        - It should be run after any significant change to point cloud sampling radius, subsampling, or batch size.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.use_potentials:
            sampler_method = 'potentials'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                               self.dataset.config.in_radius,
                                               self.dataset.config.first_subsampling_dl,
                                               self.dataset.config.batch_num)
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num
            
            # Expected batch size order of magnitude
            expected_N = 100000

            # Calibration parameters. Higher means faster but can also become unstable
            # Reduce Kp and Kd if your GP Uis small as the total number of points per batch will be smaller 
            low_pass_T = 100
            Kp = expected_N / 200
            Ki = 0.001 * Kp
            Kd = 5 * Kp
            finer = False
            stabilized = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False
            error_I = 0
            error_D = 0
            last_error = 0

            debug_in = []
            debug_out = []
            debug_b = []
            debug_estim_b = []

            #####################
            # Perform calibration
            #####################

            # number of batch per epoch 
            sample_batches = 999
            for epoch in range((sample_batches // self.N) + 1):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.cloud_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b
                    error_I += error
                    error_D = error - last_error
                    last_error = error


                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 30:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit += Kp * error + Ki * error_I + Kd * error_D

                    # Unstability detection
                    if not stabilized and self.dataset.batch_limit < 0:
                        Kp *= 0.1
                        Ki *= 0.1
                        Kd *= 0.1
                        stabilized = True

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.dataset.batch_limit)))

                    # Debug plots
                    debug_in.append(int(batch.points[0].shape[0]))
                    debug_out.append(int(self.dataset.batch_limit))
                    debug_b.append(b)
                    debug_estim_b.append(estim_b)

                if breaking:
                    break

            # Plot in case we did not reach convergence
            if not breaking:
                import matplotlib.pyplot as plt

                print("ERROR: It seems that the calibration have not reached convergence. Here are some plot to understand why:")
                print("If you notice unstability, reduce the expected_N value")
                print("If convergece is too slow, increase the expected_N value")

                plt.figure()
                plt.plot(debug_in)
                plt.plot(debug_out)

                plt.figure()
                plt.plot(debug_b)
                plt.plot(debug_estim_b)

                plt.show()

                a = 1/0


            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles


            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            if self.dataset.use_potentials:
                sampler_method = 'potentials'
            else:
                sampler_method = 'random'
            key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                                   self.dataset.config.in_radius,
                                                   self.dataset.config.first_subsampling_dl,
                                                   self.dataset.config.batch_num)
            batch_lim_dict[key] = float(self.dataset.batch_limit)
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)


        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class S3DISCustomBatch:
    """Custom batch definition with memory pinning for S3DIS"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 7) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def S3DISCollate(batch_data):
    return S3DISCustomBatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


def debug_upsampling(dataset, loader):
    """Shows which labels are sampled according to strategy chosen"""


    for epoch in range(10):

        for batch_i, batch in enumerate(loader):

            pc1 = batch.points[1].numpy()
            pc2 = batch.points[2].numpy()
            up1 = batch.upsamples[1].numpy()

            print(pc1.shape, '=>', pc2.shape)
            print(up1.shape, np.max(up1))

            pc2 = np.vstack((pc2, np.zeros_like(pc2[:1, :])))

            # Get neighbors distance
            p0 = pc1[10, :]
            neighbs0 = up1[10, :]
            neighbs0 = pc2[neighbs0, :] - p0
            d2 = np.sum(neighbs0 ** 2, axis=1)

            print(neighbs0.shape)
            print(neighbs0[:5])
            print(d2[:5])

            print('******************')
        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_timing(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config.batch_num
    estim_N = 0

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.cloud_inds) - estim_b) / 100
            estim_N += (batch.features.shape[0] - estim_N) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}'
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b,
                                     estim_N))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset, loader):


    for epoch in range(10):

        clouds = []
        cloud_normals = []
        cloud_labels = []

        L = dataset.config.num_layers

        for batch_i, batch in enumerate(loader):

            # Print characteristics of input tensors
            print('\nPoints tensors')
            for i in range(L):
                print(batch.points[i].dtype, batch.points[i].shape)
            print('\nNeigbors tensors')
            for i in range(L):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print('\nPools tensors')
            for i in range(L):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print('\nStack lengths')
            for i in range(L):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print('\nFeatures')
            print(batch.features.dtype, batch.features.shape)
            print('\nLabels')
            print(batch.labels.dtype, batch.labels.shape)
            print('\nAugment Scales')
            print(batch.scales.dtype, batch.scales.shape)
            print('\nAugment Rotations')
            print(batch.rots.dtype, batch.rots.shape)
            print('\nModel indices')
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print('\nAre input tensors pinned')
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for epoch in range(10):

        for batch_i, input_list in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} '
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1]))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)
