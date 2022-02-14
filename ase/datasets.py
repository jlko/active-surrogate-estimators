"""Datasets for active testing."""
import time
import os
from copy import deepcopy
import warnings
import logging
import pickle
from pathlib import Path
from numpy.lib.arraypad import _round_if_needed
from omegaconf import OmegaConf, DictConfig
import hydra
from pathos.pools import _ProcessPool as Pool

import math
import numpy as np
from sklearn.model_selection import train_test_split as SKtrain_test_split
from scipy.stats import multivariate_normal

import torch
from torch._C import Value
from torch.distributions.multivariate_normal import MultivariateNormal

from ase.utils.data import (
    to_json, get_root, array_reduce)


class _Dataset:
    """Implement generic dataset.

    Load and preprocess data.
    Provide basic acess, train-test split.

    raise generic methods
    """
    def __init__(self, cfg):

        # Set task_type and global_std if not present.
        self.cfg = OmegaConf.merge(
                dict(cfg),
                DictConfig(dict(
                    task_type=cfg.get('task_type', 'regression'),
                    global_std=cfg.get('global_std', False),
                    n_classes=cfg.get('n_classes', -1))))

        self.N = cfg.n_points
        self.x, self.y = self.generate_data()

        # For 1D data, ensure Nx1 shape
        if self.x.ndim == 1:
            self.x = self.x[:, np.newaxis]

        self.D = self.x.shape[1:]

        self.train_idxs, self.test_idxs = self.train_test_split(self.N)
        self.x_test = self.x[self.test_idxs]

        to_json(
            dict(
                train_idxs=self.train_idxs.tolist(),
                test_idxs=self.test_idxs.tolist()),
            Path('train_test_split.json'))

        if self.cfg.standardize:
            self.standardize()

    def train_test_split(self, N, test_size=None):
        if (N < self.y.size) and self.cfg.get('with_unseen'):
            all_indices = np.random.choice(
                np.arange(0, self.y.size),
                N,
                replace=False)
        else:
            all_indices = np.arange(0, N)

        if self.cfg.get('stratify', False):
            stratify = self.y[all_indices]
        else:
            stratify = None

        if test_size is None:
            test_size = self.cfg.test_proportion

        if test_size == 1:
            train = np.array([]).astype(np.int)
            test = all_indices
        else:
            train, test = SKtrain_test_split(
                    all_indices, test_size=test_size,
                    stratify=stratify)

        assert np.intersect1d(train, test).size == 0
        assert np.setdiff1d(
            np.union1d(train, test),
            all_indices).size == 0

        # this option splits the test into test and unseen
        if p := self.cfg.get('test_unseen_proportion', False):
            if (N < self.y.size) and self.cfg.get('with_unseen'):
                raise ValueError('Not compatible.')
            test, test_unseen_idxs = SKtrain_test_split(
                np.arange(0, len(test)), test_size=p)
            self.test_unseen_idxs = test_unseen_idxs

        # this option takes all test idxs as unseen
        if self.cfg.get('with_unseen', False):
            test_unseen_idxs = np.setdiff1d(
                np.arange(0, self.y.size), train)
            self.test_unseen_idxs = test_unseen_idxs

        if self.cfg.get('remove_initial_75_keep_train_size', False):
            # need to do this here, before unseen is applied
            train, test_unseen_idxs = self.remove_initial_75_keep_train_size(
                train, test, self.test_unseen_idxs, self.y)
            self.test_unseen_idxs = test_unseen_idxs

        if (freq := self.cfg.get('filter_nums_relative_frequency', 1)) != 1:
            test = self.reduce_filter_num_frequency(freq, test)

        assert np.intersect1d(train, test).size == 0

        if self.cfg.get('with_unseen', False):
            assert np.intersect1d(test_unseen_idxs, train).size == 0
            assert np.intersect1d(
                test_unseen_idxs, test).size == test.size
        if self.cfg['task_type'] == 'classification':
            logging.info(
                f'Final bincount for test {np.bincount(self.y[test])}.')

        return train, test

    def reduce_filter_num_frequency(self, freq, test):
        nums = self.cfg.filter_nums

        _7_bin = array_reduce(self.y, nums)
        _7 = np.where(_7_bin)
        test_7 = np.intersect1d(test, _7)
        assert np.all(array_reduce(self.y[test_7], nums))

        old_num_7 = len(test_7)
        new_num_7 = round(freq * old_num_7)

        # draw those indices to delete
        delete_7 = np.random.choice(
            test_7, size=old_num_7 - new_num_7, replace=False)

        new_test = np.setdiff1d(test, delete_7)

        # delete 7 no longer in teset
        assert np.intersect1d(new_test, delete_7).size == 0
        # test been reduced by delete 7 amount
        assert len(test) - len(new_test) == len(delete_7)
        # deleted as many as we wanted
        assert len(delete_7) == old_num_7 - new_num_7
        # new num count is reduced by frequency
        assert array_reduce(self.y[new_test], nums).sum() == new_num_7

        return new_test

    def remove_initial_75_keep_train_size(
            self, train, test, test_unseen_idxs, y):

        # need to have extra data, because we don't want to copy from test
        if not self.cfg.get('with_unseen', False):
            raise ValueError

        nums = self.cfg.filter_nums
        # find number of 5s and 7s in train
        # train_7_bin = (y[train] == 7) | (y[train] == 5)
        train_7_bin = array_reduce(y[train], nums)

        train_7 = np.where(train_7_bin)[0]
        n_replace = len(train_7)

        # assert np.all((y[train[train_7]] == 5) | (y[train[train_7]] == 7))
        assert np.all(array_reduce(y[train][train_7], nums))

        # find some *unseen* idxs that are not 5 or 7
        y = self.y
        unseen_not_test = np.setdiff1d(test_unseen_idxs, test)
        assert np.intersect1d(unseen_not_test, train).size == 0
        assert np.intersect1d(unseen_not_test, test).size == 0
        unseen_not_test_bin = np.zeros(y.size, dtype=np.bool)
        unseen_not_test_bin[unseen_not_test] = 1
        # unseen_not_75_bin = unseen_not_test_bin & (y != 7) & (y != 5)
        unseen_not_75_bin = unseen_not_test_bin & array_reduce(
            y, nums, compare='neq', combine='and')
        unseen_not_75 = np.where(unseen_not_75_bin)[0]

        # assert np.all((y[unseen_not_75] != 7) | (y[unseen_not_75] != 5))
        assert np.all(array_reduce(y[unseen_not_75], nums, compare='neq'))

        # chose as many as we want to swap
        chosen = np.random.choice(unseen_not_75, n_replace, replace=False)

        # remove the chosen ones from the unseen idx
        # I don't want to bias my true loss
        # (don't need to remove from test because they where never in test)
        pre_size = len(test_unseen_idxs)
        test_unseen_idxs = np.setdiff1d(test_unseen_idxs, chosen)
        post_size = len(test_unseen_idxs)
        assert pre_size - post_size == len(chosen)
        assert np.intersect1d(test_unseen_idxs, chosen).size == 0

        # now replace the train idxs with the chosen idxs
        train[train_7] = chosen

        # assert np.all((y[train[train_7]] != 5) | (y[train[train_7]] != 7))
        # assert np.all((y[train] != 5) | (y[train] != 7))
        assert np.all(array_reduce(y[train[train_7]], nums, compare='neq'))
        assert np.all(array_reduce(y[train], nums, compare='neq'))

        assert np.intersect1d(test_unseen_idxs, train).size == 0
        assert np.intersect1d(
            test_unseen_idxs, test).size == test.size

        return train, test_unseen_idxs

    @property
    def train_data(self):
        return self.x[self.train_idxs], self.y[self.train_idxs]

    @property
    def test_data(self):
        return self.x[self.test_idxs], self.y[self.test_idxs]

    def standardize(self):
        """Standardize to zero mean and unit variance using train_idxs."""

        ax = None if self.cfg['global_std'] else 0

        x_train, y_train = self.train_data

        x_std = self.cfg.get('x_std', True)
        if x_std:
            self.x_train_mean = x_train.mean(ax)
            self.x_train_std = x_train.std(ax)
            self.x = (self.x - self.x_train_mean) / self.x_train_std

        y_std = self.cfg.get('y_std', True)
        if (self.cfg['task_type'] == 'regression') and y_std:
            self.y_train_mean = y_train.mean(ax)
            self.y_train_std = y_train.std(ax)
            self.y = (self.y - self.y_train_mean) / self.y_train_std

    def export(self):
        package = dict(
            x=self.x,
            y=self.y,
            train_idxs=self.train_idxs,
            test_idxs=self.test_idxs
            )
        return package


class _ActiveTestingDataset(_Dataset):
    """Active Testing Dataset.

    Add functionality for active testing.

    Split test data into observed unobserved.

    Add Methods to keep track of unobserved/observed.
    Use an ordered set or sth to keep track of that.
    Also keep track of activation function values at time that
    sth was observed.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.start()

    def start(self):
        self.test_observed = np.array([], dtype=np.int)
        self.test_remaining = self.test_idxs

    def restart(self, *args):
        self.start()

    def observe(self, idx, with_replacement=True):
        """Observe data at idx and move from unobserved to observed.

        Note: For efficiency reasons idx is index in test
        """

        self.test_observed = np.append(self.test_observed, idx)
        if not with_replacement:
            self.test_remaining = self.test_remaining[
                self.test_remaining != idx]

        return self.x[[idx]], self.y[[idx]]

    @property
    def total_observed(self):
        """Return train and observed test data"""
        test = self.x[self.test_observed], self.y[self.test_observed]
        train = self.train_data
        # concatenate x and y separately
        total_observed = [
            np.concatenate([train[i], test[i]], 0)
            for i in range(2)]

        return total_observed


class MNISTDataset(_ActiveTestingDataset):
    """MNIST Data.

    TODO: Respect train/test split of MNIST.
    """
    def __init__(self, cfg, n_classes=10, *args, **kwargs):

        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='classification', global_std=True,
                 n_classes=n_classes))

        super().__init__(cfg)

    # def generate_data(self):
    #     from tensorflow.keras import datasets

    #     # data_home = Path(hydra.utils.get_original_cwd()) / 'data/MNIST'

    #     # # from sklearn.datasets import fetch_openml
    #     # # x, y = fetch_openml(
    #     # #     'mnist_784', version=1, return_X_y=True, data_home=data_home,
    #     # #     cache=True)
    #     # data = datasets.mnist.load_data(
    #     #     path=data_home / 'mnist.npz'
    #     # )

    #     data = datasets.mnist.load_data()

    #     return self.preprocess(data)

    def generate_data(self):
        from torchvision.datasets import MNIST

        train = MNIST(get_root() / 'data/torch_mnist', download=True)
        x_train, y_train = train.data, train.targets
        test = MNIST(
            get_root() / 'data/torch_mnist', download=False,
            train=False)
        x_test, y_test = test.data, test.targets

        data = ((x_train, y_train), (x_test, y_test))

        return self.preprocess(data)

    def preprocess(self, data):

        (x_train, y_train), (x_test, y_test) = data
        x = np.concatenate([x_train, x_test], 0)
        x = x.astype(np.float32) / 255
        x = x.reshape(x.shape[0], -1)
        y = np.concatenate([y_train, y_test], 0)
        y = y.astype(np.int)

        N = self.N

        if (N < y.size) and not self.cfg.get('with_unseen', False):
            logging.info('Keeping only a subset of the input data.')
            # get a stratified subset
            # note that mnist does not have equal class count
            # want to keep full data for unseeen
            idxs, _ = SKtrain_test_split(
                np.arange(0, y.size), train_size=N, stratify=y)
            x = x[idxs]
            y = y[idxs]
        # no longer want this. keep indices always true to dataset!!
        # elif (N < y.size) and not self.cfg.get('respect_train_test', False):
        #     logging.info('Scrambling input data.')
        #     # still want to scramble because the first N entries are used
        #     # to assign test and train data
        #     idxs = np.random.permutation(np.arange(0, y.size))
        #     x = x[idxs]
        #     y = y[idxs]

        return x, y

    def train_test_split(self, N):

        if self.cfg.get('respect_train_test', False):
            # use full train set, subsample from original test set
            train_lim = self.cfg.get('train_limit', 50000)
            train = np.arange(0, train_lim)

            n_test = round(self.cfg.test_proportion * N)
            max_test = 60e3 - train_lim
            if n_test <= max_test:

                replace = self.cfg.get('test_with_replacement', False)

                test = np.random.choice(
                    np.arange(train_lim, 60000), n_test, replace=replace)
                test = np.sort(test)
            else:
                raise ValueError

            self.test_unseen_idxs = np.setdiff1d(
                np.arange(train_lim, 60000), test)

            return train, test

        else:
            train, test = super().train_test_split(N)

        # only keep the first n sevens in the train distribution
        if n7 := self.cfg.get('n_initial_7', False):
            # to get correct indices, need to first select from y
            old7 = np.where(self.y == 7)[0]
            # then filter to train indicees
            old_train7 = np.intersect1d(old7, train)
            # now only keep the first n7
            sevens_remove = old_train7[n7:]
            # and now remove those from the train set
            train = np.setdiff1d(train, sevens_remove)

        return train, test


class FashionMNISTDataset(MNISTDataset):
    """FashionMNIST Data.

    TODO: Respect train/test split of FashionMNIST.
    """
    def __init__(self, cfg,
                 *args, **kwargs):

        super().__init__(cfg)

    # def generate_data(self):
    #     from tensorflow.keras import datasets
    #     data = datasets.fashion_mnist.load_data()

    #     return self.preprocess(data)

    def generate_data(self):
        from torchvision.datasets import FashionMNIST

        train = FashionMNIST(get_root() / 'data/torch_fmnist', download=True)
        x_train, y_train = train.data, train.targets
        test = FashionMNIST(
            get_root() / 'data/torch_fmnist',
            train=False, download=False)
        x_test, y_test = test.data, test.targets

        data = ((x_train, y_train), (x_test, y_test))

        return self.preprocess(data)


class Cifar10Dataset(MNISTDataset):
    """CIFAR10 Data.
    """
    def __init__(self, cfg,
                 *args, **kwargs):

        super().__init__(cfg)

    def generate_data(self):
        from tensorflow.keras import datasets
        data = datasets.cifar10.load_data()

        x, y = self.preprocess(data)
        x = x.reshape(len(x), 32, 32, 3).transpose(0, 3, 1, 2)
        x = x.reshape(len(x), -1)
        return x, y[:, 0]


class Cifar100Dataset(MNISTDataset):
    """CIFAR100 Data.
    """
    def __init__(self, cfg,
                 *args, **kwargs):

        super().__init__(cfg, n_classes=100)

    def generate_data(self):
        from tensorflow.keras import datasets
        data = datasets.cifar100.load_data()

        x, y = self.preprocess(data)
        x = x.reshape(len(x), 32, 32, 3).transpose(0, 3, 1, 2)
        x = x.reshape(len(x), -1)
        return x, y[:, 0]
