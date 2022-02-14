import os
import pickle
from functools import reduce
import json
import hydra
from pathlib import Path

from torch._C import Value


def to_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(data, file)


def from_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def to_pickle(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def get_root():
    return Path(hydra.utils.get_original_cwd())


def array_reduce(arr, nums, compare='equality', combine='or'):
    if compare in ['equality', 'eq']:
        conditions = [arr == num for num in nums]
    elif compare in ['inequality', 'neq', 'ineq']:
        conditions = [arr != num for num in nums]
    else:
        raise ValueError

    if combine == 'or':
        reduction = reduce(lambda x, y: x | y, conditions)
    elif combine == 'and':
        reduction = reduce(lambda x, y: x & y, conditions)
    else:
        raise ValueError

    return reduction
