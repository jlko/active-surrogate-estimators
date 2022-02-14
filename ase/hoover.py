"""Keep track of data across runs."""
from pathlib import Path
import logging
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd


class Hoover:
    """Sucks up all data generated by experiment."""

    def __init__(self, cfg, name=None):
        self.datasets = dict()
        self.pmfs = defaultdict(dict)
        self.loss_data = defaultdict(dict)
        self.results = None
        self.cfg = cfg
        if name is None:
            self.name = None
        else:
            self.name = name

    def add_data(self, run, export):
        if self.cfg.get('save_data', False):
            self.datasets.update({run: export})

    def add_results(self, run, acquisition, export):
        export, all_pmfs, loss_data = export
        export['run'] = run
        export['acquisition'] = acquisition

        if self.results is None:
            self.results = export
        else:
            self.results = self.results.append(export, ignore_index=True)

        if all_pmfs is not None:
            self.pmfs[run][acquisition] = all_pmfs

        if all_pmfs is not None:
            self.loss_data[run][acquisition] = loss_data

    def save(self):
        if self.name is None:
            base = Path('.')
        else:
            base = Path(f'model_{self.name}')
            base.mkdir(parents=True, exist_ok=True)

        if self.cfg.save_data:
            pickle.dump(self.datasets, open(base / "datasets.pkl", "wb"))

        pickle.dump(self.pmfs, open(base / "pmfs.pkl", "wb"))
        pickle.dump(self.loss_data, open(base / "loss_data.pkl", "wb"))

        self.results.to_pickle(base / 'results.pkl')
        logging.info('Saving results to file.')


class ModelSelectionHoover:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logs = []
        self.cols = [
            'run', 'acquisition', 'step', 'model', 'risk', 'prob', 'pos',
            'best', 'true_pos', 'true_best']

    def add_data(self, run, acquisition, logs):

        logs = np.array(logs)

        runs = np.array(logs.shape[0] * [run])
        acquisitions = np.array(logs.shape[0] * [acquisition])
        combined_log = np.concatenate([
            runs[:, np.newaxis], acquisitions[:, np.newaxis], logs], 1)

        self.logs += [combined_log]

    def save(self):
        base = Path('.')

        logs = np.concatenate(self.logs, 0)
        log_df = pd.DataFrame(data=logs, columns=self.cols)

        log_df.to_pickle(base / 'model_selection.pkl')
