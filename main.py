"""Main active testing loop."""
import os
import sys
import socket
import logging
import hydra
import warnings

import numpy as np
import torch

from ase.experiment import Experiment
from ase.utils import maps
from ase.utils.utils import add_val_idxs_to_cfg
from ase.utils.data import to_json
from ase.hoover import Hoover
from ase.models import make_efficient
from omegaconf import OmegaConf


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    """Run main experiment loop.

    Repeat active testing across multiple data splits and acquisition
    functions for all risk estimators.
    """

    rng = cfg.experiment.random_seed
    if rng == -1:
        rng = np.random.randint(0, 1000)

    if rng is not False:
        np.random.seed(rng)
        torch.torch.manual_seed(rng)

    dcc = cfg.dataset.get('creation_args', dict())
    if dcc.get('dim_normalise_mean', False):
        dim = dcc.dim
        dcc.f_mean = float(dcc.f_mean / np.sqrt(dim))
        dcc.p_mean = float(dcc.p_mean / np.sqrt(dim))
        logging.info(
            f'Updating means in dataset cfg: {cfg.dataset.creation_args}')

    stats = dict(
        dir=os.getcwd(),
        host=socket.gethostname(),
        job_id=os.getenv("SLURM_JOB_ID", None),
        random_state=rng)
    STATS_STATUS = False

    logging.info(
        f'Logging to {stats["dir"]} on {stats["host"]} '
        f'for id={cfg.get("id", -1)}')

    logging.info(f'Slurm job: {stats["job_id"]}.')
    logging.info(f'Setting random seed to {rng}.')
    logging.info(f'Uniform clip val is {cfg.acquisition.uniform_clip}.')

    hoover = Hoover(cfg.hoover)

    model = None

    # Right now this averages over both train and testing!
    for run in range(cfg.experiment.n_runs):
        if run % cfg.experiment.log_every == 0 or cfg.experiment.debug:
            logging.info(f'Run {run} in {os.getcwd()} ****NEW RUN****')
            if cuda := torch.cuda.is_available():
                logging.info(f'Still using cuda: {cuda}.')
            else:
                logging.info('No cuda found!')
                os.system('touch cuda_failure.txt')

        dataset = maps.dataset[cfg.dataset.name](
            cfg.dataset, model_cfg=cfg.model)

        # Train model on training data.
        if (not cfg.model.get('keep_constant', False)) or (model is None):
            model = maps.model[cfg.model.name](cfg.model)
            model.fit(*dataset.train_data)

            loss = model.performance(
                *dataset.test_data, dataset.cfg['task_type'])

            if cfg.experiment.get('constant_val_set', False):
                add_val_idxs_to_cfg(cfg, model.val_idxs)

            if not STATS_STATUS:
                STATS_STATUS = True
                stats['loss'] = loss
                to_json(stats, 'stats.json')
            # test_data = model.make_loader(dataset.test_data, train=False)
            # loss = model.evaluate(model.model, test_data)
            # logging.info(f'Model test loss is {loss}.')

        # Always predict on test data again
        # TODO: need to fix this for efficient prediction
        if cfg.model.get('efficient', False):
            logging.debug('Eficient prediction on test set.')
            model = make_efficient(model, dataset)

        # if cfg.experiment.debug:
            # Report train error
            # logging.info('Model train error:')
            # model.performance(
            #     *dataset.train_data, dataset.cfg.task_type)

        # if not check_valid(model, dataset):
            # continue

        if run < cfg.experiment.save_data_until:
            hoover.add_data(run, dataset.export())

        for acq_dict in cfg.acquisition_functions:
            # Slightly unclean, but could not figure out how to make
            # this work with Hydra otherwise
            acquisition = list(acq_dict.keys())[0]
            acq_cfg_name = list(acq_dict.values())[0]

            if cfg.experiment.debug:
                logging.info(f'\t Acquisition: {acquisition}')

            # Reset selected test_indices.
            dataset.restart(acquisition)

            if (n := acq_cfg_name) is not None:
                acq_config = cfg['acquisition_configs'][n]
            else:
                acq_config = None

            experiment = Experiment(
                run, cfg, dataset, model, acquisition, acq_config)

            i = 0
            while not experiment.finished:
                i += 1
                # print('debug', i)
                if cfg.experiment.debug:
                    logging.info(
                        f'\t Acquisition: {acquisition} – \t Step {i}.')

                experiment.step(i)

            # Add config to name for logging.
            if (n := acq_cfg_name) is not None:
                acquisition = f'{acquisition}_{n}'

            # Extract results from acquisition experiment
            hoover.add_results(run, acquisition, experiment.export_data())

        if run % cfg.experiment.get('save_every', 1e19) == 0:
            logging.info('Intermediate save.')
            hoover.save()

    logging.info('Completed all runs.')
    hoover.save()


def check_valid(model, dataset):
    """For classification with small number of points and unstratified."""
    if hasattr(model.model, 'n_classes_'):
        if (nc := model.model.n_classes_) != dataset.cfg.n_classes:
            warnings.warn(
                f'Not all classes present in train data. '
                f'Skipping run.')
            return False
    return True


if __name__ == '__main__':
    os.environ['HYDRA_FULL_ERROR'] = '1'

    BASE_DIR = os.getenv('BASE_DIR', default='.')
    RAND = np.random.randint(10000)

    print(
        f"Env variable BASE_DIR: {BASE_DIR}")
    sys.argv.append(f'+BASE_DIR={BASE_DIR}')
    sys.argv.append(f'+RAND={RAND}')

    OmegaConf.register_new_resolver('BASE_DIR', lambda: BASE_DIR)
    OmegaConf.register_new_resolver('RAND', lambda: RAND)

    main()
