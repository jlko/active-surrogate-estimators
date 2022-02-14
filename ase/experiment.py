"""Run experiment."""
import logging
from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np

from ase.utils import maps


class Experiment:
    """Orchestrates experiment.

    Main goal: Just need to call Experiment.run_experiment()
    and a model will get trained and tested.

    This trains and actively tests the models.

    Has a step() method.
    Main loop is probably externally controlled for logging purposes.
    Maybe not..
    """
    def __init__(self, run, cfg, dataset, model, acquisition, acq_config):
        self.cfg = cfg
        self.dataset = dataset
        self.model = model

        self.risk_estimators = self.get_risk_estimators()

        true_loss = self.risk_estimators['TrueRiskEstimator'].true_loss_vals

        # TODO: model_cfg is used for surr model.
        # TODO: allow to pass a different cfg here, matching the aux model
        # TODO: this is needed when aux model and model are different!

        acq_config = model.cfg if acq_config is None else acq_config

        self.acquisition_name = acquisition
        self.acquisition = (
            maps.acquisition[acquisition](
                [self.cfg.acquisition, run],
                self.dataset,
                true_loss_vals=true_loss,
                model=self.model,
                model_cfg=acq_config))

        self.has_surr = hasattr(self.acquisition, 'surr_model')

        self.finished = False
        self.predictions = None
        self.novels = []

    def get_risk_estimators(self):

        risk_config = self.cfg.get('risk_configs', False)

        if not risk_config:
            risk_estimators = {
                risk_estimator: maps.risk_estimator[risk_estimator](
                    self.cfg.experiment.loss,
                    self.dataset,
                    self.model,
                    )
                for risk_estimator in self.cfg.risk_estimators}
        else:
            risk_estimators = dict()
            for i in self.cfg.risk_estimators:
                risk_estimator = list(i.keys())[0]
                cfg_name = list(i.values())[0]
                if cfg_name is None:
                    risk_name = risk_estimator
                    risk_cfg = None
                else:
                    risk_name = f'{risk_estimator}-{cfg_name}'
                    risk_cfg = (
                        risk_config.get(cfg_name)
                        if cfg_name is not None else None)
                risk_estimators[risk_name] = maps.risk_estimator[
                            risk_estimator](
                        self.cfg.experiment.loss,
                        self.dataset,
                        self.model,
                        risk_cfg=risk_cfg)

        return risk_estimators

    # # @profile
    def estimate_risks(self):
        """Estimate test risk."""
        pred = self.predictions
        obs = self.dataset.y[self.dataset.test_observed]

        # For QuadratureRiskEstimator
        surr_model = getattr(self.acquisition, 'surr_model', None)

        for risk_estimator in self.risk_estimators.values():
            risk_estimator.estimate(
                pred, obs, self.acquisition.weights, surr_model,
                acquisition_name=self.acquisition_name)

    # @profile
    def step(self, i):
        """Perform a single testing step."""

        # choose index for next observation
        test_idx, pmf_idx = self.acquisition.acquire(update_surrogate=False)
        # logging.info(f'Acquiring idx {test_idx}')

        self.observe_at_idx(i, test_idx)


        y = self.dataset.y[self.dataset.test_observed]
        # logging.info(f'DEEEBUUGG  {y.max()}, {i}, {test_idx}, {y.argmax()}')
        if y.max() > 1e10:
            logging.info(f'FOUND BUG!')
            raise

        return test_idx, pmf_idx

    # @profile
    def observe_at_idx(self, i, idx):

        # add true pmf to logging to plot loss dist
        if self.acquisition.check_save(off=1):
            true_pmf = (
                self.risk_estimators[
                    'TrueRiskEstimator'].true_loss_all_idxs[
                    self.dataset.test_remaining])
            true_pmf = (
                self.acquisition.safe_normalise(
                    true_pmf))

            self.acquisition.all_pmfs[-1]['true_pmf'] = true_pmf

        # sampling with replacement?
        with_replacement = self.acquisition.cfg.get('with_replacement', False)

        # Option to only print increments in terms of novel acquisitions
        if with_replacement:
            self.novels.append(float(idx not in self.dataset.test_observed))
        else:
            # Always true.
            self.novels.append(1)

        # observe point
        x, _ = self.dataset.observe(idx, with_replacement=with_replacement)

        # predict at point
        y_pred = self.model.predict(x, [idx])

        if self.predictions is None:
            self.predictions = y_pred
        else:
            self.predictions = np.concatenate(
                [self.predictions, y_pred], 0)

        # update surrogates after acquisition but before risk estimation
        # (and before next loop as before)
        if hasattr(self.acquisition, 'update_surrogate'):
            self.acquisition.update_surrogate()

        # estimate test risk
        self.estimate_risks()

        # print(
        #     x, idx, self.dataset.test_observed.size,
        #     self.dataset.test_remaining.size)

        # Need the set for sampling w/ replacement
        if len(set(self.dataset.test_observed)) == len(self.dataset.test_idxs):
            self.finished = True

        if lim := self.cfg.experiment.get('abort_test_after', False):
            if i >= lim:
                self.finished = True

    def external_step(self, i, test_idx, pmf_idx, update_surrogate=True):
        """Externally force experiment to acquire data at 'idx'. """
        # hot-patch the forced acquisition
        # (would ideally make this passable to acquire()
        # but I can't be bothered
        self.acquisition.externally_controlled = True
        self.acquisition.ext_test_idx = test_idx
        self.acquisition.ext_pmf_idx = pmf_idx

        # need to call this s.t. acquisition weights are properly written
        self.acquisition.acquire(update_surrogate=update_surrogate)
        self.observe_at_idx(i, test_idx)

        # make valid for next round again
        self.acquisition.externally_controlled = False
        self.acquisition.ext_test_idx = None
        self.acquisition.ext_pmf_idx = None

    def export_data(self):
        """Extract data from experiment."""
        if self.dataset.cfg.task_type == 'classification':
            preds = np.argmax(self.predictions, 1)
        else:
            preds = self.predictions

        result = dict(
            id=np.arange(0, len(self.dataset.test_observed)),
            idx=self.dataset.test_observed,
            y_preds=preds,
            y_true=self.dataset.y[self.dataset.test_observed],
            novel=self.novels,
        )

        result.update(
            {risk_name: risk.risks for risk_name, risk
                in self.risk_estimators.items()})
        result = pd.DataFrame.from_dict(result)

        # also export true losses together with quadrature risk estimates
        # for these data (no clipping!)
        # only do this for first iteration for now.

        loss_data = dict()

        return result, self.acquisition.all_pmfs, loss_data
