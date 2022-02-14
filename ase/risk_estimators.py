import logging
import numpy as np


class RiskEstimator:
    def __init__(self, loss, *args, risk_cfg=None, **kwargs):
        from ase.utils import maps
        self.loss = maps.loss[loss]()
        self.risks = np.array([[]])
        self.risk_cfg = risk_cfg

    def return_and_save(self, loss):
        self.risks = np.append(self.risks, loss)
        return loss


class TrueRiskEstimator(RiskEstimator):
    """Used for performance evaluation."""
    def __init__(self, loss, dataset, model, *args, **kwargs):
        super().__init__(loss)

        idxs = dataset.test_idxs
        y_true = dataset.y[idxs]
        y_pred = model.predict(dataset.x[idxs], idxs=idxs)
        self.true_loss_vals = self.loss(y_pred, y_true)
        self.true_loss = self.true_loss_vals.mean()

        if (dataset.N < dataset.y.size) and dataset.cfg.get('with_unseen'):
            N = dataset.y.size
        else:
            N = dataset.N

        self.true_loss_all_idxs = np.zeros(N)
        self.true_loss_all_idxs[idxs] = self.true_loss_vals
        # print('true loss debug', self.true_loss)

    def estimate(self, *args, **kwargs):
        return self.return_and_save(self.true_loss)


class ExactExpectedRiskEstimator(RiskEstimator):
    """Used for performance evaluation."""
    def __init__(self, loss, dataset, model, *args, **kwargs):
        super().__init__(loss)

        # We have E_x[f(x)] = E_x[\sum x_j]
        # with f(x) = \prod_j p(x_j)
        # \int p(x) f(x) dx
        # = \int \prod_j p(x_j) \sum_j x_j
        # = \sum_j \int p(x_j) x_j
        # = \sum_j E[x_j]
        # = \sum_j \mu_j
        # = N * \mu

        # the individual values don't matter for this one..
        # I'm not sure why
        if dataset.__class__.__name__ == 'DoubleGaussianDataset':
            self.true_loss = dataset.true_expectation

        elif dataset.__class__.__name__ == 'OnlineToyDataset':
            dargs = dataset.creation_args
            test_mean, n_pixels = dargs['test_mean'], dargs['n_pixels']

            if dargs['normalise']:
                self.true_loss = test_mean
            else:
                self.true_loss = n_pixels * test_mean

        else:
            raise ValueError

        logging.info(f'*******THE TRUE VALUE IS {self.true_loss}')

    def estimate(self, *args, **kwargs):
        return self.return_and_save(self.true_loss)


class TrueUnseenRiskEstimator(RiskEstimator):
    """Used for performance evaluation."""
    def __init__(self, loss, dataset, model, *args, **kwargs):
        super().__init__(loss)

        # not compatible with lazy prediction
        idxs = dataset.test_unseen_idxs
        y_true = dataset.y[idxs]
        y_pred = model.predict(dataset.x[idxs], idxs=idxs)
        self.true_loss_vals = self.loss(y_pred, y_true)
        self.true_loss = self.true_loss_vals.mean()

        if (dataset.N < dataset.y.size) and dataset.cfg.get('with_unseen'):
            N = dataset.y.size
        else:
            N = dataset.N

        self.true_loss_all_idxs = np.zeros(N)
        self.true_loss_all_idxs[idxs] = self.true_loss_vals
        # print('true loss debug', self.true_loss)

    def estimate(self, *args, **kwargs):
        return self.return_and_save(self.true_loss)


class BiasedRiskEstimator(RiskEstimator):
    def __init__(self, loss, *args, **kwarg):
        super().__init__(loss)

    def estimate(self, predictions, observed, *args, **kwargs):
        l_i = self.loss(predictions, observed).mean()
        # logging.info(f'debug biased risk estimator {l_i}')
        return self.return_and_save(l_i)


class NaiveUnbiasedRiskEstimator(RiskEstimator):
    def __init__(self, loss, dataset, *args, **kwargs):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    def estimate(self, predictions, observed, acq_weights, *args, **kwargs):

        l_i = self.loss(predictions, observed)
        N = self.N
        M = len(predictions)
        m = np.arange(1, M+1)

        v = 1/(N * acq_weights) + (M-m) / N

        R = 1/M * (v * l_i).sum()

        return self.return_and_save(R)


class FancyUnbiasedRiskEstimator(RiskEstimator):
    def __init__(self, loss, dataset, *args, **kwargs):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    @staticmethod
    def get_weights(acq_weights, N):
        M = len(acq_weights)
        if M < N:
            m = np.arange(1, M+1)
            v = (
                1
                + (N-M)/(N-m) * (
                        1 / ((N-m+1) * acq_weights)
                        - 1
                        )
                )
        else:
            v = 1

        return v

    def estimate(self, predictions, observed, acq_weights, *args, **kwargs):

        l_i = self.loss(predictions, observed)
        N = self.N
        M = len(predictions)

        v = self.get_weights(acq_weights, N)

        R = 1/M * (v * l_i).sum()

        return self.return_and_save(R)

class FullSurrogateASMC(RiskEstimator):
    """Rely entirely on surrogate for ASMC. Do not update with observations.
    """
    def __init__(self, loss, dataset, model, risk_cfg=None, *args, **kwargs):
        super().__init__(loss)

        self.dataset = dataset
        self.task = self.dataset.cfg.task_type

        self.cfg = risk_cfg

        if self.cfg is not None and (lim := self.cfg.get('limit', False)):
            n_sub = len(self.dataset.test_idxs)
            self.test_idxs = np.random.choice(
                self.dataset.test_idxs, size=round(n_sub * lim), replace=False)
        else:
            self.test_idxs = self.dataset.test_idxs

        self.x_test = self.dataset.x[self.test_idxs]

        self.model = model

    # @profile
    def estimate(
            self, predictions, observed, weights, surrogate, acquisition_name,
            *args, **kwargs):

        if acquisition_name == 'RandomAcquisition':
            return self.return_and_save(-1e19)

        # Risk estimate using the surrogate model over all test data
        # ---------------------------------------
        # Remaining data
        if self.cfg is not None and self.cfg.get('increase_pool', False):
            if (len(observed) > self.cfg.get('after_m_steps', 0)):
                N_test = self.cfg['N_test']
                x_test = self.dataset.p.rvs(N_test)
                test_idxs = None
                if self.dataset.cfg.creation_args['dim'] == 1:
                    x_test = x_test[:, np.newaxis]
            else:
                return self.return_and_save(-1e19)
        else:
            x_test = self.x_test
            test_idxs = self.test_idxs

        model_predictions = self.model.predict(x_test, idxs=test_idxs)

        if surrogate is None:

            # A bit hacky. Fail silently.
            # This estimator will only return sensible values for acquisition
            # strategies that have a surrogate model.
            # This is ugly like this in the code because
            # estimators are always applied to all acquisition strategies.
            surr_predictions = model_predictions
        else:
            surr_predictions = surrogate.predict(x_test, idxs=test_idxs)

        if self.task == 'regression':
            # will return predictions as truth, will raise if used for
            # loss estimation for now (because model does not predict)
            # over all datapoints)
            R = self.loss(model_predictions, surr_predictions).mean()
        else:
            eps = 1e-20

            model_predictions = np.clip(model_predictions, eps, 1 - eps)
            model_predictions /= model_predictions.sum(axis=1, keepdims=True)

            R = -1 * (surr_predictions * np.log(model_predictions)).sum(-1)
            R = R.mean()

        return self.return_and_save(R)


QuadratureRiskEstimator = FancyUnbiasedRiskEstimator
