"""Implement acquisition functions."""

from copy import deepcopy
import logging
import numpy as np
from omegaconf import OmegaConf

from ase.models import make_efficient


class AcquisitionFunction:
    """Acquisition function is its own class.

    In the beginning this may seem like overkill, but remember that our
    acquisition function will likely have a powerfull substitute model.

    Implement get_next_point
    """
    def __init__(self, cfg_run, dataset):
        self.cfg, run = cfg_run
        logging.info(f'**Initialising acquisition {self.__class__}.')

        self.dataset = dataset
        # keep track of acquisition weights
        self.weights = np.array([])

        if self.cfg.animate and run < self.cfg.animate_until:
            self.all_pmfs = list()
        else:
            self.all_pmfs = None

        self.counter = 0

        if self.cfg.lazy_save:

            if L := self.cfg.get('lazy_save_schedule', False):
                L = list(L)
            else:
                L = list(range(1000))
                L += list(range(int(1e3), int(1e4), 500))
                L += list(range(int(1e4), int(10e4), int(1e3)))

            self.lazy_list = L

        # For model selection hot-patching.
        self.externally_controlled = False
        self.ext_test_idx = None
        self.ext_pmf_idx = None

    @staticmethod
    def acquire():
        raise NotImplementedError

    def check_save(self, off=0):
        if self.all_pmfs is None:
            return False
        if self.cfg.lazy_save and (self.counter - off in self.lazy_list):
            return True
        else:
            return False

        return True

    def sample_pmf(self, pmf):
        """Sample from pmf."""

        if isinstance(pmf, float) or (len(pmf) == 1):
            # Always choose last datum
            pmf = [1]

        if self.externally_controlled:
            idx = self.ext_pmf_idx
            test_idx = self.ext_test_idx

        else:
            # print(self.cfg['sample'])
            if self.cfg['sample']:

                try:
                    # we don't want to have normalised over pool for the real
                    # importance sampling.. normalise again internally
                    pmf = np.array(pmf)
                    pmf_sample = pmf/pmf.sum()
                    # this is one-hot over all remaining test data
                    sample = np.random.multinomial(1, pmf_sample)
                except Exception as e:
                    logging.info(e)
                    logging.info(f'This was pmf {pmf}')
                    raise ValueError

                # idx in test_remaining
                idx = np.where(sample)[0][0]
            else:
                idx = np.argmax(pmf)

            # get index of chosen test datum
            test_idx = self.dataset.test_remaining[idx]

        # get value of acquisition function at that index
        self.weights = np.append(
            self.weights, pmf[idx])

        if self.check_save():
            self.all_pmfs.append(dict(
                idx=idx,
                test_idx=test_idx,
                pmf=pmf,
                remaining=self.dataset.test_remaining,
                observed=self.dataset.test_observed))

        self.counter += 1
        return test_idx, idx

    @staticmethod
    def safe_normalise(pmf):
        """If loss is 0, we want to sample uniform and avoid nans."""

        if (Σ := pmf.sum()) != 0:
            pmf /= Σ
        else:
            pmf = np.ones(len(pmf))/len(pmf)

        return pmf


class RandomAcquisition(AcquisitionFunction):
    def __init__(self, cfg, dataset, *args, **kwargs):
        super().__init__(cfg, dataset)

    def acquire(self, *args, **kwargs):
        n_remaining = len(self.dataset.test_remaining)
        pmf = np.ones(n_remaining)/n_remaining
        return self.sample_pmf(pmf)


class TrueLossAcquisition(AcquisitionFunction):
    def __init__(self, cfg, dataset, true_loss_vals, *args, **kwargs):
        super().__init__(cfg, dataset)

        # make sure indexes are aligned
        if (dataset.N < dataset.y.size) and dataset.cfg.get('with_unseen'):
            N = dataset.y.size
        else:
            N = dataset.N

        self.true_loss = np.zeros(N)
        self.true_loss[self.dataset.test_idxs] = true_loss_vals

    def acquire(self, *args, **kwargs):
        """Sample according to true loss dist."""

        pmf = self.true_loss[self.dataset.test_remaining]

        pmf = self.safe_normalise(pmf)

        return self.sample_pmf(pmf)

# --- Acquisition Functions Based on Expected Loss

class _LossAcquisitionBase(AcquisitionFunction):
    def __init__(self, cfg, dataset, model):

        super().__init__(cfg, dataset)

        # also save original model
        self.model = model

    def acquire(self, *args, **kwargs):
        # predict + std for both models on all remaining test points
        remaining_idxs = self.dataset.test_remaining
        remaining_data = self.dataset.x[remaining_idxs]

        # build expected loss
        expected_loss = self.expected_loss(remaining_data, remaining_idxs)
        # old = expected_loss
        # if len(self.dataset.test_observed) >= 1000:
            # a = 1
        # plt.figure(); plt.scatter(expected_loss, self.dataset.y[remaining_idxs], s=1, alpha=0.2); plt.plot(*2*[[self.dataset.y[remaining_idxs].min(), self.dataset.y[remaining_idxs].max()]], c='grey'); plt.yscale('log'); plt.xscale('log'); plt.savefig('tmp.png')
        # np.mean((expected_loss - self.dataset.y[remaining_idxs])**2) / self.surr_model.model.stats['y_std']**2
        # np.mean((self.dataset.y[self.dataset.test_observed] - self.dataset.y[remaining_idxs])**2) / self.surr_model.model.stats['y_std']**2

        if self.cfg['sample'] and (expected_loss < 0).sum() > 0:
            # Log-lik can be negative.
            # Make all values positive.
            # Alternatively could set <0 values to 0.
            expected_loss += np.abs(expected_loss.min())

        if np.any(np.isnan(expected_loss)):
            logging.warning(
                'Found NaN values in expected loss, replacing with 0.')
            logging.info(f'{expected_loss}')
            expected_loss = np.nan_to_num(expected_loss, nan=0)

        # this might fail for expected_loss.sum() == 0
        if expected_loss.sum() != 1:
            expected_loss = expected_loss / expected_loss.sum()

        if self.cfg.get('uniform_clip', False):
            # clip all values less than 10 percent of uniform propability
            p = self.cfg['uniform_clip_val']
            expected_loss = np.maximum(p * 1/expected_loss.size, expected_loss)
            expected_loss /= expected_loss.sum()

        if self.cfg.get('defensive', False):
            a = self.cfg['defensive_val']
            n = len(expected_loss)
            expected_loss = (
                a * np.ones(n)/n) + (1-a) * expected_loss
            # small numerical inaccuracies
            expected_loss /= expected_loss.sum()

        if np.all(np.isclose(expected_loss, 0)):
            logging.warning(
                'All exp losses 0, replace by random acq.')
            expected_loss = np.ones_like(expected_loss)
            expected_loss /= len(expected_loss)

        return self.sample_pmf(expected_loss)

    def get_aleatoric(self):
        if getattr(self.dataset.cfg, 'expose_aleatoric', True):
            ret = getattr(self.dataset, 'aleatoric', 0)
        else:
            ret = 0

        return ret


class _SurrogateAcquisitionBase(_LossAcquisitionBase):
    def __init__(self, cfg_run, dataset, model, SurrModel, surr_cfg):
        logging.info(
            f'**Initialising {self.__class__} with name {model.cfg["name"]}.')
        logging.info(f'Config {surr_cfg}')

        if surr_cfg.get('acquisition', False):
            # the surrogate acquisition can specialise the
            # acquisition configs. this mostly affects clipping behaviour
            cfg = OmegaConf.merge(
                OmegaConf.structured(cfg_run[0]),
                OmegaConf.structured(surr_cfg.acquisition))
            cfg_run = [cfg, cfg_run[1]]

        super().__init__(cfg_run, dataset, model)

        if surr_cfg.get('copy_main', False):
            self.surr_cfg = deepcopy(model.cfg)
        else:
            self.surr_cfg = deepcopy(surr_cfg)

        self.surr_class = SurrModel
        self.surr_model = SurrModel(self.surr_cfg)

        if len(self.dataset.train_idxs) > 0:
            logging.info('Initial surrogate fit b/c on train data.')
            self.surr_model.fit(*self.dataset.total_observed)
        else:
            logging.info('Skipping initial surrogate fit b/c no train data.')

        if self.surr_cfg.get('efficient', False):
            # make efficient predictions on remaining test data
            self.surr_model = make_efficient(self.surr_model, self.dataset)

        if surr_cfg.get('lazy', False):
            if (s := self.surr_cfg.get('lazy_schedule', False)) is not False:
                retrain = list(s)
            else:
                retrain = [5]
                retrain += list(range(10, 50, 10))
                retrain += [50]
                retrain += list(range(100, 1000, 150))
                retrain += list(range(1000, 10000, 2000))
                retrain += list(range(int(10e3), int(100e3), int(10e3)))

            # always remove 0, since we train at it 0
            # self.retrain = list(set(retrain) - {0})
            self.retrain = list(set(retrain))
            self.update_surrogate = self.lazy_update_surrogate
        else:
            self.update_surrogate = self.vanilla_update_surrogate

        if risk := self.surr_cfg.get('weights', False):
            assert self.dataset.cfg.test_proportion == 1, (
                'No weights for train data available.')
            assert not self.surr_cfg.get('on_train_only', False), (
                'No weights for train data available.')

            from ase.utils.maps import risk_estimator
            self.get_weights = risk_estimator[risk].get_weights

    def get_weight_kwargs(self):
        if self.surr_cfg.get('weights', False):
            kwargs = dict(sample_weight=self.get_weights(
                self.weights, self.dataset.N))
        else:
            kwargs = dict()
        return kwargs

    def vanilla_update_surrogate(self):
        logging.info(f'calling vanilla_update_surr in it {self.counter}')
        # from matplotlib import pyplot as plt; plt.figure(); plt.scatter(self.dataset.x, self.dataset.y, s=1, label='noised'); plt.scatter(self.dataset.x, self.dataset.f.pdf(self.dataset.x), s=1, label='true'); plt.scatter(self.dataset.x, self.surr_model.predict(self.dataset.x), s=1, label='fit'); plt.legend(); plt.savefig('tmp.png')
        # from matplotlib import pyplot as plt; plt.figure(); plt.scatter(self.dataset.x[:, 0], self.dataset.y, s=1, label='noised'); plt.scatter(self.dataset.x[:, 0], self.dataset.f.pdf(self.dataset.x), s=1, label='true'); plt.scatter(self.dataset.x[:, 0], self.surr_model.predict(self.dataset.x), s=1, label='fit'); plt.legend(); plt.savefig('tmp.png')

        is_fit = getattr(self.surr_model, 'is_fit', False)
        keep_params = self.surr_cfg.get('init_with_last_params', False)
        if keep_params and is_fit:
            params = self.surr_model.get_params()
        else:
            params = None

        self.surr_model = self.surr_class(self.surr_cfg, params=params)

        if self.surr_cfg.get('on_train_only', False):
            self.surr_model.fit(*self.dataset.train_data)
        else:
            # fit on all observed data
            self.surr_model.fit(
                *self.dataset.total_observed,
                **self.get_weight_kwargs())

        if self.surr_cfg.get('efficient', False):
            # make efficient predictions on remaining test data
            self.surr_model = make_efficient(self.surr_model, self.dataset)

    def lazy_update_surrogate(self):
        # logging.info('calling lazy_update_surrogate')
        if self.counter in self.retrain:
            self.vanilla_update_surrogate()
            # logging.info(
            #     f'>> Triggering lazy refit for {self.__class__}/'
            #     f'{self.surr_model.cfg["name"]} '
            #     f'of surrogate in it {self.counter}.')

            # logging.info(
                # f'>> Finish lazy refit of surrogate in it {self.counter}.')

    def acquire(self, update_surrogate=True):

        if update_surrogate:
            self.update_surrogate()

        return super().acquire()


class _SelfSurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        from ase.utils.maps import model as model_maps
        SurrModel = model_maps[model.cfg['name']]
        super().__init__(cfg, dataset, model, SurrModel, model_cfg)


class SelfSurrogateAcquisitionEntropy(
        _SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model)


class SelfSurrogateAcquisitionSurrogateEntropy(
        _SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.surr_model, self.surr_model)


class SelfSurrogateAcquisitionSurrogateMutualInformation(
        _SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):

        mutual_information = self.surr_model.predict(
            remaining_data, idxs=remaining_idxs, mutual_info=True)

        return mutual_information



class SelfSurrogateAcquisitionSurrogateWeightedBALD2(
        _SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):

        eps = 1e-19
        model_pred = self.model.predict(remaining_data, remaining_idxs)
        model_pred = np.clip(model_pred, eps, 1 - eps)
        model_pred /= model_pred.sum(axis=1, keepdims=True)

        surr_pred = np.exp(self.surr_model.joint_predict(
            remaining_data, remaining_idxs))
        surr_mean_pred = surr_pred.mean(0)
        weights = - np.log(model_pred)

        # sum over classes to get entropy
        entropy_average = - weights * surr_mean_pred * np.log(surr_mean_pred)
        entropy_average = entropy_average.sum(-1)

        # these are all probabilities
        # N_ensemble x N_data x N_classes
        weights = weights[np.newaxis, ...]
        average_entropy = - weights * surr_pred * np.log(surr_pred)
        average_entropy = np.sum(average_entropy, -1)
        average_entropy = np.mean(average_entropy, 0)

        bald = entropy_average - average_entropy

        return bald


class SelfSurrogateAcquisitionSurrogateMI(_SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):

        mutual_information = self.surr_model.predict(
            remaining_data, idxs=remaining_idxs, mutual_info=True)

        return mutual_information


class _AnySurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        from ase.utils.maps import model as model_maps
        SurrModel = model_maps[model_cfg.name]
        super().__init__(cfg, dataset, model, SurrModel, model_cfg)


class AnySurrogateAcquisitionEntropy(
        _AnySurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg)


class ClassifierAcquisitionEntropy(_LossAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):

        if model_cfg is not None and model_cfg.get('acquisition', False):
            _cfg = OmegaConf.merge(
                OmegaConf.structured(cfg[0]),
                OmegaConf.structured(model_cfg.acquisition))
            cfg = [_cfg, cfg[1]]

        super().__init__(cfg, dataset, model)
        logging.info(f'Config {cfg}.')
        self.T = model_cfg.get('temperature', None)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, None, T=self.T,
            cfg=self.cfg)


def entropy_loss(
        remaining_data, remaining_idxs, model, surr_model=None,
        eps=1e-15, T=None, cfg=None, extra_log=False):

    model_pred = model.predict(remaining_data, idxs=remaining_idxs)

    if T is not None:
        model_pred = np.exp(np.log(model_pred)/T)

        model_pred = np.clip(model_pred, eps, 1/eps)
        model_pred[np.isnan(model_pred)] = 1/eps

        model_pred /= model_pred.sum(axis=1, keepdims=True)

        model_pred = np.clip(model_pred, eps, 1/eps)
        model_pred[np.isnan(model_pred)] = 1/eps

    if surr_model is not None:
        surr_model_pred = surr_model.predict(
            remaining_data, idxs=remaining_idxs)

        if T is not None:
            surr_model_pred = np.exp(np.log(surr_model_pred)/T)
            surr_model_pred = np.clip(surr_model_pred, eps, 1/eps)
            surr_model_pred[np.isnan(surr_model_pred)] = 1/eps

            surr_model_pred /= surr_model_pred.sum(axis=1, keepdims=True)
            surr_model_pred = np.clip(surr_model_pred, eps, 1/eps)
            surr_model_pred[np.isnan(surr_model_pred)] = 1/eps

    else:
        surr_model_pred = model_pred

    if T is None:
        model_pred = np.clip(model_pred, eps, 1 - eps)
        model_pred /= model_pred.sum(axis=1, keepdims=True)

    # Sum_{y=c} p_surr(y=c|x) log p_model(y=c|x)
    if not extra_log:
        res = -1 * (surr_model_pred * np.log(model_pred)).sum(-1)
    else:
        raise NotImplementedError('Not sure what this should look like')
        res = -1 * (surr_model_pred * np.log(model_pred)).sum(-1)

    if T is not None:
        res[np.isnan(res)] = np.nanmax(res)

    if cfg is not None and not cfg.get('uniform_clip', False):
        clip_val = np.percentile(res, 10)
        res = np.clip(res, clip_val, 1/eps)

    # clipping has moved to after acquisition
    return res
