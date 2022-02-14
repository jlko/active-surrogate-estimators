from copy import deepcopy
from pathlib import Path
import logging
import math
import numpy as np
import torch
from omegaconf import OmegaConf
import hydra

from scipy.special import softmax

from .radial_bnn import bnn_models
from .sk2torch import SK2TorchBNN
from .cnn.models import DeepModel
from .skmodels import BaseModel

from ase.utils.data import to_json, from_json
from ase.utils.calibration_library import calibrate

# ---- Final Pytorch Wrappers ----
# Import these models. From the outside, they can be used like Sklearn models.


def modify_bnn(model, data_CHW, *args, **kwargs):

    class Sanify(model):
        """Change default behaviour of Radial BNN.

        In particular, hide sampling behaviour and special input/output
        formatting, s.t. SaneRadialBNN behaves as a normal Pytorch model.

        For calibrating the BNN (i.e. when with_log_softmax=False) is passed,
        we also just calibrate the log predictions instead, identical to how
        we deal with calibrating the ensemble.
        """
        def __init__(self, *args, **kwargs):
            self.data_CHW = data_CHW
            super().__init__(*args, **kwargs)
            self.log_softmax = torch.nn.LogSoftmax(dim=1)

        def forward(
                self, data, n_samples, log_sum_exp=True, T=None,
                *args, **kwargs):

            data = self.radial_bnn_forward_reshape(data, n_samples)
            out = super().forward(data)

            if log_sum_exp:
                # average of log probabilities
                # each prediction already is a log probability
                out = torch.logsumexp(out, dim=1) - math.log(n_samples)

                if (T is not None) and (T != 1):
                    # no temperature scaling for samples
                    # we are only calibrating the average prediction
                    out = self.log_softmax(out / T)

            return out

        def radial_bnn_forward_reshape(self, data_N_HW, n_samples):
            # expects empty channel dimension after batch dim
            data_N_C_HW = torch.unsqueeze(data_N_HW, 1)

            if self.data_CHW is None:
                data_N_C_H_W = data_N_C_HW
            else:
                # Radial BNN and RGB Data actually does not work yet
                data_N_C_H_W = data_N_C_HW.reshape(
                    list(data_N_C_HW.shape[:-1]) + list(self.data_CHW[1:]))

            # assert len(data_N_C_H_W.shape) == 4
            data_N_V_C_H_W = torch.unsqueeze(data_N_C_H_W, 1)
            data_N_V_C_H_W = data_N_V_C_H_W.expand(
                -1, n_samples, -1, -1, -1
            )
            return data_N_V_C_H_W

    return Sanify(*args, **kwargs)


class RadialBNN(SK2TorchBNN):
    def __init__(self, cfg, *args, **kwargs):
        data_CHW = cfg.get('data_CHW', None)
        kwargs = dict(channels=cfg['channels'])
        model = modify_bnn(bnn_models.RadialBNN, data_CHW, **kwargs)
        self.has_mi = True
        super().__init__(model, cfg)


class TinyRadialBNN(SK2TorchBNN):
    def __init__(self, cfg, *args, **kwargs):
        data_CHW = cfg.get('data_CHW', None)
        model = modify_bnn(bnn_models.TinyRadialBNN, data_CHW)
        super().__init__(model, cfg)
        self.has_mi = True


def modify_cnns(model, data_CHW, debug_mnist):

    class Sanify(model):
        """Change default behaviour of Deterministic CNNs.

        Make them ignore args, kwargs in forward pass.
        """
        def __init__(self, *args, **kwargs):
            self.data_CHW = list(data_CHW)
            self.debug_mnist = debug_mnist
            super().__init__(*args, **kwargs)

            # original model uses Crossentropy loss
            # we use NLL loss --> need to add logsoftmax layer
            self.log_softmax = torch.nn.LogSoftmax(dim=1)

        def forward(self, data, T=1, with_log_softmax=True, *args, **kwargs):
            N = data.shape[0]
            data = data.reshape([N]+self.data_CHW)
            if self.debug_mnist:
                data = data.repeat(1, 3, 1, 1)

            out = super().forward(data)
            if with_log_softmax:
                out = self.log_softmax(out/T)

            return out

    return Sanify


class ResNet18(SK2TorchBNN):
    def __init__(self, cfg, *args, **kwargs):
        model = modify_cnns(DeepModel, cfg['data_CHW'], cfg['debug_mnist'])(
            cfg['data_CHW'][-1], cfg['num_classes'], 'resnet18')
        super().__init__(model, cfg)


class ResNet34(SK2TorchBNN):
    def __init__(self, cfg, *args, **kwargs):
        model = modify_cnns(DeepModel, cfg['data_CHW'], cfg['debug_mnist'])(
            cfg['data_CHW'][-1], cfg['num_classes'], 'resnet34')
        super().__init__(model, cfg)


class ResNet50(SK2TorchBNN):
    def __init__(self, cfg, *args, **kwargs):
        model = modify_cnns(DeepModel, cfg['data_CHW'], cfg['debug_mnist'])(
            cfg['data_CHW'][-1], cfg['num_classes'], 'resnet50')
        super().__init__(model, cfg)


class ResNet101(SK2TorchBNN):
    def __init__(self, cfg, *args, **kwargs):
        model = modify_cnns(DeepModel, cfg['data_CHW'], cfg['debug_mnist'])(
            cfg['data_CHW'][-1], cfg['num_classes'], 'resnet101')
        super().__init__(model, cfg)


class WideResNet(SK2TorchBNN):
    def __init__(self, cfg, *args, **kwargs):
        model = modify_cnns(DeepModel, cfg['data_CHW'], cfg['debug_mnist'])(
            cfg['data_CHW'][-1], cfg['num_classes'], 'wideresnet')
        super().__init__(model, cfg)


class TorchEnsemble(BaseModel):
    def __init__(self, cfg, TorchModel, *args, **kwargs):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if cfg.get('calibrated', False):
            self.calibrated = True
            logging.info(
                'Global calibration for ensemble enabled. Disabling locally.')
        else:
            self.calibrated = False
            logging.info(
                'Global calibration disabled for ensemble')

        self.T = None

        n_models = cfg['n_models']
        self.models = []
        for i in range(n_models):
            # update model save path
            if cfg.get('skip_fit_debug', False):
                cfg_i = OmegaConf.merge(
                    OmegaConf.structured(cfg),
                    dict(
                        save_path=cfg.save_path.format(i),
                        skip_fit_debug=cfg.skip_fit_debug.format(i),
                    ),
                    )
            else:
                cfg_i = cfg

            cfg_i = deepcopy(cfg_i)
            cfg_i.calibrated = False

            model = TorchModel(cfg_i)
            self.models.append(model)

        self.log_softmax = self.models[0].model.log_softmax

        super().__init__(cfg, None)

    def eval(self):
        for model in self.models:
            model.model.eval()

    def train(self):
        for model in self.models:
            model.train()

    def __call__(self, *args, **kwargs):
        # this is only used in calibrate right now
        # I need to make this return the log probabilities as prediction
        # despite what the call in set_temperature wants
        # that is because, for the ensemble we do
        # softmax(ensemble prediction) (which is itself already softmaxed)
        # see https://arxiv.org/pdf/2101.05397.pdf
        # so I don't want to apply the softmax twice, this will peaken things
        # instead, it seems more natural to apply the softmax to
        # the log predictions

        preds = []
        for model in self.models:
            pred = model.predict(*args, to_numpy=False, **kwargs)
            preds.append(pred)

        # don't need to detach/cpu/numpy b/c local predicts do that already
        preds = torch.stack(preds, 0)
        mean_preds = torch.mean(preds, 0)
        if kwargs.get('with_log_softmax', True):
            return mean_preds
        else:
            # I was experiencing nans. Maybe this helps.
            mean_preds = torch.maximum(mean_preds, torch.tensor(1e-18))
            mean_preds = torch.log(mean_preds)
            return mean_preds

    def predict(self, *args, **kwargs):
        preds = []
        for model in self.models:
            pred = model.predict(*args, **kwargs)
            preds.append(pred)

        # don't need to detach/cpu/numpy b/c local predicts do that already
        preds = np.stack(preds, 0)
        mean_preds = np.mean(preds, 0)

        if self.calibrated:
            if self.T is None:
                raise ValueError('Calibrated enabled but no T found.')
            else:
                T = self.T
                mean_preds = np.maximum(mean_preds, 1e-18)
                log_mean_preds = np.log(mean_preds)
                mean_preds = softmax(log_mean_preds / T, axis=1)

        return mean_preds

    def joint_predict(self, *args, **kwargs):

        preds = []
        for model in self.models:
            pred = model.predict(*args, **kwargs)
            preds.append(pred)
        return np.stack(preds, 0)

    def fit(self, x, y):
        for model in self.models:
            model.fit(x, y)

        if self.calibrated:
            logging.info('Calibrating models globally.')
            self.calibrate(x, y)

    def calibrate(self, x, y):
        """I am temperature scaling as softmax((log p_i)/T)).
        Where log p_i are the log predictions of the ensemble.
        And p_i are obtained as 1/K sum_k p(y|x, k) for ensemble components k.

        """

        if self.cfg.get('skip_fit_debug_relative', False):
            base = Path('.')
        else:
            base = Path(hydra.utils.get_original_cwd())

        load_calibration = self.cfg.get(
            'load_calibration', True)

        tpath = base / self.cfg.get(
                'temp_skip_fit_debug', 'ensemble_temperature.json')

        if load_calibration and tpath.exists():
            results = from_json(tpath)
            T = results['T']
            logging.info(f'Loaded T={T} from {tpath}.')
            logging.info(results)
            self.T = T
        else:
            logging.info(
                f'Calibrating fresh b/c {tpath} does not exist or '
                f'load_calibration is false (is {load_calibration}).')
            _, val_loader = self.models[0].val_train_loaders(x, y)

            failure = True
            it = 0
            lr = 0.01
            while failure:

                calibration_results = calibrate.set_temperature(
                    self, val_loader, self.device, lr=lr*(0.3)**it)

                failure = calibration_results['failure']
                it += 1

                if it > 10:
                    raise ValueError('Calibration failed too often.')

            self.T = calibration_results['T']
            logging.info(f'Set temperature to {self.T} after calibration.')

        temp_save_path = Path(
            self.cfg.get('temp_save_path', 'ensemble_temperature.json'))

        if not temp_save_path.exists():
            logging.info(f'Saving calibration Temp. to {temp_save_path}.')
            to_json(calibration_results, temp_save_path)


class ResNet18Ensemble(TorchEnsemble):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, ResNet18)


class ResNet34Ensemble(TorchEnsemble):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, ResNet34)


class ResNet50Ensemble(TorchEnsemble):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, ResNet50)


class ResNet101Ensemble(TorchEnsemble):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, ResNet101)

class WideResNetEnsemble(TorchEnsemble):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, WideResNet)
