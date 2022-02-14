import numpy as np
from types import MethodType


def make_efficient(model, dataset):
    """Model is constant over acquisition.

    I think this is called on an efficient model.

    Exploit this for efficiency gains.
    Predict on all unobserved test data once at the beginning of training.
    Then, when predict is called, just regurgitate these predictions.

    If make_efficient is called twice, the model will predict again!
    """
    # idxs = dataset.test_remaining
    # x = dataset.x[idxs]
    if (dataset.N < dataset.y.size) and dataset.cfg.get('with_unseen'):
        N = dataset.y.size
    else:
        N = dataset.N

    # we still just want to predict over the test set idxs

    idxs = dataset.test_idxs
    x = dataset.x[idxs]

    if getattr(model, 'efficient_instance', False):
        if model.cfg.task_type == 'regression':
            out = model.real_predict(x, return_std=True)
        else:
            out = model.real_predict(x)
    else:
        if model.cfg.task_type == 'regression':
            out = model.predict(x, return_std=True)
        else:
            out = model.predict(x)
        model = EfficientModel(model)

    if isinstance(out, tuple):
        # Handle with std
        out = list(out)
        if out[0].ndim == 1:
            preds = np.zeros(N)
            stds = np.zeros(N)
        else:
            preds = np.zeros((N, out[0].shape[1]))
            stds = np.zeros((N, out[1].shape[1]))

        preds[idxs] = out[0]
        stds[idxs] = out[1]
        model.test_predictions = preds
        model.test_predictions_std = stds
    else:
        if out.ndim == 1:
            preds = np.zeros(N)
        else:
            preds = np.zeros((N, out.shape[1]))
        preds[idxs] = out
        model.test_predictions = preds
        model.test_predictions_std = None

    if getattr(model.model, 'has_mi', False):
        mis = np.zeros(N)
        mi = model.model.predict(x, mutual_info=True)
        mis[idxs] = mi
        model.test_predictions_mi = mis
    else:
        model.test_predictions_mi = None

    c1 = getattr(model.model, 'joint_predict', False)
    # option to disable
    c2 = model.model.cfg.get('joint_predict', False)
    if c1 and c2:
        cfg = model.model.cfg
        if model.model.__class__.__name__ == 'RadialBNN':
            n_models = cfg.testing_cfg.variational_samples
        else:
            n_models = cfg.n_models
        joint_preds = np.zeros((n_models, N, cfg.num_classes))
        preds = model.model.joint_predict(x)
        joint_preds[:, idxs, :] = preds
        model.test_joint_predictions = joint_preds
    else:
        model.test_joint_predictions = None

    if hasattr(model.model, 'has_uncertainty'):
        model.has_uncertainty = model.model.has_uncertainty

    return model


class EfficientModel():

    def __init__(self, model):
        self.model = model
        self.cfg = self.model.cfg
        self.efficient_instance = True

    def fit(self, *args, **kwargs):
        if self.cfg.get('keep_constant', False):
            print('debug: no refitting, is efficient')
            pass
        else:
            return self.model.fit(self, *args, **kwargs)

    def real_fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.efficient_predict(*args, **kwargs)

    def real_predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def joint_predict(self, *args, **kwargs):
        return self.efficient_joint_predict(*args, **kwargs)

    def real_joint_predict(self, *args, **kwargs):
        return self.model.joint_predict(*args, **kwargs)

    def efficient_predict(
            self, data, idxs, return_std=False, mutual_info=False,
            no_lazy=False, *args, **kwargs):

        if no_lazy or (idxs is None):
            self.real_predict(
                data, *args, return_std=return_std,
                mutual_info=mutual_info, **kwargs)

        if return_std and self.test_predictions_std is not None:
            return (self.test_predictions[idxs],
                    self.test_predictions_std[idxs])

        elif mutual_info and self.test_predictions_mi is not None:
            return self.test_predictions_mi[idxs]

        else:
            return self.test_predictions[idxs]

    def efficient_joint_predict(
            self, data, idxs, no_lazy=False, *args, **kwargs):

        if no_lazy:
            self.real_joint_predict(data, *args, **kwargs)

        else:
            return self.test_joint_predictions[:, idxs, :]
