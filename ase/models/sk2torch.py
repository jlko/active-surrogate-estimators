"""Make PyTorch models work with SKLearn interface."""
import os
import logging
from pathlib import Path
import copy
import hydra
from omegaconf import OmegaConf

import math
import numpy as np
import torch
from torch._C import Value
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split

from ase.utils.data import to_json, from_json
from .radial_bnn.radial_layers.loss import Elbo
from .skmodels import BaseModel
from ase.utils.calibration_library import calibrate


# ---- Interface between SKLearn and Pytorch ----
# Make Pytorch model behave as SKLearn model on the outside.
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, chw, transform):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.chw = chw

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = x.reshape(self.chw).transpose(1, 2, 0)
            x = self.transform(x)
            x = x.reshape(-1)

        return x, y

    def __len__(self):
        return len(self.data)


class SK2TorchBNN(BaseModel):
    """Interface for Pytorch Models and SKlearn Methods."""
    def __init__(self, model, cfg, *args, **kwargs):
        logging.info(f'Initialising new model {cfg.name}.')

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if not hasattr(cfg, 'task_type'):
            cfg = OmegaConf.merge(
                OmegaConf.structured(cfg),
                dict(task_type='classification',))

        self.cfg = cfg
        self.t_cfg = cfg['training_cfg']
        self.model = model.to(device=self.device).type(
            torch.float32)
        self.T = None
        self.needs_reinit = True

        if self.t_cfg.get('standardize', False):
            stats = dict(x_mean=0., x_std=1., y_mean=0., y_std=1.)
            stats = {
                i: torch.tensor(j).to(self.device)
                for i, j in stats.items()}
            self.stats = stats

    def predict(self, x, for_acquisition=True, mutual_info=False,
                check_calibrated=True, to_numpy=True,
                log_sum_exp=True, n_samples=None,
                *args, **kwargs):

        if n_samples is None:
            n_samples = self.t_cfg.get('variational_samples', None)
        if isinstance(x, torch.Tensor):
            loader = [[x, ]]
        elif len(x) > self.t_cfg['batch_size']:
            loader = self.make_loader(
                [x], train=False, persistent=False, pin_memory=False,
                num_workers=0)
        elif isinstance(x, np.ndarray):
            loader = [[torch.from_numpy(x)]]
        else:
            raise ValueError

        if for_acquisition:
            self.model.eval()

        if check_calibrated and self.cfg.get('calibrated', False):
            if self.T is None:
                raise ValueError('Calibrated enabled but no T found.')
            else:
                T = self.T
        else:
            T = 1

        preds = []
        stats=getattr(self, 'stats', 'none')
        kwargs = dict(n_samples=n_samples, T=T, stats=stats)
        with torch.no_grad():
            for (data, ) in loader:
                data = data.to(self.device)
                if not mutual_info:
                    pred = self.model(data, log_sum_exp=log_sum_exp, **kwargs)
                    # model outputs log probabilities, our code does not expect
                    # this the additional exp_logging hopefully does not
                    # introduce too much error
                    if self.cfg.task_type == 'classification':
                        pred = torch.exp(pred)
                    if self.t_cfg.get('standardize', False):
                        pred = self.apply_pred_std(pred)
                else:
                    # N x Samples x Classes
                    out = self.model(data, log_sum_exp=False, **kwargs)
                    if self.t_cfg.get('standardize', False):
                        pred = self.apply_pred_std(pred)

                    mean_samples = torch.logsumexp(out, dim=1) - math.log(
                        n_samples)

                    entropy_average = -torch.sum(
                        mean_samples.exp() * mean_samples, dim=1)

                    average_entropy = -torch.sum(
                        out.exp() * out, dim=2).mean(1)

                    mi = entropy_average - average_entropy

                    pred = mi

                preds.append(pred)

        preds = torch.cat(preds, 0)
        del loader

        if to_numpy:
            return preds.detach().cpu().numpy()
        else:
            return preds

    def joint_predict(
            self, x, for_acquisition=True,
            to_numpy=True, n_samples=None,
            *args, **kwargs):

        if n_samples is None:
            n_samples = self.cfg.testing_cfg.get('variational_samples', None)
        if isinstance(x, torch.Tensor):
            loader = [[x, ]]
        elif len(x) > self.t_cfg['batch_size']:
            loader = self.make_loader(
                [x], train=False, persistent=False, pin_memory=False,
                num_workers=0)
        elif isinstance(x, np.ndarray):
            loader = [[torch.from_numpy(x)]]
        else:
            raise ValueError

        if for_acquisition:
            self.model.eval()

        preds = []

        kwargs = dict(n_samples=n_samples)
        with torch.no_grad():
            for (data, ) in loader:
                data = data.to(self.device)
                # N x Samples x Classes
                out = self.model(data, log_sum_exp=False, **kwargs)

                preds.append(out)

        preds = torch.cat(preds, 0)
        preds = torch.permute(preds, (1, 0, 2))

        if to_numpy:
            return preds.detach().cpu().numpy()
        else:
            return preds

    def fit(self, x, y):
        """
        How to use the calibration feature?

        Set `cfg.calibrated=True` to enable calibration.
        If you want to load a temperature from a previous experiment, set
        `cfg.temp_skip_fit_debug` to that path. (This will respect if you
        have `cfg.skip_fit_debug_relative` enabled.)

        Further, you can set a `cfg.temp_save_path` (e.g. needed for ensembles)
        to save the temperature in a desired location, relative to the current
        hydra logging dir.

        """

        # don't fit model and load instead (but only if chkpt exists)
        p = self.cfg['skip_fit_debug']

        # set up base bath
        if self.cfg.get('skip_fit_debug_relative', False):
            base = Path('.')
        else:
            base = Path(hydra.utils.get_original_cwd())

        # load or fit model
        if p and (loc := (base / p)).exists():
            logging.info(f'Loading model from {p}.')
            self.model.load_state_dict(
                torch.load(loc, map_location=self.device))
        else:
            if p and not (loc := (base / p)).exists():
                logging.info(f'Tried to load model, but {p} does not exist.')

            if self.t_cfg.get('standardize', False):
                self.stats = self.standardize(x, y)

            train_loader, val_loader = self.val_train_loaders(x, y)
            # from ase.datasets import get_CIFAR10
            # train_loader, val_loader = get_CIFAR10()
            if train_loader or val_loader:
                self.model = self.train_to_convergence(
                    self.model, train_loader, val_loader)
                # try to avoid errors
                del train_loader
                del val_loader

        path = Path(self.cfg.get('save_path', 'model.pth'))

        if not path.exists():
            logging.info(f'Saving model to {path}.')
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), path)

        # enable/disable calibration. if enabled load if available.
        calibrated = self.cfg.get('calibrated', False)
        if not calibrated:
            logging.info('Calibration disabled.')
        else:
            load_calibration = self.cfg.get(
                'load_calibration', True)
            tpath = base / self.cfg.get(
                    'temp_skip_fit_debug', 'temperature.json')
            if load_calibration and tpath.exists():
                calibration_results = from_json(tpath)
                T = calibration_results['T']
                logging.info(f'Loaded T={T} from {tpath}.')
                logging.info(calibration_results)
                self.T = T
            else:
                logging.info(
                    f'Calibrating fresh b/c {tpath} does not exist or '
                    f'load_calibration is false (is {load_calibration}).')
                _, val_loader = self.val_train_loaders(x, y)
                failure = True
                it = 0
                lr = 0.01
                while failure:
                    calibration_results = calibrate.set_temperature(
                        self.model, val_loader, self.device,
                        n_samples=self.t_cfg.get('variational_samples', None),
                        lr=lr*(0.3)**it)
                    failure = calibration_results['failure']
                    it += 1

                    if it > 10:
                        raise ValueError('Calibration failed too often.')

                self.T = calibration_results['T']
                logging.info(f'Set temperature to {self.T} after calibration.')

        temp_save_path = Path(
            self.cfg.get('temp_save_path', 'temperature.json'))

        if calibrated and not temp_save_path.exists():
            logging.info(f'Saving calibration Temp. to {temp_save_path}.')
            to_json(calibration_results, temp_save_path)

        logging.info('Finished fitting.')

    def standardize(self, x, y):
        # I don't really care about the fact that I'm standardising
        # using the val set as well. this might taint my val set
        # but who cares
        stats = dict(
            x_mean=x.mean(0),
            x_std=x.std(0),
            y_mean=y.mean(),
            y_std=y.std()
            )
        stats = {
            i: torch.tensor(j).to(self.device)
            for i, j in stats.items()}

        # when to apply stds? maybe in forward of model?
        return stats

    def val_train_loaders(self, x, y):
        skipping = lambda _: logging.info(
                    f'Not enough data to create train, val split. '
                    f'Skipping training.')

        if len(x) == 0:
            skipping()
            return False, False
        val_idxs = self.cfg.get('val_idxs', False)
        val_size = int(self.t_cfg['validation_set_size'])

        # given val_idxs
        if val_idxs:
            if len(val_idxs) == 0:
                train_only = True
            else:
                train_only = False
                x_val, y_val = x[val_idxs], y[val_idxs]
                train_idxs = np.setdiff1d(np.arange(0, len(x)), val_idxs)
                assert np.intersect1d(train_idxs, val_idxs).size == 0
                assert len(train_idxs) + len(val_idxs) == len(y)
                x_train, y_train = x[train_idxs], y[train_idxs]
                logging.info(
                    f'Using fixed val set w/ len {len(y_val)} for {self}.')
                logging.debug('NEED TO CHECK THAT VAL IDXS ARE STILL VALID')
        else:
            if val_size == 0:
                train_only = True
            else:
                train_only = False
                # generate random splits for train and val
                if len(x) < val_size + 1:
                    skipping()
                    # (need at least 1 sample in training)
                    return False, False

                if self.t_cfg.get('stratify_val', False):
                    strata = y
                else:
                    strata = None

                # Just added random seed. Prev. no idea how this split happened
                # it's also important that these are constant for constant data
                # b/c otherwise ensembles will get different train/val splits
                # which would lead to some information leak for the calibration
                # actually, isn't it kinda neat if we use different information
                # for the different splits?
                # --> well I guess then we would have to start counting the
                # validation set. if anyone ever asks. this is an easy way
                # to get rid of it. let's not do it for now

                # try:
                #     x_train, x_val, y_train, y_val = train_test_split(
                #         x, y, test_size=val_size, stratify=strata,
                #         random_state=0)
                # except Exception as e:
                #     logging.info('shitty except trigggered')
                #     logging.info(e)
                #     x_train, x_val, y_train, y_val = train_test_split(
                #         x, y, test_size=val_size,
                #         random_state=0)

                idxs = np.arange(0, len(x))
                try:
                    # set to NONE! here?!?!?
                    (
                        x_train, x_val,
                        y_train, y_val,
                        i_train, i_val) = train_test_split(
                            x, y, idxs, test_size=val_size, stratify=strata,
                            random_state=0)

                except Exception as e:
                    logging.info(e)
                    logging.info('COULD NOT STRATIFY VAL SET. Retry without.')
                    x_train, x_val, y_train, y_val, _, i_val = train_test_split(
                        x, y, idxs, test_size=val_size, stratify=None,
                        random_state=0)

                self.val_idxs = i_val

        if train_only:
            train_loader = self.make_loader([x, y])
            return train_loader, False

        else:
            if self.cfg.task_type == 'classification':
                logging.info(f'Unique labels in val {np.bincount(y_val)}')
                logging.info(f'Unique labels in train {np.bincount(y_train)}')

            train_loader = self.make_loader([x_train, y_train])
            val_loader = self.make_loader([x_val, y_val], train=False)

            return train_loader, val_loader

    def make_loader(
            self, arrs, train=True, persistent=None, pin_memory=None,
            num_workers=None):

        if train and (self.t_cfg.get('transforms', False) == 'cifar'):
            # transform = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.RandomCrop(self.cfg.data_CHW[-1], padding=4),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor()
                # ])

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(self.cfg.data_CHW[-1], padding=4),
                transforms.RandomHorizontalFlip(),
                ])
            dataset = TransformDataset(
                arrs[0], arrs[1], list(self.cfg.data_CHW), transform)
        else:
            arrs = [torch.from_numpy(arr) for arr in arrs]
            dataset = torch.utils.data.TensorDataset(*arrs)

        bs = self.cfg.get('testing_cfg', dict())
        if bs is None:
            bs = False
        else:
            bs = bs.get('batch_size', False)

        if bs and not train:
            batch_size = bs
        else:
            batch_size = self.t_cfg['batch_size']

        # torch does this automatically
        # batch_size = min(arrs[0].shape[0], batch_size)
        if persistent is None:
            persistent = self.t_cfg.get('persistent_workers', False)
        if pin_memory is None:
            pin_memory = self.t_cfg['pin_memory']
        if num_workers is None:
            num_workers = self.t_cfg['num_workers']

        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=train,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent,
            # num_workers=0,
            # persistent_workers=False)
            )

        return data_loader

    # def make_loader(self, arrs, train=True):

    #     if train and (self.t_cfg.get('transforms', False) == 'cifar'):
    #         transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             ])

    #         dataset = TransformDataset(
    #             arrs[0], arrs[1], list(self.cfg.data_CHW), transform)
    #     else:
    #         arrs = [torch.from_numpy(arr) for arr in arrs]
    #         dataset = torch.utils.data.TensorDataset(*arrs)

    #     data_loader = torch.utils.data.DataLoader(
    #         dataset,
    #         shuffle=train,
    #         batch_size=self.t_cfg['batch_size'],
    #         num_workers=self.t_cfg['num_workers'],
    #         pin_memory=self.t_cfg['pin_memory'])

    #     return data_loader

    def train_to_convergence(self, model, train_loader, val_loader):
        train_points = len(train_loader.dataset)
        val_points = len(val_loader.dataset) if val_loader else 0
        logging.info(
            f'Training with {train_points} train and {val_points} val points.')

        m = self.t_cfg['max_epochs']
        log_every = int(.02 * m) if m > 100 else 1
        best = np.inf
        best_model = model
        patience = 0
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        for epoch in range(self.t_cfg['max_epochs']):
            log_epoch = epoch % log_every == 0

            train_loss = self.train(
                model, train_loader, optimizer, return_loss=log_epoch)

            val_nll, val_accuracy = self.evaluate(
                model, val_loader, check_calibrated=False)

            if log_epoch:
                log_str = f'Epoch {epoch:0>3d}: '
                log_str = log_str + f'Train loss: {train_loss:.4f}, '
                if val_loader:
                    log_str = log_str + f'Val nll: {val_nll:.4f}, '
                    log_str = log_str + f'Val Accuracy: {val_accuracy}'
                logging.info(log_str)
            if val_loader:
                if val_nll <= best:
                    best = val_nll
                    best_model = copy.deepcopy(model)
                    patience = 0
                else:
                    patience += 1
                if patience >= self.t_cfg['early_stopping_epochs']:
                    logging.info(
                        f'Patience reached - stopping training. '
                        f'Best was {best}')
                    break

            if scheduler is not None:
                scheduler.step()

        logging.info('Completed training for acquisition.')
        if val_loader:
            return best_model
        else:
            return model

    def train(self, model, train_loader, optimizer, return_loss=False):
        n_samples = self.t_cfg.get('variational_samples', None)

        loss_fn = self.get_loss('train', model, train_loader)

        stats = getattr(self, 'stats', 'none')
        kwargs = dict(n_samples=n_samples, log_sum_exp=False, stats=stats)
        if return_loss:
            losses = []

        model.train()
        for idx, (data, target) in enumerate(train_loader):

            data = data.to(self.device)
            target = target.to(self.device)
            if self.t_cfg.get('standardize', False):
                # apply to target not pred, I don't want to mess with gradients
                target = self.apply_target_std(target)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(data, **kwargs)
            loss = loss_fn(prediction, target)
            loss.backward()
            optimizer.step()
            if return_loss:
                losses.append(loss.detach())

        if return_loss:
            return (torch.mean(torch.stack(losses))).cpu().numpy()

    def evaluate(
            self, model, eval_loader, check_calibrated=True,
            model_train=True, *args, **kwargs):

        if eval_loader is False:
            # I've changed the early stop to update on "<="
            return 0, 0

        if model_train:
            self.model.train()
            n_samples = False
        else:
            self.model.eval()
            n_samples = self.cfg.get('testing_cfg', dict()).get(
                'variational_samples', False)

        if not n_samples:
            n_samples = self.t_cfg.get('variational_samples', None)

        # We actually only want eval mode on when we're doing acquisition
        # because of how consistent dropout works.
        # [Update] I'm no longer sure on this one. Why is that now?
        # Because we're using this for early stopping?
        # --> NO, I think this is related to the BNN model.
        # (where for consistent dropout we want to disable stochasticity
        # by keeping dropout masks consistent between inputs of a batch...
        # at least that's what I think)
        # --> but this is not true for ensemble model!?!
        # --> for the ensemble model we want eval enabled here!!
        # hmm... but I am only using this for early stopping
        # what kind of stochasticity is there anyways in my model
        # --> BATCH NORM!

        nll = correct = 0
        loss_fn = self.get_loss(None, None, 'test')

        if check_calibrated and self.cfg.get('calibrated', False):
            if self.T is None:
                raise ValueError('Calibrated enabled but no T found.')
            else:
                T = self.T
        else:
            T = 1

        stats = getattr(self, 'stats', 'none')
        kwargs = dict(n_samples=n_samples, T=T, stats=stats)

        with torch.no_grad():
            for data, target in eval_loader:

                data = data.to(self.device)
                target = target.to(self.device)

                if self.t_cfg.get('standardize', False):
                    # apply to target not pred, this way MSE is familiar
                    target = self.apply_target_std(target)

                prediction = model(data, **kwargs)

                raw_nll_N = loss_fn(
                    prediction, target, reduction='sum')

                nll += torch.sum(raw_nll_N)

                # get the index of the max log-probability
                if self.cfg.task_type == 'classification':
                    class_prediction = prediction.max(1, keepdim=True)[1]
                    correct += class_prediction.eq(
                        target.view_as(class_prediction)).sum().item()

        nll /= len(eval_loader.dataset)
        percentage_correct = 100.0 * correct / len(eval_loader.dataset)

        return nll.item(), percentage_correct

    def get_optimizer(self):
        c = self.t_cfg.get('optimizer', False)

        if not c or c == 'adam':
            optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.t_cfg['learning_rate'],
                    weight_decay=self.t_cfg['weight_decay'],)

        elif c == 'cifar':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.t_cfg['learning_rate'],
                momentum=0.9, weight_decay=self.t_cfg['weight_decay'])
        else:
            raise ValueError

        return optimizer

    def get_scheduler(self, optimizer):
        c = self.t_cfg.get('scheduler', False)
        epochs = self.t_cfg['max_epochs']

        if not c:
            scheduler = None

        elif c == 'cifar':
            milestones = [int(epochs * 0.5), int(epochs * 0.75)]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.1)

        elif c == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)

        elif c == 'devries':
            # https://arxiv.org/abs/1708.04552v2
            # assert epochs == 200
            milestones = [60, 120, 160]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.2)

        return scheduler

    def get_loss(self, mode, model=None, train_loader=None):
        # model and train_loader only needed for radial BNN
        if mode == 'train' and self.t_cfg.get('model', False) == "radial_bnn":
            loss_object = Elbo(binary=False, regression=False)
            loss_object.set_model(model, train_loader.batch_size)
            loss_object.set_num_batches(len(train_loader))

            def loss_helper(pred, target):
                nll_loss, kl_loss = loss_object.compute_loss(pred, target)
                return (nll_loss + kl_loss / 10).mean(dim=0)

            return loss_helper

        if self.cfg.task_type == 'classification':
            # raw_loss = F.cross_entropy
            return F.nll_loss
        elif self.cfg.task_type == 'regression':
            return F.mse_loss
        else:
            raise ValueError

    def apply_pred_std(self, pred):
        return pred * self.stats['y_std'] + self.stats['y_mean']

    def apply_target_std(self, target):
        return (target - self.stats['y_mean']) / self.stats['y_std']


# ---- Make Radial BNN Sane ----
# There is some unexpected behaviour in terms of the data shapes
# expected as input and given as output from Radial BNN.
# Make them behave more similar to any other PyTorch model.

# If I ever want to build sane MCDO, use this to modify forward
# if model_arch == "consistent_mcdo":
#     prediction = torch.logsumexp(
#         model(data_N_C_H_W, variational_samples), dim=1
#     ) - math.log(variational_samples)
