import os
import logging
from pathlib import Path
import copy
import hydra
from omegaconf import OmegaConf

import math
import numpy as np
import torch
from torch import nn
from torch._C import Value
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split

from ase.utils.data import to_json, from_json
from ase.models.sk2torch import TransformDataset
from .skmodels import BaseModel


class SK2TorchRegression(BaseModel):
    """Interface for Pytorch Models and SKlearn Methods."""
    def __init__(self, model, cfg):
        logging.info(f'Initialising new model {cfg.name}.')

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='regression',))

        self.cfg = cfg
        self.t_cfg = cfg['training_cfg']
        self.model = model.to(device=self.device).type(
            torch.float64)
        self.T = None
        self.needs_reinit = True

        self.gauss_lik = cfg.get('gaussian_lik', False)
        self.loss_name = 'NLL' if self.gauss_lik else 'MSE'
        self.loss = self.gaussian_nll_loss() if self.gauss_lik else F.mse_loss

    def gaussian_nll_loss(self):

        def extract_var_nll_loss(input, target, *args, **kwargs):
            """Extract variance from prediction."""
            return F.gaussian_nll_loss(
                input[..., 0], target, input[..., 1],
                *args, **kwargs)

        return extract_var_nll_loss

    def predict(
            self, x, for_acquisition=True, to_numpy=True, return_std=False,
            *args, **kwargs):

        if isinstance(x, torch.Tensor):
            loader = [[x, ]]
        elif len(x) > self.t_cfg['batch_size']:
            loader = self.make_loader([x], train=False)
        elif isinstance(x, np.ndarray):
            loader = [[torch.from_numpy(x)]]
        else:
            raise ValueError

        if for_acquisition:
            self.model.eval()

        preds = []
        with torch.no_grad():
            for (data, ) in loader:
                data = data.to(self.device)
                pred = self.model(data)

                preds.append(pred)

        preds = torch.cat(preds, 0)

        if to_numpy:
            preds = preds.detach().cpu().numpy()

        if return_std:
            return preds, None
        else:
            return preds

    def fit(self, x, y):
        # don't fit model and load instead (but only if chkpt exists)
        p = self.cfg['skip_fit_debug']

        # set up base bath
        if self.cfg.get('skip_fit_debug_relative', False):
            base = Path('.')
        else:
            base = Path(hydra.utils.get_original_cwd())

        # need to always call this to fill in y_train_mean_std
        train_loader, val_loader = self.val_train_loaders(x, y)

        # load or fit model
        if p and (loc := (base / p)).exists():
            logging.info(f'Loading model from {p}.')
            self.model.load_state_dict(
                torch.load(loc, map_location=self.device))
        else:
            if p and not (loc := (base / p)).exists():
                logging.info(f'Tried to load model, but {p} does not exist.')

            # from ase.datasets import get_CIFAR10
            # train_loader, val_loader = get_CIFAR10()
            self.model = self.train_to_convergence(
                self.model, train_loader, val_loader)

        path = Path(self.cfg.get('save_path', 'model.pth'))

        if not path.exists():
            logging.info(f'Saving model to {path}.')
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), path)

    def val_train_loaders(self, x, y):

        # generate random splits for train and val
        val_size = int(self.t_cfg['validation_set_size'])

        # Just added random seed. Previously no idea how this split happened.
        # it's also important that these are constant for constant data
        # because otherwise the ensembles will get different train/val splits
        # which would lead to some information leak for the calibration!
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=val_size, random_state=0)

        # I cannot standardise y because of estimation
        # so, instead, I unstandardise predictions
        self.model.y_train_mean = y_train.mean()
        self.model.y_train_std = y_train.std()

        train_loader = self.make_loader([x_train, y_train])
        val_loader = self.make_loader([x_val, y_val], train=False)

        return train_loader, val_loader

    def make_loader(self, arrs, train=True):

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

        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=train,
            batch_size=batch_size,
            num_workers=self.t_cfg['num_workers'],
            pin_memory=self.t_cfg['pin_memory'])

        return data_loader

    def train_to_convergence(self, model, train_loader, val_loader):
        logging.info(
            f'Beginning training with {len(train_loader.dataset)} training '
            f'points and {len(val_loader.dataset)} validation.'
        )

        m = self.t_cfg['max_epochs']
        log_every = int(.02 * m) if m > 100 else 1
        best = np.inf
        best_model = model
        patience = 0
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        for epoch in range(self.t_cfg['max_epochs']):
            if epoch > 0:
                # get val error for untrained prediction
                train_loss = self.train(model, train_loader, optimizer)
            else:
                train_loss = -1

            if self.gauss_lik:
                val_loss, val_mse = self.evaluate(
                    model, val_loader, with_mse=True)
            else:
                val_loss = self.evaluate(model, val_loader)

            if epoch % log_every == 0:
                logging.info(
                    f'Epoch {epoch:0>3d} eval:'
                    f'Val {self.loss_name}: {val_loss:.9f}, '
                    f'Train {self.loss_name}: {train_loss:.9f}')
                if self.gauss_lik:
                    logging.info(f'Val mse: {val_mse:.9f}')

            if val_loss < best:
                best, best_i = val_loss, epoch
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
        logging.info(f'Taking model from epoch {best_i} with loss {best_i}.')
        return best_model

    def train(self, model, train_loader, optimizer):
        model.train()

        loss = 0

        for data, target in train_loader:

            data = data.to(self.device)
            # target = target + 50 * torch.randn_like(target)
            target = target.to(self.device)

            optimizer.zero_grad()

            prediction = model(data)

            loss_i = self.loss(prediction, target)

            loss_i.backward()

            optimizer.step()

            # mean -> sum
            loss += loss_i.item() * data.shape[0]

        loss /= len(train_loader.dataset)
        return loss

    def evaluate(self, model=None, eval_loader=None, with_mse=False):
        if model is None:
            model = self.model
        if eval_loader is None:
            raise ValueError

        self.model.eval()

        loss = 0
        if with_mse:
            mse_loss = 0

        with torch.no_grad():
            for data, target in eval_loader:

                data = data.to(self.device)
                target = target.to(self.device)

                prediction = model(data)

                loss_i = self.loss(prediction, target, reduction='sum')
                loss += loss_i
                if with_mse:
                    mse_loss_i = F.mse_loss(
                        prediction[..., 0], target, reduction='sum')
                    mse_loss += mse_loss_i

        loss /= len(eval_loader.dataset)

        if with_mse:
            mse_loss /= len(eval_loader.dataset)
            std = torch.log(prediction[..., 1]).mean()
            logging.info(f'Mean Std of last batch: {std}.')
            return loss.item(), mse_loss.item()

        else:
            return loss.item()

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

        elif c == 'custom':
            milestones = [int(epochs * 0.3), int(epochs * 0.5), int(epochs * 0.8)]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.2)

        elif c == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)

        elif c == 'devries':
            # https://arxiv.org/abs/1708.04552v2
            assert epochs == 200
            milestones = [60, 120, 160]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.2)

        return scheduler


class _SimpleFF(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.gauss_lik = cfg.get('gaussian_lik', False)
        out_dim = 2 if self.gauss_lik else 1
        input_dim = cfg.data_CHW

        # set from SK2TorchRegression.val_train_loaders
        self.y_train_mean, self.y_train_std = None, None
        if cfg.get('fully_linear', False):
            self.model = nn.Linear(input_dim, out_dim, bias=False)
            if cfg.get('perfect_linear', False):
                self.model.weight = torch.nn.Parameter(torch.ones_like(
                    self.model.weight.data))
            else:
                self.model.weight.data = self.model.weight.data + 1

            self.forward = self.simple_forward

        else:
            nonlinear = nn.LeakyReLU
            self.model = nn.Sequential(*[
                nn.Linear(input_dim, 1024),
                # nn.Dropout(p=0.2),
                nonlinear(),
                nn.Linear(1024, 512),
                nonlinear(),
                nn.Linear(512, 128),
                nonlinear(),
                nn.Linear(128, out_dim)
            ])

            if self.gauss_lik:
                self.forward = self.gauss_forward
            else:
                self.forward = self.mse_forward

    def simple_forward(self, x):
        return self.model(x).reshape(-1)

    def mse_forward(self, x):
        out = self.model(x).reshape(-1)
        out = out * self.y_train_std + self.y_train_mean
        return out

    def gauss_forward(self, x):
        out = self.model(x)
        # exp transform variances to enforce positive and also make easier to
        # cross magnitudes
        out = torch.stack([
            out[..., 0] * self.y_train_std + self.y_train_mean,
            # torch.exp(out[..., 1])
            torch.log(torch.exp(out[..., 1]) + 1)
            ], 1)

        return out


class SimpleFF(SK2TorchRegression):
    def __init__(self, cfg):
        model = _SimpleFF(cfg)
        super().__init__(model, cfg)
