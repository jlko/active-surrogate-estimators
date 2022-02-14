import logging
import wandb
import numpy as np


def t2n(tensor):
    return tensor.detach().cpu().numpy()


class TwoComponentMixtureDistribution:
    def __init__(self, p, q, a):
        """a: prob that we sample from safe dist p"""
        self.p = p
        self.q = q
        self.a = a

    def rvs(self, n):
        # probability that unif[0, 1] is < a is a
        n_a = (np.random.rand(n) < self.a).sum()

        # this is the probability with which we sample from p
        p_samples = np.atleast_2d(self.p.rvs(n_a))

        # all remaining samples are unsafe
        n_b = n - n_a
        assert n_b + n_a == n
        q_samples = np.atleast_2d(self.q.rvs(n_b))

        if (p_samples.size > 0) and (q_samples.size > 0):
            samples = np.concatenate([p_samples, q_samples], 0)
        elif q_samples.size == 0:
            samples = p_samples
        elif p_samples.size == 0:
            samples = q_samples
        else:
            raise


        # just because I'm paranoid, maybe the order of the samples matters
        if samples.shape[0] > 1:
            np.random.shuffle(samples)

        return samples

    def pdf(self, x):
        return self.a * self.p.pdf(x) + (1-self.a) * self.q.pdf(x)


def add_val_idxs_to_cfg(cfg, val_idxs):

    for key in cfg['acquisition_configs'].keys():
        cfg['acquisition_configs'][key]['val_idxs'] = [
            int(i) for i in val_idxs]
