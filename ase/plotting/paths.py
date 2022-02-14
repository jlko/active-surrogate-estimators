from pathlib import Path


def plus_base(base, end):
    if isinstance(end, list):
        return [plus_base(base, i) for i in end]
    else:
        if not base.endswith('/'):
            base = base + '/'
        return base + end


class ReproduciblePaths:
    """Store paths for experiment results."""
    base = 'outputs'
    distshift = 'Missing7'
    distshift = plus_base(base, distshift)

    resnets = [
        'ResNetFMNIST',
        'ResNetCifar10',
        'ResNetCifar100']
    resnets = plus_base(base, resnets)
    resnets_names = ['Fashion-MNIST', 'CIFAR-10', 'CIFAR-100']


class LegacyPaths:
    """Legacy paths from before reproducible."""
    base = 'outputs'

    distshift = 'quadrature_temp_scaled_retrain/Missing7New'
    distshift = plus_base(base, distshift)

    resnets = [
        'LargeFMNISTResNetMulti/2021-09-30-18-26-46',
        'LargeCIFAR10ResNetMulti/2021-09-30-18-26-52',
        'LargeCIFAR100WideResNetMulti/2021-09-30-18-27-00']
    resnets = plus_base(f'{base}/quadrature_unseen', resnets)
    resnets_names = ['Fashion-MNIST', 'CIFAR-10', 'CIFAR-100']

