# https://github.com/Jonathan-Pearce/calibration_library/blob/a210e5f5607348c54263e6e677b559377605ef61/recalibration.py#L24
import logging
import torch
from ase.utils.calibration_library import metrics


def set_temperature(model, valid_loader, device, lr=0.1, n_samples=None):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    logging.info(f'Starting calibration with lr={lr}.')
    temperature = torch.nn.Parameter(
        1*torch.ones(1).to(device), requires_grad=True)
    model.eval()

    # nll_criterion = torch.nn.functional.nll_loss
    ece_criterion = metrics.ECELoss()

    # First: collect all the logits and labels for the validation set
    preds = []
    labels = []

    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_loader):
            data = data.to(device)
            target = target.to(device)

            # get logit predictions
            preds += [model(data, with_log_softmax=False, n_samples=n_samples)]
            labels += [target]

        preds = torch.cat(preds)
        labels = torch.cat(labels)

    # Calculate NLL and ECE before temperature scaling
    log_softmax = torch.nn.LogSoftmax(dim=1)
    before_temperature_nll = torch.nn.functional.nll_loss(
        log_softmax(preds/temperature), labels).item()

    before_temperature_ece = ece_criterion.loss(
        torch.exp(log_softmax(preds/temperature)).detach().cpu().numpy(),
        labels.cpu().numpy(), 15, logits=False)

    logging.info(
        f'Before temperature - NLL: %.5f, ECE: %.5f' % (
            before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    # optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=500)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=2000)
    # optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=2000)

    def eval():
        optimizer.zero_grad()
        loss = torch.nn.functional.nll_loss(
            log_softmax(preds/temperature), labels)
        loss.backward()
        return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = torch.nn.functional.nll_loss(
        log_softmax(preds/temperature), labels).item()
    after_temperature_ece = ece_criterion.loss(
        torch.exp(log_softmax(preds/temperature)).detach().cpu().numpy(),
        labels.cpu().numpy(), 15, logits=False)

    logging.info('Optimal temperature: %.3f' % temperature.item())
    logging.info(
        'After temperature - NLL: %.5f, ECE: %.5f' % (
            after_temperature_nll, after_temperature_ece))

    failure = before_temperature_nll < after_temperature_nll
    logging.info(f'Failure of optimisation: {failure}')

    results = dict(
        before_temperature_nll=before_temperature_nll,
        before_temperature_ece=before_temperature_ece,
        after_temperature_nll=after_temperature_nll,
        after_temperature_ece=after_temperature_ece,
        T=temperature.item(),
        failure=failure)

    return results
