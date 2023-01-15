import torch
from torch.nn import functional as F


def select_loss_function(loss_function):
    if loss_function == "l2_loss":
        return F.mse_loss
    elif loss_function == "l1_loss":
        return F.l1_loss
    else:
        raise ValueError("Unknown loss function!")


def select_optimizer(optimizer_type, params, lr=None, weight_decay=None):
    if optimizer_type == "ADAM":
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "RMS_PROP":
        return torch.optim.RMSprop(params=params, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer type!")
