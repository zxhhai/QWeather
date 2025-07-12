import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau


def get_optimizer(model, name="adam", lr=1e-3, weight_decay=0.0, momentum=0.9):
    """
    Get optimizer for the model.
    
    Args:
        model: model to optimize
        
    """
    name = name.lower()
    
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_scheduler(optimizer, name="step", step_size=30, gamma=0.1, T_max=200, patience=10):
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        name: Scheduler name ("step", "cosine", "plateau", "none")
        step_size: StepLR step size
        gamma: Learning rate decay factor
        T_max: CosineAnnealingLR maximum number of epochs
        patience: ReduceLROnPlateau patience value
    """
    name = name.lower()
    
    if name == "none" or name is None:
        return None
    elif name == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=T_max)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=patience)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")
