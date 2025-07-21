'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''
# part of the code is from dps

from abc import ABC, abstractmethod
from torchvision import torch
from torch.nn.functional import interpolate


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


def get_operator(config):
    assert config.task_name is not None, "Task must be provided for sampling"
    assert config.task_name in ['SR', 'SO'], "Task must be either super resolution or sparse observation"

    if config.task_name == 'SR': # super resolution
        assert config.SR_ratio is not None, "SR_ratio must be provided for super resolution"
        size = config.hi_size // config.SR_ratio
        operator = lambda x: interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    elif config.task_name == 'SO': # sparse observation
        assert config.SO_ratio is not None, "SO_ratio must be provided for sparse observation"
        tmp = torch.randn(1, 1, config.hi_size, config.hi_size)
        ratio = config.SO_ratio
        mask = torch.rand_like(tmp) < ratio
        mask = mask.float().to(config.device)
        operator = lambda x: x * mask
    else:
        raise ValueError(f"Task {config.task_name} is not supported")
    return operator


def get_noiser(config):
    assert config.noise_type is not None, "Noise type must be provided for sampling"
    assert config.noise_sigma is not None, "Noise sigma must be provided for sampling"
    return get_noise(config.noise_type, sigma=config.noise_sigma)