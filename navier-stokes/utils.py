import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torchvision import transforms as T
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.functional import interpolate
from torchvision import transforms as T
from torchvision.utils import make_grid

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import seaborn as sns
import math
import wandb
import argparse
from datetime import datetime
from time import time
import os
import numpy as np
import logging
from dataclasses import dataclass
import shutil

# local
from unet import Unet



# create a data structure for the constants
@dataclass(frozen=True)
class Constants:
    QG_V1_AVG_PIXEL_NORM_TRAIN = 0.6351982153248673
    QG_V1_AVG_PIXEL_NORM_FULL = 0.6352024454011677

    QG_V1_AVG_PIXEL_NORM_TRAIN_HALF = 0.475635826587677

    QG_V2_AVG_PIXEL_NORM_TRAIN = 1.0793864383430332
    QG_V2_AVG_PIXEL_NORM_FULL = 1.079314860803363
    QG_V3_AVG_PIXEL_NORM_TRAIN = 0.3492

    NSE_AVG_PIXEL_NORM = 3.0679163932800293

class Config:
    # def __init__(self, dataset, debug, overfit, sigma_coef, beta_fn):
    def __init__(self, args):

        self.dataset = args.dataset

        self.debug = args.debug
        
        logging.info("\n\n********* MODE *********\n\n")
        logging.info(f"DEBUG MODE: {self.debug}")
        
        # interpolant + sampling
        self.sigma_coef = args.sigma_coef
        self.beta_fn = args.beta_fn
        self.EM_sample_steps = args.EM_sample_steps
        self.t_min_sampling = 0.0  # no min time needed
        self.t_max_sampling = .999
        self.device = torch.device(args.device)

        # data
        self.s_in = args.s_in
        self.s_out = args.s_out

        # data
        if self.dataset == 'cifar':

            self.center_data = True
            self.C = 3
            self.H = 32
            self.W = 32
            self.num_classes = 10
            self.data_path = '../data/'
            self.grid_kwargs = {'normalize' : True, 'value_range' : (-1, 1)}

        elif self.dataset == 'nse' or 'qg' in self.dataset or 'square' in self.dataset:
            self.center_data = False
            self.home = args.home
            maybe_create_dir(self.home)
            self.data_fname = args.data_fname # to be changed to full data.
            self.num_classes = args.num_classes
            self.lo_size = args.lo_size
            self.hi_size = args.hi_size
            self.time_lag = args.time_lag
            self.subsampling_ratio = args.subsampling_ratio
            self.grid_kwargs = {'normalize': False}
            self.C = args.C
            self.H = self.hi_size
            self.W = self.hi_size

        else:
            assert False, "dataset must be 'cifar', 'nse', or 'qg'"

        # shared
        self.num_workers = args.num_workers
        self.delta_t = 0.5
        self.wandb_project = args.wandb_project
        self.wandb_entity = args.wandb_entity
        self.use_wandb = args.use_wandb
        self.noise_strength = 1.0
        self.sample_only = args.sample_only
        self.load_path = args.load_path
        self.overfit = args.overfit
        self.auto = args.auto
        logging.info(f"OVERFIT MODE: {self.overfit}")
        logging.info(f"SAMPLING ONLY: {self.sample_only}")
        logging.info(f"AUTO MODE: {self.auto}")

        if self.debug:
            self.EM_sample_steps = 10
            self.sample_every = 10
            self.print_loss_every = 10
            self.save_every = 10000000
        else:
            self.EM_sample_steps = args.EM_sample_steps
            self.sample_every = args.sample_every
            self.print_loss_every = args.print_loss_every # 1000
            self.save_every = args.save_every
        
        # some training hparams
        self.batch_size = 128 if self.dataset == 'cifar' else args.batch_size # TODO: change back to 32? 
        # self.batch_size = 64
        if self.overfit:
            self.batch_size = 8 # TODO: change back to 1 (jiayx)
        self.sampling_batch_size = self.batch_size if self.dataset=='cifar' else args.sampling_batch_size
        self.t_min_train = 0.0
        self.t_max_train = 1.0
        self.max_grad_norm = 1.0
        self.base_lr = args.base_lr # (IV. training & sampling)
        self.max_steps = args.max_steps # 1_000_000 # TODO: change back
        self.val_ratio = args.val_ratio
        self.validate_every = args.validate_every
        self.ckpt_dir = args.ckpt_dir

        # arch (III. model)
        self.unet_use_classes = True if self.dataset == 'cifar' else False
        self.unet_channels = args.unet_channels
        self.unet_dim_mults = args.unet_dim_mults
        self.unet_resnet_block_groups = args.unet_resnet_block_groups
        self.unet_learned_sinusoidal_dim = args.unet_learned_sinusoidal_dim
        self.unet_attn_dim_head = args.unet_attn_dim_head
        self.unet_attn_heads = args.unet_attn_heads
        self.unet_learned_sinusoidal_cond = args.unet_learned_sinusoidal_cond
        self.unet_random_fourier_features = args.unet_random_fourier_features

        # FlowDAS
        self.MC_times = args.MC_times
        self.auto_step = args.auto_step
        self.grad_scale = args.grad_scale

        # measurements
        self.task_name = args.task_name
        self.SR_ratio = args.SR_ratio
        self.SO_ratio = args.SO_ratio
        self.noise_type = args.noise_type
        self.noise_sigma = args.noise_sigma

def maybe_create_dir(f):
    if not os.path.exists(f):
        print("making", f)
        os.makedirs(f)

def bad(x):
    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))                                                                                        

def is_type_for_logging(x):
    '''
    Check if a variable is of a type that can be logged to wandb.

    Args:
        x (any): The variable to check.

    Returns:
        bool: True if the variable is of a type that can be logged to wandb, False otherwise.
    '''
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return True
    elif isinstance(x, bool):
        return True
    elif isinstance(x, str):
        return True
    elif isinstance(x, list):
        return True
    elif isinstance(x, set):
        return True
    else:
        return False

## if you want to make a grid of images
def to_grid(x, grid_kwargs):
    # nrow = int(np.floor(np.sqrt(x.shape[0])))
    # nrow = x.shape[0]
    nrow = 1
    return make_grid(x, nrow = nrow, **grid_kwargs) 
    # torchvision.utils.make_grid returns a tensor containing a grid of images

def clip_grad_norm(model, max_norm):
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm = max_norm, 
        norm_type= 2.0, 
        error_if_nonfinite = False
    )

def get_cifar_dataloader(config):

    Flip = T.RandomHorizontalFlip()
    Tens = T.ToTensor()
    transform = T.Compose([Flip, Tens])
    ds = datasets.CIFAR10(
        config.data_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    batch_size = config.batch_size

    return DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = True, 
        num_workers = config.num_workers,
        pin_memory = True,
        drop_last = True, 
    )

def setup_wandb(config):
    if not config.use_wandb:
        return

    config.wandb_run = wandb.init(
        project = config.wandb_project,
        entity = config.wandb_entity,
        resume = None,
        id = None,
    )
    # Manually push config values with allow_val_change
    wandb.config.update(config.__dict__, allow_val_change=True)

    config.wandb_run_id = config.wandb_run.id

    for key in vars(config):
        item = getattr(config, key)
        if is_type_for_logging(item):
            setattr(wandb.config, key, item)
    print("finished wandb setup")

class DriftModel(nn.Module):
    def __init__(self, config):
        
        super(DriftModel, self).__init__()
        self.config = config
        c = config
        self._arch = Unet(
            num_classes = c.num_classes,
            in_channels = c.C * 2, # times two for conditioning
            out_channels= c.C,
            dim = c.unet_channels,
            dim_mults = c.unet_dim_mults ,
            resnet_block_groups = c.unet_resnet_block_groups,
            learned_sinusoidal_cond = c.unet_learned_sinusoidal_cond,
            random_fourier_features = c.unet_random_fourier_features,
            learned_sinusoidal_dim = c.unet_learned_sinusoidal_dim,
            attn_dim_head = c.unet_attn_dim_head,
            attn_heads = c.unet_attn_heads,
            use_classes = c.unet_use_classes,
        )
        num_params = np.sum([int(np.prod(p.shape)) for p in self._arch.parameters()])
        logging.info("\n\n********* NETWORK *********\n\n")
        logging.info(f"Num params in main arch for drift is {num_params:,}")

    def forward(self, zt, t, y, cond=None):        
        if not self.config.unet_use_classes:
            y = None

        if cond is not None:
            zt = torch.cat([zt, cond], dim = 1)

        out = self._arch(zt, t, y)

        return out

def maybe_subsample(x, subsampling_ratio):
    if subsampling_ratio:
        x = x[ : int(subsampling_ratio * x.shape[0]), ...]
    return x

def maybe_lag(data, time_lag):
    """
    Args:
        data: (N, T, H, W)
        time_lag: int
    Returns:
        inputs: (N, T-time_lag, H, W)
        outputs: (N, T-time_lag, H, W)
    """
    if time_lag > 0:
        inputs = data[:, :-time_lag, ...]
        outputs = data[:, time_lag:, ...]
    else:
        inputs, outputs = data, data
    return inputs, outputs

def maybe_downsample(inputs, outputs, lo_size, hi_size):    
    upsampler = nn.Upsample(scale_factor=int(hi_size/lo_size), mode='nearest')
    hi = interpolate(outputs, size=(hi_size,hi_size),mode='bilinear').reshape([-1,hi_size,hi_size])
    lo = upsampler(interpolate(inputs, size=(lo_size,lo_size),mode='bilinear')) # TODO: why downsample first and then upsample?
    return lo, hi

def flatten_time(lo, hi, hi_size):
    hi = hi.reshape([-1,hi_size,hi_size])
    lo = lo.reshape([-1,hi_size,hi_size])
    # make the data N C H W
    hi = hi[:,None,:,:]
    lo = lo[:,None,:,:]
    return lo, hi

def loader_from_tensor(lo, hi, batch_size, shuffle):
    return DataLoader(TensorDataset(lo, hi), batch_size = batch_size, shuffle = shuffle)

def compute_avg_pixel_norm(data_raw):
    return torch.sqrt(torch.mean(data_raw ** 2))

def get_avg_pixel_norm(config, data):
    if config.dataset == 'qgv1':
        return Constants.QG_V1_AVG_PIXEL_NORM_TRAIN
    elif config.dataset == 'qgv2':
        return Constants.QG_V2_AVG_PIXEL_NORM_TRAIN
    elif config.dataset == 'qgv3':
        return Constants.QG_V3_AVG_PIXEL_NORM_TRAIN
    elif config.dataset == 'qgv4':
        assert False, "Not implemented yet!"
    elif config.dataset == 'nse':
        return Constants.NSE_AVG_PIXEL_NORM
    else:
        if 'half' in config.dataset:
            logging.info(f"Notice: using avg_pixel_norm computed from data for {config.dataset}")
            return compute_avg_pixel_norm(data)
        raise ValueError(f"Dataset {config.dataset} is not supported")

def get_forecasting_dataloader_nse(config, shuffle=False):
    data_raw, time_raw = torch.load(config.data_fname)
    del time_raw
    
    # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    
    
    # avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori
    avg_pixel_norm = compute_avg_pixel_norm(data_raw)
    data = data_raw/avg_pixel_norm
    new_avg_pixel_norm = compute_avg_pixel_norm(data) # 1.0

    logging.info("\n\n********* DATA *********\n\n")

    logging.info(f"avg_pixel_norm: {avg_pixel_norm}")

    # here "lo" will be the conditioning info (initial condition) and "hi" will be the target
    # lo is x_t and hi is x_{t+tau}, and lo might be lower res than hi

    logging.info(f"data_raw.shape: {data_raw.shape}")
    logging.info(f"config.time_lag: {config.time_lag}")
    logging.info(f"config.lo_size: {config.lo_size}")
    logging.info(f"config.hi_size: {config.hi_size}")
    logging.info(f"config.subsampling_ratio: {config.subsampling_ratio}")
    logging.info(f"config.batch_size: {config.batch_size}")

    logging.info("\n")

    lo, hi = maybe_lag(data, config.time_lag)

    # logging.info(f"lo.shape after maybe_lag: {lo.shape}")
    # logging.info(f"hi.shape after maybe_lag: {hi.shape}")

    # logging.info("\n")

    lo, hi = maybe_downsample(lo, hi, config.lo_size, config.hi_size)

    # logging.info(f"lo.shape after maybe_downsample: {lo.shape}")
    # logging.info(f"hi.shape after maybe_downsample: {hi.shape}")

    # logging.info("\n")

    lo, hi = flatten_time(lo, hi, config.hi_size)

    # logging.info(f"lo.shape after flatten_time: {lo.shape}")
    # logging.info(f"hi.shape after flatten_time: {hi.shape}")

    # logging.info("\n")

    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)

    logging.info(f"shape of the dataset (lo): {lo.shape}")
    logging.info(f"shape of the dataset (hi): {hi.shape}")

    # logging.info("\n")

    # now they are image shaped. Be sure to shuffle to de-correlate neighboring samples when training. 
    # loader = loader_from_tensor(lo, hi, config.batch_size, shuffle = shuffle)

    dataset = TensorDataset(lo, hi)

    # split into train and val according to config.val_ratio
    N = len(dataset)
    val_size = int(N * config.val_ratio)
    train_size = N - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)   

    return train_loader, val_loader, avg_pixel_norm, new_avg_pixel_norm

def get_forecasting_dataloader_qg(config, shuffle=False):
    data_raw = torch.load(config.data_fname)
    data_raw = data_raw.float()

    # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    
    
    # avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori

    ans = input("Have you update the avg_pixel_norm for the dataset? (y/n): ")
    if ans == 'y' or ans == '':
        avg_pixel_norm = get_avg_pixel_norm(config, data_raw)
        logging.info(f"avg_pixel_norm: {avg_pixel_norm}")
    else:
        avg_pixel_norm = 1.0
        assert False, "You must update the avg_pixel_norm for the dataset!"

    # avg_pixel_norm = compute_avg_pixel_norm(data_raw)
    data = data_raw/avg_pixel_norm
    new_avg_pixel_norm = 1.0

    logging.info("\n\n********* DATA *********\n\n")

    logging.info(f"avg_pixel_norm: {avg_pixel_norm}")

    # here "lo" will be the conditioning info (initial condition) and "hi" will be the target
    # lo is x_t and hi is x_{t+tau}, and lo might be lower res than hi

    logging.info(f"data_raw.shape: {data_raw.shape}")
    logging.info(f"config.time_lag: {config.time_lag}")
    logging.info(f"config.lo_size: {config.lo_size}")
    logging.info(f"config.hi_size: {config.hi_size}")
    logging.info(f"config.subsampling_ratio: {config.subsampling_ratio}")
    logging.info(f"config.batch_size: {config.batch_size}")

    logging.info("\n")

    lo, hi = maybe_lag(data, config.time_lag)

    # logging.info(f"lo.shape after maybe_lag: {lo.shape}")
    # logging.info(f"hi.shape after maybe_lag: {hi.shape}")

    logging.info("\n")

    lo, hi = flatten_time(lo, hi, config.hi_size)

    # logging.info(f"lo.shape after flatten_time: {lo.shape}")
    # logging.info(f"hi.shape after flatten_time: {hi.shape}")

    # logging.info("\n")

    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)

    logging.info(f"shape of the dataset (lo): {lo.shape}")
    logging.info(f"shape of the dataset (hi): {hi.shape}")

    # logging.info("\n")

    # now they are image shaped. Be sure to shuffle to de-correlate neighboring samples when training. 
    # loader = loader_from_tensor(lo, hi, config.batch_size, shuffle = shuffle)

    dataset = TensorDataset(lo, hi)

    # split into train and val according to config.val_ratio
    N = len(dataset)
    val_size = int(N * config.val_ratio)
    train_size = N - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, avg_pixel_norm, new_avg_pixel_norm

def get_forecasting_dataloader_qg_half(config, shuffle=False):
    data_raw = torch.load(config.data_fname)
    data_raw = data_raw.float()

    half_T = data_raw.shape[1] // 2
    data_raw = data_raw[:, :half_T]

    # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    
    
    # avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori

    avg_pixel_norm = get_avg_pixel_norm(config, data_raw)

    # avg_pixel_norm = compute_avg_pixel_norm(data_raw)
    data = data_raw/avg_pixel_norm
    new_avg_pixel_norm = 1.0

    logging.info("\n\n********* DATA *********\n\n")

    logging.info(f"avg_pixel_norm: {avg_pixel_norm}")

    # here "lo" will be the conditioning info (initial condition) and "hi" will be the target
    # lo is x_t and hi is x_{t+tau}, and lo might be lower res than hi

    logging.info(f"data_raw.shape: {data_raw.shape}")
    logging.info(f"config.time_lag: {config.time_lag}")
    logging.info(f"config.lo_size: {config.lo_size}")
    logging.info(f"config.hi_size: {config.hi_size}")
    logging.info(f"config.subsampling_ratio: {config.subsampling_ratio}")
    logging.info(f"config.batch_size: {config.batch_size}")

    logging.info("\n")

    lo, hi = maybe_lag(data, config.time_lag)

    # logging.info(f"lo.shape after maybe_lag: {lo.shape}")
    # logging.info(f"hi.shape after maybe_lag: {hi.shape}")

    logging.info("\n")

    lo, hi = flatten_time(lo, hi, config.hi_size)

    # logging.info(f"lo.shape after flatten_time: {lo.shape}")
    # logging.info(f"hi.shape after flatten_time: {hi.shape}")

    # logging.info("\n")

    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)

    logging.info(f"shape of the dataset (lo): {lo.shape}")
    logging.info(f"shape of the dataset (hi): {hi.shape}")

    # logging.info("\n")

    # now they are image shaped. Be sure to shuffle to de-correlate neighboring samples when training. 
    # loader = loader_from_tensor(lo, hi, config.batch_size, shuffle = shuffle)

    dataset = TensorDataset(lo, hi)

    # split into train and val according to config.val_ratio
    N = len(dataset)
    val_size = int(N * config.val_ratio)
    train_size = N - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, avg_pixel_norm, new_avg_pixel_norm

def get_forecasting_dataloader_qg_sampling(config):
    # Should use the unseen data for sampling
    data_raw = torch.load(config.data_fname)
    data_raw = data_raw.float()
    '''
    ans = input("Have you update the avg_pixel_norm for the dataset? (y/n): ")
    if ans == 'y' or ans == '':
        avg_pixel_norm = get_avg_pixel_norm(config)
        logging.info(f"avg_pixel_norm: {avg_pixel_norm}")
    else:
        avg_pixel_norm = 1.0
        assert False, "You must update the avg_pixel_norm for the dataset!"
    '''
    avg_pixel_norm = get_avg_pixel_norm(config)
    
    data = data_raw/avg_pixel_norm
    
    logging.info("\n\n********* DATA FOR SAMPLING *********\n\n")

    logging.info(f"avg_pixel_norm: {avg_pixel_norm}")

    # here "lo" will be the conditioning info (initial condition) and "hi" will be the target
    # lo is x_t and hi is x_{t+tau}, and lo might be lower res than hi

    logging.info(f"data_raw.shape: {data_raw.shape}")
    logging.info(f"config.time_lag: {config.time_lag}")
    logging.info(f"config.lo_size: {config.lo_size}")
    logging.info(f"config.hi_size: {config.hi_size}")
    logging.info(f"config.subsampling_ratio: {config.subsampling_ratio}")
    logging.info(f"config.batch_size: {config.batch_size}")

    logging.info("\n")

    lo, hi = maybe_lag(data, config.time_lag) # (N, T-time_lag, H, W)

    config.max_T = lo.shape[1] # length of the longest trajectory

    # logging.info(f"lo.shape after maybe_lag: {lo.shape}")
    # logging.info(f"hi.shape after maybe_lag: {hi.shape}")

    # logging.info("\n")

    lo, hi = maybe_downsample(lo, hi, config.lo_size, config.hi_size)

    # logging.info(f"lo.shape after maybe_downsample: {lo.shape}")
    # logging.info(f"hi.shape after maybe_downsample: {hi.shape}")

    # logging.info("\n")

    lo, hi = flatten_time(lo, hi, config.hi_size)

    # logging.info(f"lo.shape after flatten_time: {lo.shape}")
    # logging.info(f"hi.shape after flatten_time: {hi.shape}")

    # logging.info("\n")

    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)

    logging.info(f"shape of the dataset (lo): {lo.shape}")
    logging.info(f"shape of the dataset (hi): {hi.shape}")

    # logging.info("\n")

    # now they are image shaped. Be sure to shuffle to de-correlate neighboring samples when training. 
    # loader = loader_from_tensor(lo, hi, config.batch_size, shuffle = shuffle)

    dataset = TensorDataset(lo, hi)

    # split into train and val according to config.val_ratio
    N = len(dataset)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    return test_loader, avg_pixel_norm

def get_forecasting_dataloader_qg_half_sampling(config, shuffle = False):
    data_raw = torch.load(config.data_fname)
    data_raw = data_raw.float()

    half_T = data_raw.shape[1] // 2
    data_raw = data_raw[:, half_T:]

    # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    
    
    # avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori

    avg_pixel_norm = Constants.QG_V1_AVG_PIXEL_NORM_TRAIN_HALF

    # avg_pixel_norm = compute_avg_pixel_norm(data_raw)
    data = data_raw/avg_pixel_norm
    new_avg_pixel_norm = 1.0

    logging.info("\n\n********* DATA *********\n\n")

    logging.info(f"avg_pixel_norm: {avg_pixel_norm}")

    # here "lo" will be the conditioning info (initial condition) and "hi" will be the target
    # lo is x_t and hi is x_{t+tau}, and lo might be lower res than hi

    logging.info(f"data_raw.shape: {data_raw.shape}")
    logging.info(f"config.time_lag: {config.time_lag}")
    logging.info(f"config.lo_size: {config.lo_size}")
    logging.info(f"config.hi_size: {config.hi_size}")
    logging.info(f"config.subsampling_ratio: {config.subsampling_ratio}")
    logging.info(f"config.batch_size: {config.batch_size}")

    logging.info("\n")

    lo, hi = maybe_lag(data, config.time_lag)
    
    config.max_T = lo.shape[1] # length of the longest trajectory
    # logging.info(f"lo.shape after maybe_lag: {lo.shape}")
    # logging.info(f"hi.shape after maybe_lag: {hi.shape}")

    logging.info("\n")

    lo, hi = flatten_time(lo, hi, config.hi_size)

    # logging.info(f"lo.shape after flatten_time: {lo.shape}")
    # logging.info(f"hi.shape after flatten_time: {hi.shape}")

    # logging.info("\n")

    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)

    logging.info(f"shape of the dataset (lo): {lo.shape}")
    logging.info(f"shape of the dataset (hi): {hi.shape}")

    # logging.info("\n")

    # now they are image shaped. Be sure to shuffle to de-correlate neighboring samples when training. 
    # loader = loader_from_tensor(lo, hi, config.batch_size, shuffle = shuffle)

    dataset = TensorDataset(lo, hi)

    # split into train and val according to config.val_ratio
    N = len(dataset)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    return test_loader, avg_pixel_norm

def extract_patches(inputs, outputs, s_in, s_out):
    """
    For each trajectory, extract all valid patches (lo, hi) windows and do not cross trajectory boundaries.
    Input:
        inputs: (N, T, C, H, W)
        outputs: (N, T, C, H, W)
        s_in: int, the size of the input patch
        s_out: int, the size of the output patch
    Output:
        lo: (M, C*s_in, H, W)
        hi: (M, C*s_out, H, W)
    """

    assert inputs.ndim == 5, "inputs must be 5D"
    assert outputs.ndim == 5, "outputs must be 5D"

    assert inputs.shape == outputs.shape, "inputs and outputs must have the same shape"

    N, T, C, H, W = inputs.shape

    M = 0
    lo_list, hi_list = [], []

    max_t = T - s_in - s_out + 1 # maximum time index for the start of lo
    for n in range(N):
        for t in range(max_t):
            lo_patch = inputs[n, t:t+s_in] # (s_in, C, H, W)
            hi_patch = outputs[n, t+s_in:t+s_in+s_out] # (s_out, C, H, W)
            lo_patch = lo_patch.reshape(-1, H, W) # (s_in*C, H, W)
            hi_patch = hi_patch.reshape(-1, H, W) # (s_out*C, H, W)
            lo_list.append(lo_patch)
            hi_list.append(hi_patch)
            M += 1
    
    lo = torch.stack(lo_list, dim=0) # (M, C*s_in, H, W)
    hi = torch.stack(hi_list, dim=0) # (M, C*s_out, H, W)

    assert max_t*N == M, "M must be max_t*N"

    return lo, hi

def get_dataloader_C(config):
    """
    Get the dataloader for the dataset with C channels, where C is not necessarily 1.
    Input:
        data_raw: (N, T, H, W)
    Output:
        train_loader: dataloader for training
            lo: (M, s_in, H, W)
            hi: (M, s_out, H, W)
    """

    data_raw = torch.load(config.data_fname)
    data_raw = data_raw.float()
    assert data_raw.ndim == 4, "data_raw must be 4D"

    data = data_raw.unsqueeze(2) # (N, T, C=1, H, W) in the case of QG and NSE
    assert data.ndim == 5, "data must be 5D now"

    N, T, C, H, W = data.shape

    lo, hi = maybe_lag(data, config.time_lag)
    assert lo.shape[1] == T - config.time_lag, "lo must have N - time_lag trajectories"
    assert hi.shape[1] == T - config.time_lag, "hi must have N - time_lag trajectories"

    lo, hi = extract_patches(lo, hi, config.s_in, config.s_out)

    dataset = TensorDataset(lo, hi)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    return dataloader

def get_dataloader_C_latent(config):
    """
    Get the dataloader for the dataset with C channels, where C is not necessarily 1.
    Input:
        data_raw: (N, T, C, H, W)
    Output:
        train_loader: dataloader for training
            lo: (B, C_in, H, W)
            hi: (B, C_out, H, W)
    """

def make_one_redblue_plot(x, fname, vmin=-2, vmax=2):
    """
    Make a single redblue plot from a tensor of images.

    Args:
        x (torch.Tensor): A tensor of images, shape: (H, W), C=1. 
        fname (str): A string, the path to the file to save the plot.

    Returns:
        None
    """
    plt.ioff()
    fig = plt.figure(figsize=(3,3))
    plt.imshow(x.T, cmap=sns.cm.icefire, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(fname, bbox_inches = 'tight')
    plt.close("all")         

def open_redblue_plot_as_tensor(fname):
    return T.ToTensor()(Image.open(fname))

def make_redblue_plots(x, config, name):
    # write a docstring for this function
    """
    Make a grid of redblue plots from a tensor of images.

    Args:
        x (torch.Tensor): A tensor of images, shape: (bsz, C, H, W), C=1. 
        config (Config): A Config object.
            - config.home: a string, the path to the home directory.

    Returns:
        out (torch.Tensor): A tensor of redblue plots. # (bsz, C=3, H, W)
    """
    plt.ioff()
    x = x.cpu()
    bsz = x.size()[0] # 1
    # logging.info(f"bsz: {bsz}")
    vmin = -2
    vmax = 2
    for i in range(bsz):
        make_one_redblue_plot(x[i,0,...], fname=config.home+name+f'_tmp{i}.jpg', vmin=vmin, vmax=vmax)
    tensor_img = T.ToTensor()(Image.open(config.home+name+f'_tmp0.jpg'))
    # tensor_img = T.ToTensor()(Image.open(config.home + f'tmp1.jpg'))

    C, H, W = tensor_img.size()
    out = torch.zeros((bsz,C,H,W))
    for i in range(bsz):
        out[i,...] = open_redblue_plot_as_tensor(config.home+name+f'_tmp{i}.jpg')
    return out

def setup_logger(save_dir=None, log_level=logging.INFO, config_path=None):
    # Create top-level log directory
    if save_dir is None:
        save_dir = "./logs"
    os.makedirs(save_dir, exist_ok=True)

    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Setup log file path
    log_path = os.path.join(run_dir, "log.txt")

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging to {log_path}")

    # Save config file if provided
    if config_path is not None:
        try:
            config_save_path = os.path.join(run_dir, "config.yml")
            shutil.copy(config_path, config_save_path)
            logging.info(f"Saved config file to {config_save_path}")
        except Exception as e:
            logging.error(f"Failed to save config file: {e}")

    return run_dir  # you can return the folder path for saving models/checkpoints etc.

# BELOW IS THE ORIGINAL CODE (NOT USED)

def main_raw():

    assert False, "why are you using this?!"

    parser = argparse.ArgumentParser(description='hello')
    parser.add_argument('--dataset', type = str, choices = ['cifar', 'nse'], default = 'nse')
    parser.add_argument('--load_path', type = str, default = None)
    parser.add_argument('--use_wandb', type = int, default = 1)
    parser.add_argument('--sigma_coef', type = float, default = 1.0) 
    parser.add_argument('--beta_fn', type = str, default = 't^2', choices=['t','t^2'])
    parser.add_argument('--debug', type = int, default = 0)
    parser.add_argument('--sample_only', type = int, default = 0)
    parser.add_argument('--overfit', type = int, default = 0)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    for k in vars(args):
        logging.info(f"{k}: {getattr(args, k)}")
    
    logging.info("************************************************")

    logging.info(f"********* RUNNING at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} *********")

    conf = Config(
        dataset = args.dataset, 
        debug = bool(args.debug), # use as desired 
        overfit = bool(args.overfit),
        sigma_coef = args.sigma_coef, 
        beta_fn = args.beta_fn
    )

    trainer = Trainer(
        conf, 
        load_path = args.load_path, # none trains from scratch 
        sample_only = bool(args.sample_only), 
        use_wandb = bool(args.use_wandb)
    )

    if bool(args.sample_only):
        trainer.sample_ckpt()
    else:
        trainer.fit()
