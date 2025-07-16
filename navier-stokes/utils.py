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
import datetime
from time import time
import os
import numpy as np
import logging

# local
from unet import Unet


class Config:
    
    # def __init__(self, dataset, debug, overfit, sigma_coef, beta_fn):
    def __init__(self, args):

        self.dataset = args.dataset

        self.debug = args.debug
        logging.info(f"DEBUG MODE: {self.debug}")
        
        # interpolant + sampling
        self.sigma_coef = args.sigma_coef
        self.beta_fn = args.beta_fn
        self.EM_sample_steps = args.EM_sample_steps
        self.t_min_sampling = 0.0  # no min time needed
        self.t_max_sampling = .999

        # data
        if self.dataset == 'cifar':

            self.center_data = True
            self.C = 3
            self.H = 32
            self.W = 32
            self.num_classes = 10
            self.data_path = '../data/'
            self.grid_kwargs = {'normalize' : True, 'value_range' : (-1, 1)}

        elif self.dataset == 'nse':
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
        
        elif self.dataset == 'qg':
            self.center_data = False
            self.home = args.home
            maybe_create_dir(self.home)
            self.data_fname = args.data_fname
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
            assert False

        # shared
        self.num_workers = args.num_workers
        self.delta_t = 0.5
        self.wandb_project = args.wandb_project
        self.wandb_entity = args.wandb_entity
        # self.use_wandb = True
        self.noise_strength = 1.0

        self.overfit = args.overfit
        logging.info(f"OVERFIT MODE: {self.overfit}")

        if self.debug:
            self.EM_sample_steps = args.EM_sample_steps
            self.sample_every = 10
            self.print_loss_every = 10
            self.save_every = 10000000
        else:
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


def maybe_create_dir(f):
    if not os.path.exists(f):
        print("making", f)
        os.makedirs(f)

def bad(x):
    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))                                                                                        

def is_type_for_logging(x):
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
    nrow = int(np.floor(np.sqrt(x.shape[0])))
    return make_grid(x, nrow = nrow, **grid_kwargs)

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


def get_forecasting_dataloader_nse(config, shuffle = False):
    data_raw, time_raw = torch.load(config.data_fname)
    del time_raw
    
    # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    
    
    # avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori
    avg_pixel_norm = compute_avg_pixel_norm(data_raw)
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

    logging.info(f"lo.shape after maybe_lag: {lo.shape}")
    logging.info(f"hi.shape after maybe_lag: {hi.shape}")

    logging.info("\n")

    lo, hi = maybe_downsample(lo, hi, config.lo_size, config.hi_size)

    logging.info(f"lo.shape after maybe_downsample: {lo.shape}")
    logging.info(f"hi.shape after maybe_downsample: {hi.shape}")

    logging.info("\n")

    lo, hi = flatten_time(lo, hi, config.hi_size)

    logging.info(f"lo.shape after flatten_time: {lo.shape}")
    logging.info(f"hi.shape after flatten_time: {hi.shape}")

    logging.info("\n")

    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)

    logging.info(f"lo.shape after maybe_subsample: {lo.shape}")
    logging.info(f"hi.shape after maybe_subsample: {hi.shape}")

    logging.info("\n")

    # now they are image shaped. Be sure to shuffle to de-correlate neighboring samples when training. 
    loader = loader_from_tensor(lo, hi, config.batch_size, shuffle = shuffle)
    return loader, avg_pixel_norm, new_avg_pixel_norm


def get_forecasting_dataloader_qg(config, shuffle = False):
    data_raw = torch.load(config.data_fname)
    data_raw = data_raw.float()

    # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    
    
    # avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori
    avg_pixel_norm = compute_avg_pixel_norm(data_raw)
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

    logging.info(f"lo.shape after maybe_lag: {lo.shape}")
    logging.info(f"hi.shape after maybe_lag: {hi.shape}")

    logging.info("\n")

    lo, hi = flatten_time(lo, hi, config.hi_size)

    logging.info(f"lo.shape after flatten_time: {lo.shape}")
    logging.info(f"hi.shape after flatten_time: {hi.shape}")

    logging.info("\n")

    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)

    logging.info(f"lo.shape after maybe_subsample: {lo.shape}")
    logging.info(f"hi.shape after maybe_subsample: {hi.shape}")

    logging.info("\n")

    # now they are image shaped. Be sure to shuffle to de-correlate neighboring samples when training. 
    loader = loader_from_tensor(lo, hi, config.batch_size, shuffle = shuffle)
    return loader, avg_pixel_norm, new_avg_pixel_norm


def make_one_redblue_plot(x, fname):
    plt.ioff()
    fig = plt.figure(figsize=(3,3))
    plt.imshow(x, cmap=sns.cm.icefire, vmin=-2, vmax=2.)
    plt.axis('off')
    plt.savefig(fname, bbox_inches = 'tight')
    plt.close("all")         

def open_redblue_plot_as_tensor(fname):
    return T.ToTensor()(Image.open(fname))

def make_redblue_plots(x, config):
    plt.ioff()
    x = x.cpu()
    bsz = x.size()[0]
    for i in range(bsz):
        make_one_redblue_plot(x[i,0,...], fname = config.home + f'tmp{i}.jpg')
    tensor_img = T.ToTensor()(Image.open(config.home + f'tmp1.jpg'))
    C, H, W = tensor_img.size()
    out = torch.zeros((bsz,C,H,W))
    for i in range(bsz):
        out[i,...] = open_redblue_plot_as_tensor(config.home + f'tmp{i}.jpg')
    return out




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
