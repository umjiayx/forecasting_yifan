import os
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader 
import torchvision.utils
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torchvision import transforms as T
from torchvision.utils import make_grid
from PIL import Image
import math
import argparse
from datetime import datetime
import logging
import yaml # jiayx
from types import SimpleNamespace # jiayx

from interpolant import Interpolant

from utils import (
    is_type_for_logging, 
    to_grid, 
    maybe_create_dir,
    clip_grad_norm, 
    get_cifar_dataloader, 
    get_forecasting_dataloader_nse,
    get_forecasting_dataloader_qg,
    get_forecasting_dataloader_qg_sampling,
    make_redblue_plots,
    setup_wandb, 
    DriftModel, 
    bad,
    Config,
    setup_logger,
)

from measurements import get_operator, get_noiser



class Trainer:

    def __init__(self, config):

        self.config = config
        c = config

        # Logging
        setup_wandb(c)
        
        self.sample_only = c.sample_only
        self.load_path = None
        self.device = c.device

        if self.sample_only:
            assert c.load_path is not None, "load_path must be provided for sampling"
            self.load_path = c.load_path

            # Measurements
            self.operator = get_operator(c)
            self.noiser = get_noiser(c)

            # Auto-regressive steps
            self.auto_step = c.auto_step


        # Stochastic Interpolants
        self.I = Interpolant(c)


        # Datasets
        if c.sample_only:
            self.test_loader, avg_pixel_norm = get_forecasting_dataloader_qg_sampling(c)
            self.test_batch = next(iter(self.test_loader))
        else:
            if c.dataset == 'cifar':
                self.train_loader = get_cifar_dataloader(c)
            elif c.dataset == 'nse':
                self.train_loader, self.val_loader, old_pixel_norm, new_pixel_norm=get_forecasting_dataloader_nse(c)
                c.old_pixel_norm = old_pixel_norm
                c.new_pixel_norm = new_pixel_norm
                # NOTE: if doing anything with the samples other than wandb plotting,
                # e.g. if computing metrics like spectra
                # must scale the output by old_pixel_norm to put it back into data space
                # we model the data divided by old_pixel_norm
            elif 'qg' in c.dataset:
                self.train_loader, self.val_loader, old_pixel_norm, new_pixel_norm=get_forecasting_dataloader_qg(c)
                c.old_pixel_norm = old_pixel_norm
                c.new_pixel_norm = new_pixel_norm
            else:
                assert False, "dataset must be 'cifar', 'nse', or 'qg'"

            self.overfit_batch = next(iter(self.train_loader))


        # Model & Optimizer
        self.model = DriftModel(c)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=c.base_lr)
        self.step = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_ratio = c.val_ratio
        self.best_val_loss = float('inf')
        self.last_val_loss = None
      
        if self.load_path is not None:
            self.load()

        self.U = torch.distributions.Uniform(low=c.t_min_train, high=c.t_max_train)

        # FlowDAS
        self.MC_times = c.MC_times

        # self.print_config() 

    def save_ckpt(self, best_model=False):
        D = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }
        if best_model:
            path = f"{self.config.ckpt_dir}/best.pt"
        else:
            path = f"{self.config.ckpt_dir}/latest.pt"
        maybe_create_dir(self.config.ckpt_dir)
        torch.save(D, path)

    def load(self,):
        D = torch.load(self.load_path)
        self.model.load_state_dict(D['model_state_dict'])
        self.optimizer.load_state_dict(D['optimizer_state_dict'])
        self.step = D['step']
        logging.info(f"loaded! step is {self.step}")

    def print_config(self,):
        logging.info("\n\n********* CONFIG *********\n\n")
        c = self.config
        for key in vars(c):
            val = getattr(c, key)
            if is_type_for_logging(val):
                logging.info(f"{key}: {val}")

    def get_time(self, D):
        D['t'] = self.U.sample(sample_shape = (D['N'],)).to(self.device)
        return D       

    def wide(self, t):
        return t[:, None, None, None] 

    def drift_to_score(self, D):
        z0 = D['z0']
        zt = D['zt']
        at, bt, adot, bdot, bF = D['at'], D['bt'], D['adot'], D['bdot'], D['bF']
        st, sdot = D['st'], D['sdot']
        numer = (-bt * bF) + (adot * bt * z0) + (bdot * zt) - (bdot * at * z0) # Eq (11) in the paper
        denom = (sdot * bt - bdot * st) * st * self.wide(D['t']) # Eq (11) in the paper
        assert not bad(numer)
        assert not bad(denom)
        return numer / denom # Eq (10) in the paper

    def compute_nrmse(self, z1, sample):
        return torch.linalg.norm(z1 - sample) / torch.linalg.norm(z1)

    @torch.no_grad()
    def EM(self, base=None, label=None, cond=None, diffusion_fn=None):
        '''
        Perform the Euler-Maruyama algorithm to sample from the model.

        Args:
            base (torch.Tensor): The base distribution, shape: (B, C, H, W).
            label (torch.Tensor): The label, shape: ???
            cond (torch.Tensor): The condition, shape: (B, C, H, W).
            diffusion_fn (callable): The diffusion function.

        Returns:
            torch.Tensor: The sample estimate from target distribution.
        '''
        c = self.config
        steps = c.EM_sample_steps
        tmin, tmax = c.t_min_sampling, c.t_max_sampling
        ts = torch.linspace(tmin, tmax, steps).type_as(base)
        dt = ts[1] - ts[0]
        ones = torch.ones(base.shape[0]).type_as(base)
 
        # initial condition
        xt = base

        # diffusion_fn = None means use the diffusion function that you trained with
        # otherwise, for a desired diffusion coefficient, do the model surgery to define
        # the correct drift coefficient

        def step_fn(xt, t, label):
            logging.info(f'xt.shape: {xt.shape}')
            logging.info(f't.shape: {t.shape}')

            # assert False, "Stop here"
            D = self.I.interpolant_coefs({'t': t, 'zt': xt, 'z0': base})

            bF = self.model(xt, t, label, cond=cond)
            D['bF'] = bF
            sigma = self.I.sigma(t)

            # specified diffusion func
            if diffusion_fn is not None:
                # Eq (8) in the paper
                g = diffusion_fn(t)
                s = self.drift_to_score(D)
                f = bF + .5 *  (g.pow(2) - sigma.pow(2)) * s

            # default diffusion func
            else:
                f = bF
                g = sigma

            mu = xt + f * dt
            xt = mu + g * torch.randn_like(mu) * dt.sqrt()
            return xt, mu # return sample and its mean

        for i, tscalar in enumerate(ts):
            
            if i == 0 and (diffusion_fn is not None):
                # only need to do this when using other diffusion coefficients that you didn't train with
                # because the drift-to-score conversion has a denominator that features 0 at time 0
                # if just sampling with "sigma" (the diffusion coefficient you trained with) you
                # can skip this
                tscalar = ts[1] # 0 + (1/500)

            if (i+1) % 100 == 0:
                logging.info(f"Step {i+1} of total {steps} steps...")

            xt, mu = step_fn(xt, tscalar*ones, label = label)
        assert not bad(mu)
        return mu

    def taylor_est_x1(self, xt, t, bF, g, use_original_sigma = True, analytical = True):
        if use_original_sigma == True and analytical == False:
            hat_x1 = xt + bF * (1-t) + g * torch.randn_like(xt) * (1-t).sqrt()
        elif use_original_sigma == True and analytical == True:
            hat_x1 = xt + bF * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
        return hat_x1.requires_grad_(True)

    def taylor_est2rd_x1(self, xt, t, bF, g, label, cond,use_original_sigma = True, analytical = True):
        '''
        xt: (B, C=1, H, W)
        t: (B,)
        bF: (B, C=1, H, W)
        g: (B, 1, 1, 1)
        cond: (B, C=1, H, W)
        '''
        MC_times = self.MC_times
        
        if use_original_sigma == True and analytical == False:
            hat_x1 = xt + bF * (1-t) + g * torch.randn_like(xt) * (1-t).sqrt()
        elif use_original_sigma == True and analytical == True and MC_times == 1:
            hat_x1 = xt + bF * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
            t1 = torch.FloatTensor([1])
            bF2 = self.model(hat_x1,t1.to(hat_x1.device),label,cond=cond).requires_grad_(True)
            hat_x1 =  xt + (bF + bF2)/2 * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
            return hat_x1.requires_grad_(True)
        elif use_original_sigma == True and analytical == True and MC_times != 1:
            hat_x1 = xt + bF * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
            t1 = torch.FloatTensor([1])
            bF2 = self.model(hat_x1,t1.to(hat_x1.device),label,cond=cond).requires_grad_(True)
            hat_x1_list = []
            for i in range(MC_times):
                hat_x1 =  xt + (bF + bF2)/2 * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
                hat_x1_list.append(hat_x1.requires_grad_(True))
            return hat_x1_list

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
            # print('if require grad',x_prev.requires_grad,x_0_hat.requires_grad)
        if isinstance(x_0_hat, torch.Tensor):
            assert 1==0
            difference = (measurement - self.noiser(self.operator(x_0_hat))).requires_grad_(True)
            norm = torch.linalg.norm(difference).requires_grad_(True)
            print('diff',norm)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, allow_unused=True)[0]
        else:
            difference = 0
            for i in range(len(x_0_hat)):
                difference +=(measurement - self.operator(x_0_hat[i])).requires_grad_(True)
            difference = difference/len(x_0_hat)
            # print('difference',difference)
            norm = torch.linalg.norm(difference).requires_grad_(True)
            # logging.info(f'difference norm: {norm}')
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, allow_unused=True)[0]
        return norm_grad, norm
    
    # @torch.no_grad(), we need to compute the grad like DPS.
    def EM_flowdas(self, measurement=None,base=None, label=None, cond=None, diffusion_fn=None):
        '''
        Perform the Euler-Maruyama+DPS (flowdas) algorithm to sample from the model.
        '''
        c = self.config
        steps = c.EM_sample_steps
        tmin, tmax = c.t_min_sampling, c.t_max_sampling
        ts = torch.linspace(tmin, tmax, steps)
        dt = ts[1] - ts[0]
        ones = torch.ones(base.shape[0])

        # initial condition
        # logging.info(f'base.shape: {base.shape}')
        # logging.info(f'measurement.shape: {measurement.shape}')
        # logging.info(f'cond.shape: {cond.shape}')

        xt = base.requires_grad_(True)

        def step_fn(xt, t, label, measurement):
            '''
            xt: (B, C=1, H, W)
            t: (B,)
            measurement: (B, C, H, W)
            '''
            #logging.info(f'xt.shape: {xt.shape}')
            #logging.info(f't.shape: {t.shape}')
            #logging.info(f'measurement.shape: {measurement.shape}')
            
            D = self.I.interpolant_coefs({'t': t, 'zt': xt, 'z0': base})

            t = t.numpy()
            t = torch.FloatTensor(t)
            t = t.to(xt.device)

            bF = self.model(xt, t, label, cond=cond)
            D['bF'] = bF
            sigma = self.I.sigma(t)

            #logging.info(f'bF.requires_grad: {bF.requires_grad}') # False
            #logging.info(f'xt.requires_grad: {xt.requires_grad}') # True
            
            if diffusion_fn is not None:
                g = diffusion_fn(t)
                s = self.drift_to_score(D)
                f = bF + .5 *  (g.pow(2) - sigma.pow(2)) * s
            else:
                f = bF
                g = sigma
            
            scale = 1

            #logging.info(f'bF.shape: {bF.shape}') # (B, C=1, H, W)
            #logging.info(f'sigma.shape: {sigma.shape}') # (B, 1, 1, 1)
            #logging.info(f't.shape: {t.shape}') # (B,)

            
            es_x1 = self.taylor_est2rd_x1(xt, t, bF, g, label, cond)
            norm_grad, norm = self.grad_and_value(xt, es_x1, measurement)
            
            # assert False, "Stop here"
            
            mu = xt + f*dt

            if norm_grad is None:
                norm_grad = 0
                logging.info(f'No grad!')

            xt = mu + g*torch.randn_like(mu)*dt.sqrt() - scale*norm_grad

            # xt = xt.detach().clone().requires_grad_(True) # TODO: ????

            return xt, mu



        for i, tscalar in enumerate(ts):
            if i == 0 and (diffusion_fn is not None):
                tscalar = ts[1]
            
            if (i+1) % 100 == 0:
                logging.info(f"EM step {i+1} of total {steps} steps...")
            xt, mu = step_fn(xt, tscalar*ones, label=label, measurement=measurement)

        assert not bad(mu)
        return mu

    @torch.no_grad()
    def definitely_sample(self,):
        '''
        1. Training Phase: sampling during validation steps
        2. Inference Phase: pure forecasting for only one time step
        '''
        c = self.config
        assert c.auto is False, "Sampling is not supported for auto mode"

        logging.info("SAMPLING")

        self.model.eval()
        
        D = self.prepare_batch(batch = None, for_sampling = True)

        EM_args = {
            'base': D['z0'], 
            'label': D['label'], 
            'cond': D['cond']
            }
       
        # list diffusion funcs
        # None means use the one you trained with
        diffusion_fns = {
            'g_sigma': None,
            # 'g_other': lambda t: c.sigma_coef * self.wide(1-t).pow(4),
        }
       
        if c.dataset == 'cifar':
            preprocess_fn = lambda x : to_grid(x, c.grid_kwargs)

        else:
            assert c.dataset == 'nse' or 'qg' in c.dataset
            preprocess_fn = lambda x, name: to_grid(make_redblue_plots(x, c, name), c.grid_kwargs)

        logging.info(f"D['z1'].shape: {D['z1'].shape}") # (4, 1, 128, 128)
        logging.info(f"D['z0'].shape: {D['z0'].shape}") # (4, 1, 128, 128)

        z1_plot = preprocess_fn(D['z1'], name='z1') # (3, 508, 506)
        z0_plot = preprocess_fn(D['z0'], name='z0') # (3, 508, 506)

        logging.info(f"z1_plot.shape: {z1_plot.shape}") # (3, 508, 506)
        logging.info(f"z0_plot.shape: {z0_plot.shape}") # (3, 508, 506)

        cond_plot = preprocess_fn(D['cond'], name='cond')

        plotD = {}

        # make samples
        for k in diffusion_fns.keys():
           
            logging.info(f'sampling for diffusion function: {k}')
            sample = self.EM(diffusion_fn=diffusion_fns[k], **EM_args) 

            sample_plot = preprocess_fn(sample, name='sample')

            all_tensors = torch.cat([z0_plot, sample_plot, z1_plot], dim=-1) 

            logging.info(f"all_tensors.shape: {all_tensors.shape}")
            
            plotD[k + "(cond, sample, real)"] = wandb.Image(all_tensors)


        assert self.config.use_wandb, "wandb must be supported for sampling"
        wandb.log(plotD, step = self.step)

    # @torch.no_grad() This is not good!!!! Ahhhhhhh...
    def autoregressive_sample(self,):
        c = self.config
        assert c.sample_only and c.auto, "Autoregressive sampling is only supported when both sampling only mode and auto mode are enabled"

        logging.info(f"*** AUTOREGRESSIVE SAMPLING! ***")
        self.model.eval()

        assert c.sampling_batch_size == 1, "Sampling batch size must be 1 for autoregressive sampling, will change this later"
        
        batch = self.test_batch

        D = self.prepare_batch_autoregressive(batch=batch)

        logging.info(f"D['z0'].shape: {D['z0'].shape}") # (auto_step, C, H, W)

        EM_args = {
            'base': D['z0'],  # (auto_step, C, H, W)
            'label': D['label'], 
            'cond': D['cond']
            }
        
        diffusion_fns = {
            'g_sigma': None,
            # 'g_other': lambda t: c.sigma_coef * self.wide(1-t).pow(4),
        }

        measurements = self.operator(D['z1']) # (B=auto_step, C, H, W)
        measurements = self.noiser(measurements)

        # logging.info(f"D['z0'].shape: {D['z0'].shape}") # (B, C=1, H, W)  
        # logging.info(f"D['z1'].shape: {D['z1'].shape}")
        # logging.info(f"measurements.shape: {measurements.shape}")

        if c.dataset == 'cifar':
            preprocess_fn = lambda x : to_grid(x, c.grid_kwargs)
        else:
            assert c.dataset == 'nse' or 'qg' in c.dataset
            preprocess_fn = lambda x, name: to_grid(make_redblue_plots(x, c, name), c.grid_kwargs)

        sample = None
        all_samples = []
        all_tensors = []
        plotD = {}
        nrmse_list = []

        for step in range(c.auto_step):
            logging.info(f"********** Time step {step+1} of total {c.auto_step} steps... **********")
            
            if step >= 1:
                assert sample is not None, "sample is not initialized"
                assert step > 0, "autoregressive step should be greater than 0"
                # autoregressive step
                cond = sample # (B, C, H, W)
                z0 = cond # (B, C, H, W)
            else:
                assert sample is None, "initial step should not have sample yet"
                assert step == 0, "initial step should be 0"
                cond = D['cond'][step].unsqueeze(0) # (1, C, H, W)
                z0 = D['z0'][step].unsqueeze(0) # (1, C, H, W)
            
            # generate sample:
            sample = self.EM_flowdas(diffusion_fn=diffusion_fns['g_sigma'], 
                             measurement=measurements[step].unsqueeze(0),
                             base=z0,
                             cond=cond) # (B, C, H, W)
            all_samples.append(sample.detach().clone())

    
        # Plot (z1, measurement, sample) for all steps
        for step in range(c.auto_step):
            z1 = D['z1'][step].unsqueeze(0) # (1, C, H, W)
            measurement = measurements[step].unsqueeze(0) # (1, C, H, W)
            sample = all_samples[step] # (1, C, H, W)
            error = z1 - sample # (1, C, H, W)

            # logging.info(f"error.max(): {error.square().sqrt().max().item():.4f}")

            nrmse = self.compute_nrmse(z1, sample)
            nrmse_list.append(nrmse)
            wandb.log({f"nrmse_step": nrmse}, step=step)

            # logging.info(f"z1.shape: {z1.shape}")
            # logging.info(f"measurement.shape: {measurement.shape}")
            # logging.info(f"sample.shape: {sample.shape}")

            z1_plot = preprocess_fn(z1, 'z1') # (3, 251, 250)
            measurement_plot = preprocess_fn(measurement, 'measurement') # (3, 251, 250)
            sample_plot = preprocess_fn(sample, 'sample') # (3, 251, 250)
            error_plot = preprocess_fn(error, 'error') # (3, 251, 250)

            # logging.info(f"z1_plot.shape: {z1_plot.shape}")
            # logging.info(f"measurement_plot.shape: {measurement_plot.shape}")
            # logging.info(f"sample_plot.shape: {sample_plot.shape}")

            all_tensors_step = torch.cat([measurement_plot, z1_plot, sample_plot, error_plot], dim=-2) 
            # logging.info(f"all_tensors_step.shape: {all_tensors_step.shape}")

            all_tensors.append(all_tensors_step)
        

        if len(all_tensors) > 1:
            all_tensors = torch.cat(all_tensors, dim=-1)
        else:
            all_tensors = all_tensors[0]
        
        # logging.info(f"all_tensors.dtype: {all_tensors.dtype}")
        # logging.info(f"all_tensors.device: {all_tensors.device}")
        # logging.info(f"all_tensors.shape: {all_tensors.shape}")

        plotD['g_sigma' + "(GT, measurement, FlowDAS)"] = wandb.Image(all_tensors)

        # log the nrmse_list to wandb and see the curve
        # plotD['nrmse'] = wandb.Image(nrmse_list)
        
        assert self.config.use_wandb, "wandb must be supported for sampling"
        wandb.log(plotD, step=self.step)

        # print all the nrmse in the list
        for i, nrmse in enumerate(nrmse_list):
            logging.info(f"nrmse at step {i+1}: {nrmse.item()*100:.4f}%")
        
        
        '''
        step = 0 # TODO: Remove this
        z1_plot = preprocess_fn(D['z1'], 'z1')
        z0_plot = preprocess_fn(D['z0'], 'z0')
        measurement_plot = preprocess_fn(measurements, 'measurement')     

        sample = self.EM_flowdas(diffusion_fn=diffusion_fns['g_sigma'], 
                             measurement = measurements[step].unsqueeze(0),
                             base = D['z0'][step].unsqueeze(0),
                             cond = D['cond'][step].unsqueeze(0)) 
        sample = sample.detach().clone()

        plotD = {}
        sample_plot = preprocess_fn(sample, 'sample')

        all_tensors_step = torch.cat([z0_plot, measurement_plot, sample_plot, z1_plot], dim=-1) 
        
        logging.info(f"all_tensors_step.dtype: {all_tensors_step.dtype}")
        logging.info(f"all_tensors_step.device: {all_tensors_step.device}")
        logging.info(f"all_tensors_step.shape: {all_tensors_step.shape}")
        
        plotD['g_sigma' + "(cond, measurement, sample, real)"] = wandb.Image(all_tensors_step)

        assert self.config.use_wandb, "wandb must be supported for sampling"
        wandb.log(plotD, step = self.step)
        '''
    
    @torch.no_grad()
    def maybe_sample(self,):
        is_time = self.step % self.config.sample_every == 0
        is_logging = self.config.use_wandb
        if is_time and is_logging:
            self.definitely_sample()

    def optimizer_step(self,):
        norm = clip_grad_norm( # TODO: What is this?
            self.model, 
            max_norm = self.config.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # self.step += 1
        return norm

    def image_sq_norm(self, x):
        return x.pow(2).sum(-1).sum(-1).sum(-1)

    def training_step(self, D):
        assert self.model.training
        model_out = self.model(D['zt'], D['t'], D['label'], cond=D['cond'])
        target = D['drift_target']
        return self.image_sq_norm(model_out - target).mean()
    
    def compute_val_loss(self, D):
        assert self.model.eval()
        model_out = self.model(D['zt'], D['t'], D['label'], cond=D['cond'])
        target = D['drift_target']
        return self.image_sq_norm(model_out - target).mean()

    def center(self, x):
        return (x * 2.0) - 1.0

    @torch.no_grad()
    def prepare_batch_nse(self, batch=None, for_sampling=False):

        assert not self.config.center_data

        xlo, xhi = batch

        if for_sampling:
            xlo = xlo[:self.config.sampling_batch_size]
            xhi = xhi[:self.config.sampling_batch_size]

        xlo, xhi = xlo.to(self.device), xhi.to(self.device)

        N = xlo.shape[0]
        y = None
        D = {'z0': xlo, 'z1': xhi, 'label': y, 'N': N}
        return D
    
    @torch.no_grad()
    def prepare_batch_qg(self, batch=None, for_sampling=False):
        assert not self.config.center_data

        xlo, xhi = batch

        if for_sampling:
            # When running validation, we do not need this. Needed when sampling only.
            xlo = xlo[:self.config.sampling_batch_size]
            xhi = xhi[:self.config.sampling_batch_size]

        xlo, xhi = xlo.to(self.device), xhi.to(self.device)

        N = xlo.shape[0]
        y = None
        D = {'z0': xlo, 'z1': xhi, 'label': y, 'N': N}
        return D

    @torch.no_grad()
    def prepare_batch_cifar(self, batch=None, for_sampling=False):

        x, y = batch

        if for_sampling:
            x = x[:self.config.sampling_batch_size]
            y = y[:self.config.sampling_batch_size]

        x, y = x.to(self.device), y.to(self.device)

        # possibly center the data, e.g., for images, from [0,1] to [-1,1]
        z1 = self.center(x) if self.config.center_data else x

        D = {'N': z1.shape[0], 'label': y, 'z1': z1}
       
        # point mass base density 
        # since we don't have any conditioning info for this cifar test
        # for PDEs, could set z0 to the previous known condition.
        D['z0'] = torch.zeros_like(D['z1'])

        return D

    @torch.no_grad()
    def prepare_batch(self, batch=None, for_sampling=False):
        if self.config.overfit:
            assert self.sample_only is False, "Overfit mode is not supported for sampling"
            assert self.config.auto is False, "Auto mode should not be implemented in this method"
            batch = self.overfit_batch
        
        if self.config.sample_only:
            assert self.config.overfit is False, "Overfit mode is not supported for sampling"
            assert self.config.auto is False, "Auto mode should not be implemented in this method"
            batch = self.test_batch

        # get (z0, z1, label, N)
        if self.config.dataset == 'cifar':
            D = self.prepare_batch_cifar(batch, for_sampling=for_sampling) 
        elif self.config.dataset == 'nse':
            D = self.prepare_batch_nse(batch, for_sampling=for_sampling)
        elif 'qg' in self.config.dataset:
            D = self.prepare_batch_qg(batch, for_sampling=for_sampling)
        else:
            assert False, "dataset must be 'cifar', 'nse', or 'qg'"

        D = self.update_D(D)
   
        return D

    @torch.no_grad()
    def prepare_batch_autoregressive(self, batch=None):
        '''
        batch: (xlo, xhi)
        xlo.shape: (batch_size, C, H, W)
        xhi.shape: (batch_size, C, H, W)

        Return:
        D: {'z0': xlo, 'z1': xhi, 'label': y, 'N': N}
        D['z0'].shape: (auto_step, C, H, W)
        D['z1'].shape: (auto_step, C, H, W)
        '''
        c = self.config
        xlo, xhi = batch

        assert c.auto_step < c.batch_size, "Auto step must be less than batch size"

        # generate a random starting index, and make sure the ending index is not out of bounds
        start_idx = torch.randint(0, c.batch_size - c.auto_step, (1,))
        end_idx = start_idx + c.auto_step

        xlo = xlo[start_idx:end_idx]
        xhi = xhi[start_idx:end_idx]

        logging.info(f"Looking at test batch: start_idx: {start_idx}, end_idx: {end_idx} out of batch size {c.batch_size}.")

        xlo, xhi = xlo.to(self.device), xhi.to(self.device)
        N = xlo.shape[0]
        y = None

        D = {'z0': xlo, 'z1': xhi, 'label': y, 'N': N}

        D = self.update_D(D)

        return D

    def update_D(self, D):
        '''
        Update D when preparing the batch for training or sampling.
        '''
        # get random batch of times
        D = self.get_time(D)

        # conditioning in the model is the initial condition
        D['cond'] = D['z0']

        # interpolant noise
        D['noise'] = torch.randn_like(D['z0'])

        # get alpha, beta, etc
        D = self.I.interpolant_coefs(D)
       
        D['zt'] = self.I.compute_zt(D)
        
        D['drift_target'] = self.I.compute_target(D)

        return D

    @torch.no_grad()
    def run_validation(self):
        self.model.eval()
        val_losses = []

        for batch in self.val_loader:
            # batch.shape: (batch_size, C, H, W)
            D = self.prepare_batch(batch)
            loss = self.compute_val_loss(D)
            val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        # logging.info(f"[Validation] Step {self.step} | Avg Val Loss: {avg_val_loss:.4f}")

        if self.config.use_wandb:
            wandb.log({'val_loss': avg_val_loss}, step=self.step)

        return avg_val_loss

    def sample_ckpt(self,):
        logging.info(f"Sampling using a checkpoint: {self.load_path}")
        assert self.config.use_wandb, "wandb must be supported for sampling"
        self.definitely_sample()
        logging.info("DONE")
    
    def sample_ckpt_auto(self,):
        logging.info(f"Autoregressively sampling using a checkpoint: {self.load_path}")
        assert self.config.use_wandb, "wandb must be supported for sampling"
        self.autoregressive_sample()
        logging.info("DONE")

    def do_step(self, batch_idx, batch):
        # Prepare batch and compute training loss
        D = self.prepare_batch(batch)
        self.model.train()
        loss = self.training_step(D)
        loss.backward()
        self.train_loss_history.append(loss.item())
        grad_norm = self.optimizer_step()
        self.maybe_sample()

        # Check if we should run validation
        if self.step % self.config.validate_every == 0:
            val_loss = self.run_validation()
            self.val_loss_history.append(val_loss)
            self.last_val_loss = val_loss  # update the last known val loss

            if val_loss < self.best_val_loss:
                logging.info(f"*** Saving best model! Val loss improved to {val_loss:.4f} at step {self.step}. ***")
                self.save_ckpt(best_model=True)
                self.best_val_loss = val_loss
        else:
            # Reuse last validation loss to keep histories aligned
            if self.last_val_loss is not None:
                self.val_loss_history.append(self.last_val_loss)
            else:
                self.val_loss_history.append(float("nan"))  # TODO: check if there is nan, I think this is not needed

        # Logging
        if self.step % self.config.print_loss_every == 0:
            logging.info(f"Step {self.step} | Training loss: {loss.item():.4f} | Val loss: {self.val_loss_history[-1]:.4f}\n")
            if self.config.use_wandb:
                wandb.log({
                    'training_loss': loss.item(),
                    'val_loss': self.val_loss_history[-1],
                    'grad_norm': grad_norm
                }, step=self.step)

        # Save checkpoint regularly
        if self.step % self.config.save_every == 0:
            logging.info(f"Regularly saving model at step {self.step}\n")
            self.save_ckpt(best_model=False)

        self.step += 1

        # save the training and validation loss history to the log folder as a npy file
        np.save(f"{self.config.log_dir}/train_loss_history.npy", np.array(self.train_loss_history))
        np.save(f"{self.config.log_dir}/val_loss_history.npy", np.array(self.val_loss_history))

    def fit(self,):

        is_logging = self.config.use_wandb

        if is_logging:
            self.definitely_sample()

        logging.info("\n\n********* STARTING TRAINING *********\n\n")
        while self.step < self.config.max_steps:

            for batch_idx, batch in enumerate(self.train_loader):
 
                if self.step >= self.config.max_steps:
                    return

                self.do_step(batch_idx, batch)



def main():
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # random seed
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parsed_args = parser.parse_args()

    # make the path to be the folder 'configs' + the config file name
    config_path = os.path.join('configs', parsed_args.config)
    
    log_dir = setup_logger(config_path=config_path)

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    args = SimpleNamespace(**config_dict)

    logging.info(f"********* RUNNING at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using config {parsed_args.config} *********")

    config = Config(args)
    config.log_dir = log_dir

    trainer = Trainer(config)

    if bool(args.sample_only):
        if args.auto:
            trainer.sample_ckpt_auto()
        else:
            trainer.sample_ckpt()
    else:
        trainer.fit()


if __name__ == '__main__':
    main()

