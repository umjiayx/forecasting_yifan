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
    make_redblue_plots,
    setup_wandb, 
    DriftModel, 
    bad,
    Config,
    setup_logger,
)



class Trainer:

    def __init__(self, config, load_path = None, sample_only = False, use_wandb = True):

        self.config = config
        c = config

        if sample_only:
            assert load_path is not None

        self.sample_only = sample_only

        c.use_wandb = use_wandb

        self.I = Interpolant(c)

        self.load_path = load_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if c.dataset == 'cifar':
            self.train_loader = get_cifar_dataloader(c)

        elif c.dataset == 'nse':
            self.train_loader, self.val_loader, old_pixel_norm, new_pixel_norm = get_forecasting_dataloader_nse(c)
            c.old_pixel_norm = old_pixel_norm
            c.new_pixel_norm = new_pixel_norm
            # NOTE: if doing anything with the samples other than wandb plotting,
            # e.g. if computing metrics like spectra
            # must scale the output by old_pixel_norm to put it back into data space
            # we model the data divided by old_pixel_norm
        
        elif c.dataset == 'qg':
            self.train_loader, self.val_loader, old_pixel_norm, new_pixel_norm = get_forecasting_dataloader_qg(c)
            c.old_pixel_norm = old_pixel_norm
            c.new_pixel_norm = new_pixel_norm

        else:
            assert False, "dataset must be 'cifar', 'nse', or 'qg'"

        self.overfit_batch = next(iter(self.train_loader))

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
        setup_wandb(c)
        
        self.print_config()  # TODO: uncomment this

    def save_ckpt(self, best_model = False):
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

    @torch.no_grad()
    def definitely_sample(self,):
      
        c = self.config

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
            assert c.dataset == 'nse' or c.dataset == 'qg'
            preprocess_fn = lambda x: to_grid(make_redblue_plots(x, c), c.grid_kwargs)

        logging.info(f"D['z1'].shape: {D['z1'].shape}") # (4, 1, 128, 128)
        logging.info(f"D['z0'].shape: {D['z0'].shape}") # (4, 1, 128, 128)

        z1 = preprocess_fn(D['z1'])
        z0 = preprocess_fn(D['z0'])

        logging.info(f"z1.shape: {z1.shape}") # (3, 508, 506)
        logging.info(f"z0.shape: {z0.shape}") # (3, 508, 506)

        cond = preprocess_fn(D['cond'])

        plotD = {}

        # make samples
        for k in diffusion_fns.keys():
           
            logging.info(f'sampling for diffusion function: {k}')
            sample = self.EM(diffusion_fn=diffusion_fns[k], **EM_args) 

            sample = preprocess_fn(sample)

            all_tensors = torch.cat([z0, sample, z1], dim=-1) 
            
            plotD[k + "(cond, sample, real)"] = wandb.Image(all_tensors)


        assert self.config.use_wandb, "wandb must be supported for sampling"
        wandb.log(plotD, step = self.step)

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
        model_out = self.model(D['zt'], D['t'], D['label'], cond = D['cond'])
        target = D['drift_target']
        return self.image_sq_norm(model_out - target).mean()
    
    def compute_val_loss(self, D):
        assert self.model.eval()
        model_out = self.model(D['zt'], D['t'], D['label'], cond = D['cond'])
        target = D['drift_target']
        return self.image_sq_norm(model_out - target).mean()

    def center(self, x):
        return (x * 2.0) - 1.0

    @torch.no_grad()
    def prepare_batch_nse(self, batch = None, for_sampling = False):

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
    def prepare_batch_qg(self, batch = None, for_sampling = False):
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
    def prepare_batch_cifar(self, batch = None, for_sampling = False):

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
    def prepare_batch(self, batch = None, for_sampling = False):
        if batch is None or self.config.overfit:
            batch = self.overfit_batch

        # get (z0, z1, label, N)
        if self.config.dataset == 'cifar':
            D = self.prepare_batch_cifar(batch, for_sampling = for_sampling) 
        elif self.config.dataset == 'nse':
            D = self.prepare_batch_nse(batch, for_sampling = for_sampling)
        elif self.config.dataset == 'qg':
            D = self.prepare_batch_qg(batch, for_sampling = for_sampling)
        else:
            assert False, "dataset must be 'cifar', 'nse', or 'qg'"

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
            D = self.prepare_batch(batch)
            loss = self.compute_val_loss(D)
            val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        # logging.info(f"[Validation] Step {self.step} | Avg Val Loss: {avg_val_loss:.4f}")

        if self.config.use_wandb:
            wandb.log({'val_loss': avg_val_loss}, step=self.step)

        return avg_val_loss

    def sample_ckpt(self,):
        logging.info("not training. just sampling a checkpoint")
        assert self.config.use_wandb
        self.definitely_sample()
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

    conf = Config(args)
    conf.log_dir = log_dir

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


if __name__ == '__main__':
    main()
