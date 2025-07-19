import argparse
import yaml
from utils import get_loader_qg, get_model, maybe_create_dir, evaluate_vae, setup_logger
import torch
import os
import logging

class Trainer:
    def __init__(self, args):
        self.args = args
        self.max_steps = args['max_steps']
        self.step = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.test_loss_history = []
        self.best_val_loss = float('inf')
        self.last_val_loss = None
        self.device = args['device']

        self.trainloader, self.valloader, _ = get_loader_qg(args)
        self.overfit_batch = next(iter(self.trainloader))
        self.overfit_batch = self.overfit_batch.to(self.device)

        logging.info(f"overfit_batch.shape: {self.overfit_batch.shape}")
        logging.info(f"overfit_batch.device: {self.overfit_batch.device}")
        logging.info(f"overfit_batch.dtype: {self.overfit_batch.dtype}")

        self.model = get_model(args)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args['base_lr'], weight_decay=args['weight_decay'])

    def loss_fn(self, x_recon, x):
        # self-reconstruction loss
        return (x_recon-x).square().mean()    


    def training_step(self, batch):
        self.model.train()
        x = batch
        x = x.to(self.device)

        # logging.info(f"x.shape: {x.shape}")

        z = self.model.encoder(x)
        x_recon = self.model.decoder(z)

        loss = self.loss_fn(x_recon, x)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.optimizer_step()
        
        self.train_loss_history.append(loss.item())

        # validate every validate_every steps
        if self.step % self.args['validate_every'] == 0:
            avg_val_loss = self.val_step()
            self.val_loss_history.append(avg_val_loss)
            self.last_val_loss = avg_val_loss

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_model(best_model=True)
                logging.info(f"*** Saving best model! Val loss: {avg_val_loss:.4f} at step {self.step}! ***")
        else:
            if self.last_val_loss is not None:
                self.val_loss_history.append(self.last_val_loss)
            else:
                self.val_loss_history.append(float('nan')) 


    def val_step(self):
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.valloader:
                x = batch
                x = x.to(self.device)
                z = self.model.encoder(x)
                x_recon = self.model.decoder(z)
                loss = self.loss_fn(x_recon, x)
                val_losses.append(loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)

        return avg_val_loss


    def save_model(self, best_model=False):
        D = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }
        if best_model:
            path = f'{self.args["model_path"]}/best_model.pth'
        else:
            path = f'{self.args["model_path"]}/model_{self.step}.pth'
        
        maybe_create_dir(self.args["model_path"])
        torch.save(D, path)


    def do_step(self, batch):
        self.training_step(batch)

        self.step += 1
        if self.step % self.args['print_every'] == 0:
            logging.info(f"Step {self.step}: train loss {self.train_loss_history[-1]:.4f}, val loss {self.val_loss_history[-1]:.4f}.\n")

    def fit(self):
        while self.step < self.max_steps:

            for batch in self.trainloader:
                if self.step >= self.max_steps:
                    return
                
                self.do_step(batch)
        

    


def train(args):
    trainer = Trainer(args)
    trainer.fit()


def evaluate(args):    
    evaluate_vae(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath', help='Provide the filepath string to the model config...')
    
    config = parser.parse_args()
    config_filepath = os.path.join('config', config.config_filepath)
    with open(config_filepath, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    log_dir = setup_logger(log_level=logging.INFO, config_path=config_filepath)
    args['log_dir'] = log_dir

    if args['mode'] == 'eval':
        evaluate(args)
    elif args['mode'] == 'train':
        train(args)
    else:
        raise ValueError(f"Invalid mode: {args['mode']}")