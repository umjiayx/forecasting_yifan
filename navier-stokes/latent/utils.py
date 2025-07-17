import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import *
from torch.utils.data import DataLoader
import os
from nn import ConvEncoderDecoder_with_stride
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import shutil

def get_model(args):
    in_features = args['input_size'] * args['window']
    hidden_channels = args['latent_channels']
    kernel_sizes = args['kernel_sizes']

    vae = ConvEncoderDecoder_with_stride(
        in_features=in_features,
        hidden_channels=hidden_channels,
        kernel_sizes=kernel_sizes,
    ).to(args['device'])
    
    print(vae)
    logging.info(f"Total parameters: {sum(p.numel() for p in vae.parameters() if p.requires_grad)}")
    
    return vae


def maybe_create_dir(f):
    if not os.path.exists(f):
        logging.info("making", f)
        os.makedirs(f)


class TrajectoryDataset(Dataset):
    '''
    data: (N, T, C, H, W)
    window: int
    flatten: bool
    '''
    def __init__(self, data, window: int=None, flatten: bool=False):
        super().__init__()
        self.data = data
        self.window = window
        self.flatten = flatten

        assert data.ndim == 5, "data must be of shape (N, T, C, H, W)"

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[Tensor, Dict]:
        
        # pick the idx-th trajectory
        x = self.data[idx]

        if self.window is not None:
            start_idx = torch.randint(0, len(x) - self.window + 1, ())
            x = torch.narrow(x, dim=0, start=start_idx, length=self.window)
        
        if self.flatten:
            return x.flatten(0, 1), {}
        else:
            return x, {}


def compute_avg_pixel_norm(data_raw):
    return torch.sqrt(torch.mean(data_raw ** 2))

def get_loader_qg(args):
    window = args['window']
    flatten = args['flatten']
    batch_size = args['batch_size']
    num_workers = args['num_workers']
    data = torch.load(args['data_path'])
    
    logging.info("data.shape", data.shape)

    data = data.float()

    avg_pixel_norm = compute_avg_pixel_norm(data)
    logging.info("avg_pixel_norm", avg_pixel_norm) # 0.6352

    data = data/avg_pixel_norm
    new_avg_pixel_norm = 1.0
    data = data * new_avg_pixel_norm

    # (N, T, H, W) -> (N, T, C, H, W)
    data = data.unsqueeze(2)

    i = int(args['train_ratio'] * data.shape[0]) # 0.8
    j = int(args['val_ratio'] * data.shape[0]) # 0.1

    # shape: (N, T, C, H, W) 
    train_data = data[:i]
    val_data = data[i:i+j]
    test_data = data[i+j:]

    trainset = TrajectoryDataset(train_data, window=window, flatten=flatten)
    valset = TrajectoryDataset(val_data, window=window, flatten=flatten)
    testset = TrajectoryDataset(test_data, window=window, flatten=flatten)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return trainloader, valloader, testloader



def evaluate_vae(args):

    model = get_model(args)
    model.load_state_dict(torch.load(args['model_path'] + '/best_model.pth')['model_state_dict'])
    model.to(args['device'])

    model.eval()

    x = torch.load(args['data_path']).float()
    # avg_pixel_norm = torch.sqrt(torch.mean(x ** 2))
    avg_pixel_norm = 0.6352 # a priori knowledge
    x = x / avg_pixel_norm

    j = int((1-args['train_ratio']-args['val_ratio']) * len(x))
    test_x = x[j:]
    test_x = test_x.unsqueeze(2)

    logging.info("test_x.shape", test_x.shape)
    
    pass
    
    timestep = 0
    sample_id = 0

    sample = test_x[sample_id]

    xt = sample[timestep].unsqueeze(0)
    z = model.encoder(xt.to(args['device']))
    x_ = model.decoder(z).squeeze(0)
    logging.info("x_.shape", x_.shape)

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(xt.squeeze().detach().cpu(), cmap=sns.cm.icefire)  # shape: H x W
    ax[0].set_title("Original")
    ax[1].imshow(x_.squeeze().detach().cpu(), cmap=sns.cm.icefire)
    ax[1].set_title("Reconstruction")
    plt.tight_layout()
    plt.savefig('./logs/sample_and_x_.png')
    logging.info("Saved reconstruction figure to ./logs/sample_and_x_.png")




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

    return run_dir