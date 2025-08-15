import torch
from torch.utils.data import TensorDataset, DataLoader
import logging
from utils import maybe_lag, maybe_subsample, flatten_time, Constants


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

