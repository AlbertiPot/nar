import os
import sys
import torch
import random
import logging
import numpy as np


def setup_seed(seed=20211117):
    # there are still other seed to set, NASBenchDataset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_logger(save_path=None, mode='a') -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s]: %(message)s",
                                  datefmt="%m/%d %H:%M:%S")

    if save_path is not None:
        if os.path.exists(save_path):
            os.remove(save_path)
        log_file = open(save_path, 'w')
        log_file.close()

        file_handler = logging.FileHandler(save_path, mode=mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
