import sys
import torch
import numpy as np

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

def main(config):
    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)

    torch.manual_seed(config.random_seed)
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)

    get_loader(config.data_path, config.batch_size)
    trainer = Trainer(config, rng)
    save_config(config.model_dir, config)

    if config.is_train:
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

    tf.logging.info("Run finished.")

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
