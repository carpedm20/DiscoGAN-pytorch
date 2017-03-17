import torch

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

def main(config):
    prepare_dirs_and_logger(config)

    torch.manual_seed(config.random_seed)
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)

    a_data_loader, b_data_loader = get_loader(
            config.data_path, config.batch_size, config.input_scale_size, config.num_worker)
    trainer = Trainer(config, a_data_loader, b_data_loader)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

    tf.logging.info("Run finished.")

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
