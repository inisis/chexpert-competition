import sys
import os
import argparse

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset import Dataset  # noqa
from model.classifier import Classifier  # noqa
from model.utils import get_optimizer, lr_schedule  # noqa
from training.trainer import Trainer  # noqa

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--num_workers', default=1, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str, help="GPU indices "
                    "comma separated, e.g. '0,1' ")
parser.add_argument('--logtofile', default=False, type=bool, help="Save log "
                    "in save_path/log.txt if set True")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")


def run(args):
    trainer = Trainer(args)
    trainer.setup()
    cfg = trainer.cfg

    epoch_steps_train = len(trainer.dataloader_train)
    total_steps_train = epoch_steps_train * cfg.epoch

    for step in range(total_steps_train):
        trainer.train_step()

        if (step + 1) % cfg.log_every == 0:
            trainer.logging('Train')
            trainer.write_summary('Train')
            trainer.log_init()

        if (step + 1) % cfg.dev_every == 0:
            trainer.dev_epoch()
            trainer.logging('Dev')
            trainer.write_summary('Dev')
            trainer.save_model('Train')
            trainer.save_model('Dev')

    trainer.close()


def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)


if __name__ == '__main__':
    main()
