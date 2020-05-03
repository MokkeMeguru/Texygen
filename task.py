import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.mle.Mle import Mle
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cast_model(gan_name: str):
    if gan_name == 'seqgan':
        return Seqgan
    elif gan_name == 'gsgan':
        return Gsgan
    elif gan_name == 'leakgan':
        return Leakgan
    elif gan_name == 'rankgan':
        return Rankgan
    elif gan_name == 'mailgan':
        return Maligan
    elif gan_name == 'mle':
        return Mle
    elif gan_name == 'textgan':
        return TextganMmd
    else:
        raise KeyError


class Task:
    def __init__(self, hparams):
        self.hparams = hparams

        # setup workspace
        self.setup_workspace()

        # gan setup
        self.gan = cast_model(self.hparams['model'])()
        self.model_setup()

        # training
        if self.hparams['data_type'] == 'oracle':
            self.train_oracle()
        elif self.hparams['data_type'] == 'real':
            self.train_real()
        else:
            self.train_cfg()

    def setup_workspace(self):
        Path('save').mkdir()

    def model_setup(self):
        self.gan.vocab_size = self.hparams['vocab_size']
        self.gan.generate_num = self.hparams['generate_num']

    def train_oracle(self):
        self.gan.train_oracle()

    def train_real(self):
        self.gan.train_real(self.hparams['source'])


@hydra.main(config_path='conf/config.yaml')
def main(cfg: DictConfig):
    logger.info(cfg.pretty())
    Task(cfg)


if __name__ == '__main__':
    main()
