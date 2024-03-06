import logging

import hydra.utils as hu
from hydra import compose, initialize
from transformers import set_seed

logger = logging.getLogger(__name__)


def main(overrides=None):
    if overrides is None:
        import sys

        overrides = sys.argv[1:]

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="base", overrides=overrides)

    logger.info(cfg)
    set_seed(cfg.seed)

    inferencer = hu.instantiate(cfg.inferencer)
    inferencer.inference()


if __name__ == "__main__":
    main()
