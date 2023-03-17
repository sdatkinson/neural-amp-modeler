import hydra
from omegaconf import DictConfig, OmegaConf

from nam.train.core import train
from nam.train._version import Version


@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # TODO: Move all default parameters to the config file
    model = train(input_path=cfg.common.x_path,     # TODO: Match variable naming in code and config
                  output_path=cfg.common.y_path,
                  train_path=cfg.common.train_path,
                  input_version=Version(1, 1, 1),   # TODO: Detect input version automatically
                  epochs=100,
                  delay=cfg.common.delay,
                  architecture="standard",
                  lr=0.04,
                  lr_decay=0.07,
                  seed=0,
                  )

if __name__ == "__main__":
    main()
    