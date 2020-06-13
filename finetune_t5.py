import os
import shutil
from datetime import datetime

from absl import app
from absl import flags

from src.trainer import Trainer
from src.utils import load_config


EXPERIMENT_PATH = "experiments"
EXPERIMENT_CONFIG_NAME = "config.yml"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "experiment_name",
    "",
    "Experiment name: experiment outputs will be saved in a created experiment name directory",
)
flags.DEFINE_string("config_path", "t5_config.yml", "Config file path")


def main(argv):
    # load config
    config = load_config(FLAGS.config_path)

    # specify and create experiment path
    timestamp = datetime.now().strftime("_%Y%m%d-%H%M%S")
    current_experiment_path = os.path.join(EXPERIMENT_PATH, FLAGS.experiment_name + timestamp)
    os.makedirs(current_experiment_path)

    # copy current config file to experiment path
    experiment_config_path = os.path.join(current_experiment_path, EXPERIMENT_CONFIG_NAME)
    shutil.copy2(FLAGS.config_path, experiment_config_path)

    # initialize trainer and train
    trainer = Trainer(config, current_experiment_path)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
