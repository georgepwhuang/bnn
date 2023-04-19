import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings

from jsonargparse import lazy_instance
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import TqdmExperimentalWarning

from bnn.constants import CONFIG_DIR, LOG_DIR

class BNNCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.dataset_name", "model.init_args.dataset_name")
        parser.link_arguments("data.num_classes", "model.init_args.labels", apply_on="instantiate")
        parser.link_arguments("data.correction_matrix", "model.init_args.correction_matrix", apply_on="instantiate")
        parser.set_defaults({"trainer.logger": lazy_instance(TensorBoardLogger, save_dir=LOG_DIR)})

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
cli = BNNCLI(parser_kwargs={"default_config_files": [str(CONFIG_DIR.joinpath("config.yaml"))]}, run=False)
cli.trainer.fit(cli.model, cli.datamodule)
teacher_path = cli.trainer.checkpoint_callbacks[0].best_model_path

class BNNTSCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("noisy_checkpoint", type=str)
        parser.set_defaults({"trainer.logger": lazy_instance(TensorBoardLogger, save_dir=LOG_DIR), "noisy_checkpoint": teacher_path})
        parser.link_arguments("data.init_args.dataset_name", "model.init_args.dataset_name")
        parser.link_arguments("data.num_classes", "model.init_args.labels", apply_on="instantiate")
        parser.link_arguments("data.correction_matrix", "model.init_args.correction_matrix", apply_on="instantiate")
        parser.link_arguments("noisy_checkpoint", "model.init_args.noisy_checkpoint")
        
cli = BNNTSCLI(parser_kwargs={"default_config_files": [str(CONFIG_DIR.joinpath("student.yaml"))]}, run=False)
cli.trainer.fit(cli.model, cli.datamodule)
cli.trainer.test(cli.model, cli.datamodule, ckpt_path="best")
