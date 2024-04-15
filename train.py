import os
from pytorch_lightning.cli import LightningCLI

import torch
import torch._dynamo
torch.set_float32_matmul_precision('high')
torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.automatic_dynamic_shapes = True  # change to False if training crashes.
torch._dynamo.config.force_parameter_static_shapes = False


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        # Add a custom argument for the checkpoint path
        parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the checkpoint file.')


if __name__ == "__main__":
    # Initialize your custom CLI
    cli = CustomCLI(run=False, save_config_kwargs={"overwrite": True})
    if cli.trainer.num_devices > 1:
        torch.multiprocessing.set_start_method('spawn')

    # Create the logging directory
    os.makedirs(cli.trainer.logger.save_dir, exist_ok=True)
    
    # Access the checkpoint path from the parsed arguments
    ckpt_path = cli.config['ckpt_path'] if 'ckpt_path' in cli.config else None
    
    if ckpt_path:
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
    else:
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
