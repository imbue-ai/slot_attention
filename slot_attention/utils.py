from typing import Any
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import torch
from pytorch_lightning import Callback
from torchvision.transforms import transforms

import wandb

Tensor = TypeVar("torch.tensor")
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


def conv_transpose_out_shape(in_size, stride, padding, kernel_size, out_padding, dilation=1):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1


def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def rescale(x: Tensor) -> Tensor:
    return x * 2 - 1


def compact(l: Any) -> Any:
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


class CoordConv(object):
    def __call__(self, tensor):
        c, H, W = tensor.shape

        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)

        xx, yy = np.meshgrid(x, y)

        return torch.cat([tensor, torch.FloatTensor(xx).unsqueeze(0), torch.FloatTensor(yy).unsqueeze(0)], dim=0)


class ClampImage(object):
    def __call__(self, tensor):
        tensor = tensor.clone()
        img_min = float(tensor.min())
        img_max = float(tensor.max())
        tensor.clamp_(min=img_min, max=img_max)
        tensor.add_(-img_min).div_(img_max - img_min + 1e-5)
        return tensor


class ToImage:
    def __init__(self):
        self.transforms = transforms.Compose([ClampImage(), transforms.ToPILImage()])

    def __call__(self, inputs):
        return self.transforms(inputs)


class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        with torch.no_grad():
            pl_module.eval()
            images = pl_module.sample_images()

        if trainer.logger:
            trainer.logger.experiment.log({"images": [wandb.Image(images)]})
