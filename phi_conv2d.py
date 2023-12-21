
import torch
import torch.nn as nn
from torch import Tensor


class phi_conv(nn.Conv2d):
    def __init__(self, init=True, *args, **kwargs):
        self.a = 1  # the propagation loss from the ring and the directional coupler
        # r1 and r2: the coupling coefficients of the add-drop MRR
        self.r1 = 0.9
        self.r2 = 0.9
        self.init = init
        super(phi_conv, self).__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        """The initialization method of the PB convolution kernels"""
        if self.init:
            nn.init.uniform_(self.weight, -0.63, -0.13)
        else:
            
            nn.init.uniform_(self.weight, -torch.pi, 0)

    def _through(self):
        """The output intensity of the Through port"""
        num = (self.r2 * self.a) ** 2 - 2 * self.r1 * self.r2 * self.a * torch.cos(self.weight) + self.r1 ** 2  # noqa
        denom = 1 - 2 * self.r1 * self.r2 * self.a * torch.cos(self.weight) + (self.r1 * self.r2 * self.a) ** 2  # noqa

        return num / denom

    def _dropout(self):
        """The output intensity of the Drop port"""
        num = (1 - self.r1 ** 2) * (1 - self.r2 ** 2) * self.a
        denom = 1 - 2 * self.r1 * self.r2 * self.a * torch.cos(self.weight) + (self.r1 * self.r2 * self.a) ** 2  # noqa

        return num / denom

    def final(self):
        """""The output intensity of the (Drop-Through) port""""""
        return self._dropout() - self._through()
        

    def forward(self, input: Tensor) -> Tensor:
        """the construction of the PB convolution kernel"""
        w = self.final()
        out = self._conv_forward(input, w, self.bias)

        return out




