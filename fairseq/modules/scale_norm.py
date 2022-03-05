# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scalar = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.scalar, 1.0)

    def forward(self, x):
        mean_square = torch.mean(torch.square(x), dim=self.dim, keepdim=True)
        x = self.scalar * x * torch.rsqrt(mean_square + self.eps)
        return x
