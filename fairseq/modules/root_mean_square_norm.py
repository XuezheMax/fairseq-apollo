# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, number_features, eps=1e-6):
        super().__init__()
        self.num_features = number_features
        self.eps = eps
        self.scale = nn.Parameter(torch.Tensor(self.num_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.scale, 1.0)

    def forward(self, x):
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps) * self.scale
        return x
