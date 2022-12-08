# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout, FairseqFeatureDropout
from fairseq.modules.norm_layer.dual_norm import MaskedBatchNorm


class NormalizedFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_hidden_dim,
        dropout=0.0,
        hidden_dropout=0.0,
        activation='silu',
        norm_affine=True,
        feature_dropout=False,
        export=False,
    ):
        super().__init__()

        self.embedding_dim = embed_dim
        self.hidden_dim = ffn_hidden_dim
        self.act_fn = activation
        self.activation = utils.get_activation_fn(activation)

        dropout_module = FairseqFeatureDropout if feature_dropout else FairseqDropout
        self.dropout = dropout_module(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(hidden_dropout, module_name=self.__class__.__name__)

        self.norm = MaskedBatchNorm(embed_dim, affine=norm_affine)

        self.fc1 = nn.Linear(embed_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, embed_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        # std = 0.02
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        # nn.init.constant_(self.fc1.bias, 0.0)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=std)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, padding_mask=None):
        residual = x
        # fc1
        x = self.activation(self.fc1(x))
        x = self.hidden_dropout(x)
        # fc2
        x = self.fc2(x)
        x = self.norm(x, padding_mask=padding_mask)
        x = self.dropout(x)
        # residual
        x = self.activation(x + residual)

        return x

    def extra_repr(self) -> str:
        return 'edim={}, hdim={}, act={}'.format(self.embedding_dim, self.hidden_dim, self.act_fn)
