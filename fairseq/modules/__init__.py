# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .adaptive_input import AdaptiveInput
from .adaptive_softmax import AdaptiveSoftmax
from .beamable_mm import BeamableMM
from .character_token_embedder import CharacterTokenEmbedder
from .conv_tbc import ConvTBC
from .cross_entropy import cross_entropy
from .downsampled_multihead_attention import DownsampledMultiHeadAttention
from .dynamic_convolution import DynamicConv, DynamicConv1dTBC
from .dynamic_crf_layer import DynamicCRF
from .fairseq_dropout import FairseqDropout, FairseqFeatureDropout
from .fp32_group_norm import Fp32GroupNorm
from .gelu import gelu, gelu_accurate
from .grad_multiply import GradMultiply
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .kmeans_vector_quantizer import KmeansVectorQuantizer
from .layer_drop import LayerDropModuleList
from .norm_layer.layer_norm import Fp32LayerNorm, LayerNorm
from .norm_layer.scale_norm import ScaleNorm
from .norm_layer.root_mean_square_norm import RMSNorm
from .norm_layer.masked_batch_norm import MaskedBatchNorm
from .norm_layer.sequence_norm import SequenceNorm
from .learned_positional_embedding import LearnedPositionalEmbedding
from .real_number_embedding import RealNumberEmbedding
from .positional_embedding import PositionalEmbedding
from .relative_positional_bias import SimpleRelativePositionalBias, RotaryRelativePositionalBias
from .lightweight_convolution import LightweightConv, LightweightConv1dTBC
from .linearized_convolution import LinearizedConvolution
from .multihead_attention import MultiheadAttention
from .luna_attention import LunarMultiheadAttention, LunarCausalAttention
from .luna_sentence_encoder import LunaSentenceEncoder, LunaSentenceEncoderLayer
from .base_moving_average import BaseMovingLayer
from .exponential_moving_average import MultiHeadEMA
from .complex_exponential_moving_average import MultiHeadComplexEMA
from .diagonal_linear_rnn import DiagonalLinearRNN
from .moving_average_gated_attention import MovingAverageGatedAttention
from .gated_attention_unit import GatedAttentionUnit
from .gated_cross_attention import GatedCrossAttention
from .flash_sentence_encoder_layer import FlashSentenceEncoderLayer
from .mega_sentence_encoder_layer import MegaSentenceEncoderLayer
from .same_pad import SamePad
from .scalar_bias import ScalarBias
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from .transformer_sentence_encoder import TransformerSentenceEncoder
from .transpose_last import TransposeLast
from .unfold import unfold1d
from .transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from .luna_layer import LunaEncoderLayer, LunaDecoderLayer
from .mega_layer import MegaEncoderLayer, MegaDecoderLayer
from .vggblock import VGGBlock

__all__ = [
    'AdaptiveInput',
    'AdaptiveSoftmax',
    'BeamableMM',
    'CharacterTokenEmbedder',
    'ConvTBC',
    'cross_entropy',
    'DownsampledMultiHeadAttention',
    'DynamicConv1dTBC',
    'DynamicConv',
    'DynamicCRF',
    'FairseqDropout',
    'Fp32GroupNorm',
    'Fp32LayerNorm',
    'gelu',
    'gelu_accurate',
    'GradMultiply',
    'GumbelVectorQuantizer',
    'KmeansVectorQuantizer',
    'LayerDropModuleList',
    'LayerNorm',
    'LearnedPositionalEmbedding',
    'LightweightConv1dTBC',
    'LightweightConv',
    'LinearizedConvolution',
    'MultiheadAttention',
    'LunarMultiheadAttention',
    'LunarCausalAttention',
    'RealNumberEmbedding',
    'PositionalEmbedding',
    'SimpleRelativePositionalBias',
    'RotaryRelativePositionalBias',
    'SamePad',
    'ScalarBias',
    'ScaleNorm',
    'RMSNorm',
    'SequenceNorm',
    'SinusoidalPositionalEmbedding',
    'TransformerSentenceEncoderLayer',
    'TransformerSentenceEncoder',
    'TransformerDecoderLayer',
    'TransformerEncoderLayer',
    'LunaEncoderLayer',
    'LunaDecoderLayer',
    'LunaSentenceEncoder',
    'LunaSentenceEncoderLayer',
    'GatedAttentionUnit',
    'GatedCrossAttention',
    'BaseMovingLayer',
    'MultiHeadEMA',
    'MultiHeadComplexEMA',
    'DiagonalLinearRNN',
    'MovingAverageGatedAttention',
    'MegaEncoderLayer',
    'MegaDecoderLayer',
    'FlashSentenceEncoderLayer',
    'MegaSentenceEncoderLayer',
    'TransposeLast',
    'VGGBlock',
    'unfold1d',
]
