import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoderModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    SinusoidalPositionalEmbedding
)
from fairseq.models.luna_lra.luna_lra_encoder import LunaLRAEncoder
from fairseq.models.luna_lra.transformer_lra_encoder import TransformerLRAEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params

logger = logging.getLogger(__name__)


@register_model('lra')
class LRAModel(FairseqEncoderModel):
    """
    Class for training a transformer for LRA tasks.
    """
    def __init__(self, args, encoder, task):
        super().__init__(encoder)
        self.encoder = encoder
        self.args = args
        self.use_p = args.use_p
        self._max_positions = args.max_positions
        self.padding_idx = task.dictionary.pad()
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None
        self.classifier = nn.ModuleList([])
        self.classifier.append(nn.Linear(args.classifier_in_dim, args.classifier_out_dim))
        self.classifier.extend([
            nn.Linear(args.classifier_out_dim, args.classifier_out_dim)
            for _ in range(args.classifier_layers - 1)
        ])
        # self.classifier = nn.Linear(args.classifier_in_dim, args.classifier_out_dim)
        self.classifier_activation = utils.get_activation_fn(args.classifier_activation_fn)
        self.sentence_projection_layer = nn.Linear(
            args.classifier_out_dim,
            self.sentence_out_dim,
            bias=False
        )
        self.sen_rep_type = getattr(args, "sen_rep_type", "first")
        self.layer_type = args.layer_type

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, 'apply_bert_init', False):
            self.apply(init_bert_params)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float,
                            metavar='D', help='dropout probability for'
                            ' attention weights')
        parser.add_argument('--act-dropout', type=float,
                            metavar='D', help='dropout probability after'
                            ' activation in FFN')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--share-encoder-input-output-embed',
                            action='store_true', help='share encoder input'
                            ' and output embeddings')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--no-token-positional-embeddings',
                            action='store_true',
                            help='if set, disables positional embeddings'
                            ' (outside self attention)')
        parser.add_argument('--num-segment', type=int, metavar='N',
                            help='num segment in the input')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N',
                            help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set,'
                            ' calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')
        
        parser.add_argument('--use-p', default=False, action='store_true',
                            help='use p for prediction')

        # misc params
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--classifier-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='Which activation function to use for classifier layer.')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        parser.add_argument(
            '--layer-type',
            choices=['transformer', 'luna']
        )
        parser.add_argument(
            '--sen-rep-type',
            choices=['first', 'mp']
        )
        parser.add_argument(
            '--encoder-projection-length', type=int, metavar='N',
            help='projected length of encoder as key'
        )
        parser.add_argument(
            '--encoder-projected-attention-heads', type=int, metavar='N',
            help='num encoder projected attention heads'
        )
        parser.add_argument(
            '--decoder-projected-attention-heads', type=int, metavar='N',
            help='num decoder projected attention heads'
        )

    def forward(self, sample):
        src_tokens = sample['net_input']['src_tokens']
        if self.use_p:
            assert self.layer_type == 'luna'
            src_tokens = src_tokens[:, 1:]
        sentence_rep = self.encoder(src_tokens)
        if not self.use_p:
            if self.layer_type == 'transformer':
                sentence_rep = sentence_rep[1]
            elif self.layer_type == 'luna':
                sentence_rep = sentence_rep[1][0]
        else:
            sentence_rep = sentence_rep[1][1].mean(dim=0)
        if 'net_input1' in sample:
            src1_tokens = sample['net_input1']
            sentence1_rep = self.encoder(src1_tokens)
            if not self.use_p:
                if self.layer_type == 'transformer':
                    sentence1_rep = sentence1_rep[1]
                elif self.layer_type == 'luna':
                    sentence1_rep = sentence1_rep[1][0]
            else:
                sentence1_rep = sentence1_rep[1][1].mean(dim=0)
            concat_rep = []
            concat_rep.append(sentence1_rep)
            concat_rep.append(sentence_rep)
            sentence_rep = torch.cat(concat_rep, dim=-1)
        for layer in self.classifier:
            sentence_rep = self.classifier_activation(layer(sentence_rep))
        if self.sentence_projection_layer:
            sentence_logits = self.sentence_projection_layer(sentence_rep)
        return {
            'encoder_out': sentence_logits
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self._max_positions
    
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.max_positions
        if not hasattr(args, 'decoder_embed_dim'):
            args.decoder_embed_dim = args.encoder_embed_dim
        embed_tokens = cls.build_embedding(args, task.dictionary, args.encoder_embed_dim)
        logger.info(args)
        encoder = LRAEncoder(args, task)
        return cls(args, encoder, task)


class LRAEncoder(FairseqEncoder):
    """LRA encoder."""

    def __init__(self, args, task):
        super().__init__(task.dictionary)
        self.args = args
        if args.layer_type == 'transformer':
            self.encoder = TransformerLRAEncoder(
                tie_layer_weights=getattr(args, 'tie_layer_weights', False),
                padding_idx=task.dictionary.pad_index,
                vocab_size=len(task.dictionary),
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                max_seq_len=args.max_positions,
                num_segments=0,
                use_position_embeddings=True,
                offset_positions_by_padding=True,
                encoder_normalize_before=True,
                apply_bert_init=getattr(args, "apply_bert_init", False),
                activation_fn=args.activation_fn,
                learned_pos_embedding=True,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )
        else:
            self.encoder = LunaLRAEncoder(
                tie_layer_weights=getattr(args, 'tie_layer_weights', False),
                projection_length=args.encoder_projection_length,
                padding_idx=task.dictionary.pad_index,
                vocab_size=len(task.dictionary),
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                num_projected_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                max_seq_len=args.max_positions,
                use_position_embeddings=True,
                offset_positions_by_padding=True,
                layernorm_embedding=True,
                apply_bert_init=getattr(args, "apply_bert_init", False),
                tie_kv=getattr(args, 'tie_kv', False),
                activation_fn=args.activation_fn,
                learned_pos_embedding=True,
                embed_scale=None,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )
    
    def forward(self, src_tokens):

        return self.encoder(src_tokens)

@register_model_architecture('lra', 'lra')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.act_dropout = getattr(args, 'act_dropout', 0.0)

    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.num_segment = getattr(args, 'num_segment', 0)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 2048)

    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.sent_loss = getattr(args, 'sent_loss', True)

    args.apply_bert_init = getattr(args, 'apply_bert_init', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_activation_fn = getattr(args, 'classifier_activation_fn', 'relu')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.layer_type = getattr(args, 'layer_type', 'transformer')
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.classifier_in_dim = getattr(args, "classifier_in_dim", args.encoder_embed_dim)


@register_model_architecture('lra', 'transformer_lra_listop')
def transformer_lra_listop(args):
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 2002)
    args.tie_layer_weights = getattr(args, 'tie_layer_weights', True)
    base_architecture(args)

@register_model_architecture('lra', 'luna_lra_listop')
def luna_lra_listop(args):
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 2001)
    args.tie_layer_weights = getattr(args, 'tie_layer_weights', True)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    base_architecture(args)

@register_model_architecture('lra', 'transformer_lra_imdb')
def transformer_lra_imdb_architecture(args):
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 1024)
    base_architecture(args)

@register_model_architecture('lra', 'transformer_lra_imdb_eff')
def transformer_lra_imdb_eff_architecture(args):
    args.max_positions = getattr(args, 'max_positions', 1000)
    transformer_lra_imdb_architecture(args)

@register_model_architecture('lra', 'luna_lra_imdb')
def luna_lra_imdb_architecture(args):
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_imdb_architecture(args)

@register_model_architecture('lra', 'luna_lra_imdb_eff')
def luna_lra_imdb_architecture(args):
    args.max_positions = getattr(args, 'max_positions', 2000)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_imdb_architecture(args)

@register_model_architecture('lra', 'transformer_lra_aan')
def transformer_lra_aan_architecture(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 512)
    args.classifier_in_dim = getattr(args, 'classifier_in_dim', args.encoder_embed_dim * 2)
    base_architecture(args)

@register_model_architecture('lra', 'luna_lra_aan')
def luna_lra_aan_architecture(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_aan_architecture(args)

@register_model_architecture('lra', 'transformer_lra_cifar10')
def transformer_lra_cifar10(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 512)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 1024)
    base_architecture(args)

@register_model_architecture('lra', 'luna_lra_cifar10')
def luna_lra_cifar10(args):
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_cifar10(args)

@register_model_architecture('lra', 'transformer_lra_pf32')
def transformer_lra_pf32(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.max_positions = getattr(args, 'max_positions', 1026)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)

@register_model_architecture('lra', 'luna_lra_pf32')
def luna_lra_pf32(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_pf32(args)

@register_model_architecture('lra', 'luna_lra_pf128')
def luna_lra_pf32(args):
    args.max_positions = getattr(args, 'max_positions', 128*128+2)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_pf32(args)