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
    FairseqDropout,
)
from fairseq.models.lra.transformer_lra_encoder import TransformerLRAEncoder
from fairseq.models.lra.luna_lra_encoder import LunaLRAEncoder
from fairseq.models.lra.lstm_lra_encoder import LSTMLRAEncoder
from fairseq.models.lra.flash_lra_encoder import FlashLRAEncoder
from fairseq.models.lra.mega_lra_encoder import MegaLRAEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model('lra')
class SCRawModel(FairseqEncoderModel):
    """
    Class for training a transformer for LRA tasks.
    """
    def __init__(self, args, encoder, task):
        super().__init__(encoder)
        self.encoder = encoder
        self.args = args
        self.use_p = args.use_p
        self._max_positions = args.max_positions
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.classifier = nn.ModuleList([])
        self.classifier.append(nn.Sequential(Linear(args.classifier_in_dim, args.classifier_out_dim),
                                             self.dropout_module))
        self.classifier.extend([
            nn.Sequential(Linear(args.classifier_out_dim, args.classifier_out_dim), self.dropout_module)
            for _ in range(args.classifier_layers - 1)
        ])
        # self.classifier = nn.Linear(args.classifier_in_dim, args.classifier_out_dim)
        self.classifier_activation = utils.get_activation_fn(args.classifier_activation_fn)
        self.sentence_projection_layer = Linear(
            args.classifier_out_dim,
            self.sentence_out_dim,
            bias=False
        )
        self.sen_rep_type = getattr(args, "sen_rep_type", "cls")
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
                            metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--act-dropout', type=float,
                            metavar='D', help='dropout probability after activation in FFN')
        parser.add_argument('--feature-dropout', action='store_true',
                            help='apply feature dropout')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--encoder-hidden-dim', type=int, metavar='N',
                            help='encoder hidden dimension for Mega')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--z-dim', type=int, metavar='N',
                            help='encoder z dimension for FLASH')
        parser.add_argument('--n-dim', type=int, metavar='N',
                            help='encoder n dimension for Mega')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--share-encoder-input-output-embed',
                            action='store_true', help='share encoder input and output embeddings')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--no-token-positional-embeddings',
                            action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')

        parser.add_argument('--input-type', choices=['text', 'image'])
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N',
                            help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set, calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')

        parser.add_argument('--use-p', default=False, action='store_true',
                            help='use p for prediction')

        # misc params
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--attention-activation-fn', choices=['softmax', 'relu2', 'laplace'],
                            help='activation function for attention mechanism')
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

        parser.add_argument('--layer-type', choices=['transformer', 'luna', 'lstm', 'flash', 'mega'])
        parser.add_argument('--norm-type', choices=['layernorm', 'scalenorm', 'rmsnorm', 'batchnorm', 'syncbatchnorm'])
        parser.add_argument('--normalize-embedding', action='store_true', help='normalize embedding for Mega.')
        parser.add_argument('--sen-rep-type', choices=['cls', 'mp'])

        parser.add_argument('--chunk-size', type=int, metavar='N',
                            help='chunk size of Mega.')
        parser.add_argument('--truncation-length', type=int, metavar='N',
                            help='truncation length of moving average layer.')
        parser.add_argument('--encoder-projection-length', type=int, metavar='N',
                            help='projected length of encoder as key')
        parser.add_argument('--encoder-projected-attention-heads', type=int, metavar='N',
                            help='num encoder projected attention heads')
        parser.add_argument('--decoder-projected-attention-heads', type=int, metavar='N',
                            help='num decoder projected attention heads')

    def forward(self, sample):
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        sentence_rep = self.encoder(src_tokens, src_lengths)
        if not self.use_p:
            if self.layer_type in ['transformer', 'lstm', 'flash', 'mega']:
                sentence_rep = sentence_rep[1]
            elif self.layer_type == 'luna':
                sentence_rep = sentence_rep[1][0]
        else:
            sentence_rep = sentence_rep[1][1].mean(dim=0)
        if 'net_input1' in sample:
            src1_tokens = sample['net_input1']['src_tokens']
            src1_lengths = sample['net_input1']['src_lengths']
            sentence1_rep = self.encoder(src1_tokens, src1_lengths)
            if not self.use_p:
                if self.layer_type in ['transformer', 'lstm', 'flash', 'mega']:
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
        encoder = LRAEncoder(args, task)
        return cls(args, encoder, task)


class LRAEncoder(FairseqEncoder):
    """LRA encoder."""

    def __init__(self, args, task):
        if args.input_type == 'text':
            dictionary = task.dictionary
            vocab_size = len(dictionary)
            padding_idx = dictionary.pad_index
            offset_positions_by_padding = True
            embedding_type = 'sparse'
        else:
            assert args.sen_rep_type == 'mp' or args.layer_type == 'lstm'
            dictionary = None
            vocab_size = None
            padding_idx = None
            offset_positions_by_padding = False
            embedding_type = 'linear'
        super().__init__(dictionary)
        self.args = args
        if args.layer_type == 'transformer':
            self.encoder = TransformerLRAEncoder(
                tie_layer_weights=getattr(args, 'tie_layer_weights', False),
                padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_encoder_layers=args.encoder_layers,
                embedding_type=embedding_type,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                max_seq_len=args.max_positions,
                use_position_embeddings=True,
                offset_positions_by_padding=offset_positions_by_padding,
                encoder_normalize_before=getattr(args, "encoder_normalize_before", False),
                apply_bert_init=getattr(args, "apply_bert_init", False),
                activation_fn=args.activation_fn,
                learned_pos_embedding=args.encoder_learned_pos,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )
        elif args.layer_type == 'lstm':
            self.encoder = LSTMLRAEncoder(
                padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_layers=args.encoder_layers,
                bidirectional=True,
                embedding_type=embedding_type,
                embedding_dim=args.encoder_embed_dim,
                hidden_dim=args.encoder_ffn_embed_dim,
                input_dropout=args.dropout,
                output_dropout=args.act_dropout,
                max_seq_len=args.max_positions,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )
        elif args.layer_type == 'flash':
            self.encoder = FlashLRAEncoder(
                padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_encoder_layers=args.encoder_layers,
                embedding_type=embedding_type,
                embedding_dim=args.encoder_embed_dim,
                hidden_dim=args.encoder_hidden_dim,
                z_dim=args.z_dim,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                hidden_dropout=args.act_dropout,
                norm_type=args.norm_type,
                max_seq_len=args.max_positions,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )
        elif args.layer_type == 'mega':
            self.encoder = MegaLRAEncoder(
                padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_encoder_layers=args.encoder_layers,
                embedding_type=embedding_type,
                embedding_dim=args.encoder_embed_dim,
                hidden_dim=args.encoder_hidden_dim,
                ffn_hidden_dim=args.encoder_ffn_embed_dim,
                z_dim=args.z_dim,
                n_dim=args.n_dim,
                activation=args.activation_fn,
                attention_activation=args.attention_activation_fn,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                hidden_dropout=args.act_dropout,
                norm_type=args.norm_type,
                normalize_before=args.encoder_normalize_before,
                normalize_embedding=args.normalize_embedding,
                feature_dropout=args.feature_dropout,
                chunk_size=getattr(args, 'chunk_size', -1),
                truncation=getattr(args, 'truncation_length', None),
                max_seq_len=args.max_positions,
                sen_rep_type=getattr(args, 'sen_rep_type', 'mp')
            )
        else:
            self.encoder = LunaLRAEncoder(
                tie_layer_weights=getattr(args, 'tie_layer_weights', False),
                projection_length=args.encoder_projection_length,
                padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_encoder_layers=args.encoder_layers,
                embedding_type=embedding_type,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                num_projected_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                max_seq_len=args.max_positions,
                use_position_embeddings=True,
                offset_positions_by_padding=offset_positions_by_padding,
                layernorm_embedding=getattr(args, "encoder_normalize_before", False),
                normalize_before=False,
                apply_bert_init=getattr(args, "apply_bert_init", False),
                tie_kv=getattr(args, 'tie_kv', False),
                activation_fn=args.activation_fn,
                learned_pos_embedding=args.encoder_learned_pos,
                embed_scale=None,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        return self.encoder(src_tokens, src_lengths, last_state_only=True)
