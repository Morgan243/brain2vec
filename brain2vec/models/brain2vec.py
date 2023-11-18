from typing import Optional, Tuple, List

import pandas as pd
from dataclasses import dataclass, field
from fairseq.modules import GradMultiply
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import math
from torch.nn import Module, Parameter
import attr
from tqdm.auto import tqdm

from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    TransposeLast,
)

from brain2vec import datasets
from brain2vec import models as base
bmp = base
from mmz.models import base_fine_tuners as base_ft
import mmz
#t = mmz.models.GaussianNoise()
t = mmz.models.GaussianNoise(.1)

from brain2vec.models import Trainer
from torchaudio.models.wav2vec2.components import _compute_mask_indices
import logging

logging.getLogger("torch").setLevel(logging.WARNING)


def Dropout1d(p, **kwargs):
    v = torch.__version__
    if v[0] == '1' and v[2:4] == '11':
        return torch.nn.Sequential(
            #mmz.models.Unsqueeze(1),
            torch.nn.Dropout2d(p, **kwargs),
            #mmz.models.Squeeze()
        )
    else:
        return torch.nn.Dropout1d(p, **kwargs)


def make_linear_block(in_size, out_size, activation=torch.nn.LeakyReLU,
                      batch_norm=True, batch_norm_affine=True, dropout=0.,
                      as_sequential_module=False
                      ):
    l = [torch.nn.Linear(in_features=in_size, out_features=out_size)]
    if activation is not None:
        l.append(activation())

    if batch_norm:
        l.append(torch.nn.BatchNorm1d(affine=batch_norm_affine, num_features=out_size))

    if dropout > 0:
        l.append(torch.nn.Dropout(p=dropout))

    return torch.nn.Sequential(*l) if as_sequential_module else l



class LearnedPositionalEmbedding(torch.nn.Module):
    def __init__(self, shape, mean=0., std=.01):
        super().__init__()
        embed_arr = torch.normal(mean, std, shape)
        self.pos_embedding_params = torch.nn.Parameter(embed_arr)

    def forward(self, x):
        #print(f"X shape: {x.shape}")
        #print(f"embed shpe: {self.pos_embedding_params.shape}")
        return x + self.pos_embedding_params



class MultiHead(nn.Module):
    def __init__(self, model_heads: List[torch.nn.Module],
                 head_dim=1,
                 output_agg: str = 'sum'):
        super().__init__()
        self.model_heads = model_heads
        self.head_dim = head_dim
        self.output_agg = output_agg

    def forward(self, x: Tensor):
        x: Tensor =  torch.cat([
            self.model_heads[ii].to(x.device)(x.select(self.head_dim, ii).unsqueeze(self.head_dim)).unsqueeze(-1)
            for ii in range(x.shape[self.head_dim])],
            # Use the last dim to stack together (see unsqueeze above)
            dim=-1
        )

        if self.output_agg is None or self.output_agg.strip().lower() == 'none':
            out = x
        elif self.output_agg.strip().lower() == 'sum':
            out = x.sum(dim=-1)
        else:
            raise ValueError(f"bad `output_agg` command: {self.output_agg}")

        return out

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        #x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, log_scale_factor: float = 10000., max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.log_scale_factor = log_scale_factor

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(self.log_scale_factor) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        return x
#in_to_context = self.temporal_position_enc(in_to_context.permute(1, 0, 2)).permute(1, 0, 2)


class Brain2Vec(torch.nn.Module):
    logit_temp = 0.1

    def __init__(self, input_shape, feature_model, context_model, projection_model,
                 feature_extractor_dropout=0.0, feature_grad_mult=1,
                 negatives_from_everywhere=True, n_negatives=100,
                 cross_sample_negatives=0, codebook_negatives=0,
                 mask_length=3, mask_prob=0.3,
                 n_encoder_heads=8, n_encoder_layers=12,
                 encoder_dim_feedforward=2048,
                 quant_num_vars=300, quant_num_groups=2,
                 quant_weight_proj_factor=2, quant_weight_proj_depth=1,
                 feature_extractor_layers='[(128, 7, 3)] + [(128, 3, 2)] * 2 + [(128, 3, 1)]',
                 feature_extractor_mode='default',
                 context_encoder_dropout=0.,
                 input_1d_dropout=0,
                 ras_pos_encoding=True, temporal_pos_encoding=True,
                 ras_regularizer=None,
                 ras_architecture='multihead',
                 ras_batch_norm=False,
                 ras_noise=0,
                 affine_std=False,
                 positional_encoding_method='combined',
                 losses_to_output=None,
                 squeeze_first=False):
        super().__init__()
        self.input_shape = input_shape
        self.feature_model = feature_model
        self.feature_grad_mult = feature_grad_mult
        self.negatives_from_everywhere = negatives_from_everywhere

        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.codebook_negatives = codebook_negatives

        self.mask_length, self.mask_prob = mask_length, mask_prob
        self.positional_encoding_method = positional_encoding_method
        self.ras_regularizer = ras_regularizer
        self.ras_batch_norm = ras_batch_norm
        self.ras_noise = ras_noise
        self.input_1d_dropout = input_1d_dropout

        # TODO: Don't assign test data as attribute? But it's so useful
        self.t_x = torch.rand((16, *self.input_shape))
        self.t_ras = torch.rand((16, 1, 3))

        # ####
        # Feature extractor: typically CNN that downsamples/aggregates input for the context and quantize stages
        # Create a default feature exractor model if not provided
        #from fairseq.models.wav2vec import ConvFeatureExtractionModel
        self.squeeze_first = squeeze_first

        if self.feature_model is None:
            conv_layers = (eval(feature_extractor_layers) if isinstance(feature_extractor_layers, str)
                           else feature_extractor_layers)
            self.feature_model = ConvFeatureExtractionModel(
                conv_layers=conv_layers,
                dropout=feature_extractor_dropout,
                mode=feature_extractor_mode,#'layer_norm',
                # mode="default",#cfg.extractor_mode,
                conv_bias=False,  # cfg.conv_bias,
                #squeeze_first=True
            )

            feature_model_layers = [
                bmp.StandardizeOnLastDim(affine=affine_std),
                self.feature_model
            ]


            if self.input_1d_dropout > 0:
                feature_model_layers = [Dropout1d(p=self.input_1d_dropout)] + feature_model_layers

            self.feature_model = torch.nn.Sequential(
                *feature_model_layers
            )

            #from fairseq
            # TODO: check on existing norm and GELU activation?
#            self.feature_model = torch.nn.Sequential(
#                # torch.nn.Conv1d((input_channels:=X_barr.shape[1]), input_channels, 10, stride=5),
#                #torch.nn.BatchNorm1d(1),
#                torch.nn.Conv1d((input_channels := 1), (h_channels := 256), 7, stride=5),
#                torch.nn.Dropout(p=dropout),
#                torch.nn.GELU(),
#
#                torch.nn.Conv1d(h_channels, h_channels, 5, stride=2),
#                #torch.nn.BatchNorm1d(h_channels),
#                #torch.nn.Dropout(p=dropout),
#                torch.nn.GELU(),
#
#                torch.nn.Conv1d(h_channels, (f_dim:=128), 3, stride=1),
#                #torch.nn.Dropout(p=dropout),
#                torch.nn.GELU(),
#            )
#            self.feature_model.apply(base.py.weights_init)


        # Run test data through to get sizes automatically
        with torch.no_grad():
            self.t_in = dict(signal_arr=self.t_x.squeeze() if self.squeeze_first else self.t_x,
                             sensor_ras_coord_arr=self.t_ras.squeeze() if self.squeeze_first else self.t_ras)
            self.t_feat_o = self.feature_model(self.t_x.squeeze() if self.squeeze_first else self.t_x)

        # Unused, but maybe useful for debugging and experiments
        _, self.C, self.T = self.t_feat_o.shape

        print("Feature extractor output shape: " + str(self.t_feat_o.shape))
        embed_dim = self.C

        self.feature_norm = torch.nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)
        bmp.weights_init(self.feature_norm)

        # for unseen regions
        self.mask_embedding = torch.nn.Parameter(
            torch.FloatTensor(embed_dim).uniform_()
        )
        # For start of sentence - is this needed?
        self.sos_embedding = torch.nn.Parameter(
            torch.FloatTensor(embed_dim).uniform_()
        )
        # For end of sentence - is this needed?
        self.eos_embedding = torch.nn.Parameter(
            torch.FloatTensor(embed_dim).uniform_()
        )

        self.ras_pos_encoding = ras_pos_encoding
        self.temporal_pos_encoding = temporal_pos_encoding

        def _make_ras_encoder_layers(output_size, name: str = 'simple', hidden_size=32, batch_norm=False):
            #_hs = hidden_size
            if name == 'simple':
                l = (
                    [mmz.models.Squeeze()]
                    +
                    make_linear_block(3, hidden_size, batch_norm=batch_norm)
                    +
                    make_linear_block(hidden_size, hidden_size, batch_norm=batch_norm)
                    +
                    make_linear_block(hidden_size, output_size, batch_norm=False)
                )
            elif name == 'multihead':
                #_hs = 32
                l = [
                    mmz.models.Squeeze(),
                    MultiHead(
                    [
                        torch.nn.Sequential(*(
                                make_linear_block(1, hidden_size, batch_norm=batch_norm)
                                +
                                make_linear_block(hidden_size, hidden_size, batch_norm=batch_norm)
                                +
                                make_linear_block(hidden_size, output_size, batch_norm=False)
                        )),
                        torch.nn.Sequential(*(
                                make_linear_block(1, hidden_size, batch_norm=batch_norm)
                                +
                                make_linear_block(hidden_size, hidden_size, batch_norm=batch_norm)
                                +
                                make_linear_block(hidden_size, output_size, batch_norm=False)
                        )),
                        torch.nn.Sequential(*(
                                make_linear_block(1, hidden_size, batch_norm=batch_norm)
                                +
                                make_linear_block(hidden_size, hidden_size, batch_norm=batch_norm)
                                +
                                make_linear_block(hidden_size, output_size, batch_norm=False)
                        ))
                    ],
                    output_agg='sum'
                )]
            else:
                raise ValueError(f"Don't understand {name}")

            if self.ras_noise > 0:
                l = [mmz.models.GaussianNoise(self.ras_noise)] + l
                
            if self.input_1d_dropout > 0:
                l = [Dropout1d(p=self.input_1d_dropout)] + l

            return l


        #self.combined_enc = None
        #if self.positional_encoding_method == 'combined':
        self.temporal_position_enc = None
        if 'combined' in self.positional_encoding_method:
            l = _make_ras_encoder_layers(self.C * self.T, batch_norm=self.ras_batch_norm, name=ras_architecture)
            self.ras_positional_enc = torch.nn.Sequential(
                *l
            )
            self.ras_positional_enc.apply(bmp.weights_init)
            self.positional_enc = None
            if self.positional_encoding_method == 'combined+':
                print("------------------------USING POSITIONAL ENCODING WITH LEARNED ONE!!!")
                self.temporal_position_enc = PositionalEncoding(d_model=self.C, max_len=self.T)

        elif self.ras_pos_encoding:
            l = _make_ras_encoder_layers(self.C, batch_norm=self.ras_batch_norm, name=ras_architecture)
            self.ras_positional_enc = torch.nn.Sequential(
                *l
            )
            self.ras_positional_enc.apply(bmp.weights_init)

            if 'pos_embedding' in self.positional_encoding_method:
                print("USING NEW POS EMBEDDING")
                self.temporal_position_enc = LearnedPositionalEmbedding((1, self.T, self.C))
            else:
                self.temporal_position_enc = PositionalEncoding(d_model=self.C, max_len=self.T)

            self.positional_enc = None
        else:
            self.positional_enc = PositionalEncoding(d_model=self.C, max_len=self.T)
            self.ras_positional_enc = None


        #if self.temporal_pos_encoding and ('combined' not in self.positional_encoding_method):
        #    self.temporal_position_enc = PositionalEncoding(d_model=self.T)

        # Init Context model (transformer encoder)
        self.n_heads = n_encoder_heads
        self.num_encoder_layers = n_encoder_layers
        self.context_model = context_model
        self.context_encoder_dropout = context_encoder_dropout
        self.encoder_dim_feedforward = encoder_dim_feedforward
        if self.context_model is None:
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=self.n_heads, batch_first=True,
                                                             activation="gelu", dropout=self.context_encoder_dropout,
                                                             dim_feedforward=encoder_dim_feedforward)
            transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers,
                                                              norm=nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6))
            transformer_encoder.apply(bmp.weights_init)
            self.context_model = transformer_encoder

            #from fairseq.models.wav2vec import TransformerEncoder
            #TransformerEncoder()

        self.quant_weight_proj_depth, self.quant_weight_proj_factor = quant_weight_proj_depth, quant_weight_proj_factor
        self.quant_num_vars = quant_num_vars
        self.quant_num_groups = quant_num_groups

        # Use existing Gumbel Quant
        import fairseq
        self.quantizer = fairseq.modules.GumbelVectorQuantizer(
            # TODO: parameterize more of these?
            dim=embed_dim, num_vars=quant_num_vars, temp=(2, 0.5, 0.999995),#temp=(1, 0.1, 0.9),
            groups=quant_num_groups, combine_groups=False, vq_dim=embed_dim,
            time_first=True,
            # defaults in fairseq config
            weight_proj_factor=self.quant_weight_proj_factor, weight_proj_depth=self.quant_weight_proj_depth
        )

        # Currently unused, but in future may need one or more linear projections from one space to another
        self.projection_q_model = projection_model

        if self.projection_q_model is None:
            self.projection_q_model = torch.nn.Linear(embed_dim, embed_dim)
            bmp.weights_init(self.projection_q_model)

        self.projection_out_model = None
        if self.projection_out_model is None:
            self.projection_out_model = torch.nn.Linear(embed_dim, embed_dim)
            bmp.weights_init(self.projection_out_model)

        # Do this last so the method has access to all attributes
        # This method also assigns the attribute
        self.set_losses_to_output(losses_to_output)

    def set_losses_to_output(self, losses_to_output: Optional[str]):
        self.losses_to_output = losses_to_output
        if self.losses_to_output is None:
            self.losses_to_output = 'fqp,cqp,cl,cp'

        ## - Setup output configuration for shadow modeling -
        self.bce_loss_output_shape = 0

        self.output_feature_q_proba = False
        self.output_context_q_proba = False
        self.output_contrastive_loss = False
        self.output_contexts_predictions = False

        loss_list = self.losses_to_output.split(',')

        ## Measure how the feature encoders quantization is distributed
        if 'fqp' in loss_list:
            self.bce_loss_output_shape += self.T * self.quant_num_groups * self.quant_num_vars
            self.output_feature_q_proba = True
        ## Measure how the context's projection output quantization is distributed
        ## - When run with features_only=False, context output is projected for direct comparison
        ##   with the code quantization values.
        if 'cqp' in loss_list:
            self.bce_loss_output_shape += self.T * self.quant_num_groups * self.quant_num_vars
            self.output_context_q_proba = True
        ## Measure how well the encoder did, given the extracted features
        if 'cl' in loss_list:
            self.bce_loss_output_shape += self.T * self.mask_length
            self.output_contrastive_loss = True
        ## Provide the context encoders prediction for each masked step
        if 'cp' in loss_list:
            self.bce_loss_output_shape += self.T * self.C
            self.output_contexts_predictions = True

        return self

    # Adapted From fairseq wave2vec2 (removed xla check)
    def compute_preds(self, x, y, negatives, ):
        # Negatives: n_negatives x B x T x C

        # determine where negatives are that are also positives
        # This is at the feature level, so identifying a specific negative,
        # within a sample in the batch, at specific time tgat is actually a positive
        neg_is_pos = (y == negatives).all(-1)
        # Add the first dimension to match with negatives addition first dim of n_negatives size
        y = y.unsqueeze(0)
        # Combine these on the first dim - actual values now appear as a +1 to the negatives
        # but always at the first (0th) dimension
        targets = torch.cat([y, negatives], dim=0)

        # Measure context models output similarity the K+1 targets
        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        logits = logits / self.logit_temp
        logits = logits.type_as(x)

        if neg_is_pos.any():
            # slice at 1, beyond 0th element (the true positive) and set these to negative inf
            # A good model would identify these, but the BCE penalty would penalize those, so
            # just wipe them out with a very low logit
            #logits[1:][neg_is_pos] = float("-inf")
            logits[1:][neg_is_pos] = float(-100)
            # original from fairseq
            #logits[1:] = index_put(logits[1:], neg_is_pos, float("-inf"))

        return logits

    # Adapted From fairseq wave2vec2 (remove xla check, add params, buffered arange inline def)
    @staticmethod
    def sample_negatives(y, num, n_negatives, cross_sample_negatives=0, padding_count=None):
        def buffered_arange(max):
            if not hasattr(buffered_arange, "buf"):
                buffered_arange.buf = torch.LongTensor()
            if max > buffered_arange.buf.numel():
                buffered_arange.buf.resize_(max)
                torch.arange(max, out=buffered_arange.buf)
            return buffered_arange.buf[:max]

        if n_negatives == 0 and cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        # Flatt to all feature samples (flatten batch and time)
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if n_negatives > 0:
                # Possible time steps to use, {0, 1, 2, 3, ... num.}  for each negative
                tszs = (
                    # Long Tensor range 0 to NUm - 1
                    buffered_arange(num)
                    # Add a singleton dim at the end
                        .unsqueeze(-1)
                    # Duplicate data across singleton dimension n_negatives times
                    # now (tsz, n_negatives), incrementing over tsz
                        .expand(-1, n_negatives)
                    # like reshape(-1), len == tsc * n_negatives
                        .flatten()
                )

                #
                neg_idxs = torch.randint(
                    # Note that high here is exlusive - it's "one above the highest number to be drawn"
                    # So this is one less than the max index, allowing the shift by one to happen later
                    low=0, high=high - 1,
                    # Each sample in batch needs a set of negatives...for each masked timestamp?
                    size=(bsz, n_negatives * num)
                )
                # Add one in order to index over the true positive at the starting position, and shift all other
                # times from that point. If only shifting the matching, then the point after will be selected twice
                neg_idxs[neg_idxs >= tszs] += 1

            if cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, cross_sample_negatives)
                        .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if n_negatives > 0:
            # Offset into each batch
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if cross_sample_negatives > 0 and n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, n_negatives + cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    @staticmethod
    def cobebook_likelihoods(quantizer, x):
        result = {"num_vars": quantizer.num_vars * quantizer.groups}

        if not quantizer.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = quantizer.weight_proj(x)
        x = x.view(bsz * tsz * quantizer.groups, -1)

        # ---
        result["avg_probs_elements"] = torch.softmax(
            x.view(bsz * tsz, quantizer.groups, -1).float(), dim=-1
        ).reshape(bsz , tsz, quantizer.groups, -1)#.mean(dim=0)

        return result

    def forward(self, X, features_only=False, mask=True, output_loss=False):
        if output_loss is not None and output_loss:
#            # Measure how the feature encoders quantization is distributed
#            output_feature_q_proba_loss: bool = True
#
#            # Measure how the context's projection output quantization is distributed
#            # - When run with features_only=False, context output is projected for direct comparison
#            #   with the code quantization values.
#            output_context_q_proba_loss: bool = True
#
#            # Measure how well the encoder did, given the extracted features
#            output_contrastive_loss: bool = True
#
#            # Provide the context encoders prediction for each masked step
#            output_contexts_predictions: bool = True

            bs = X['signal_arr'].shape[0]

            # Mask everything
            mask_indices = torch.ones((bs, self.T), dtype=torch.bool)

            # Then unmask all but the first vector
            mask_indices[:, 1:] = False

            bce_losses_l = list()
            preds_l = list()
            codebook_probas = list()

            # For each timestep
            for i in range(self.T):
                out_d = self._forward(X, features_only=False, mask=True, mask_indices=mask_indices.roll(i, 1))

                if self.output_contrastive_loss:
                    bce_losses_l.append(
                        # Each BCE loss is then (bs, n_masks)
                        Brain2VecTrainer._loss(out_d, cross_entropy_reduction='none')['bce_loss'].reshape(bs, -1)
                    )

                if self.output_context_q_proba:
                    codebook_probas.append(
                        # Each masked step in each group gets probas across all words (bs, tz, gz, wz)
                        self.cobebook_likelihoods(self.quantizer, out_d['x'])['avg_probs_elements'].reshape(bs, -1)
                    )
                #preds_l.append(out_d['preds'].permute(1, 0, 2).squeeze())
                #input_q, input_q_ix = self.quantizer.forward_idx(out_d['unmasked_features'])
                #out_d.update(loss_d)
                #out_d['bce_loss_orig'] = out_d['bce_loss'].reshape(bs, -1)

            output_arrs = dict()
            if self.output_contrastive_loss:
                output_arrs['contrastive_loss'] = -torch.log(torch.concat(bce_losses_l, dim=1) + 0.000001)
            if self.output_context_q_proba:
                output_arrs['context_q_probas'] = torch.concat(codebook_probas, dim=1)
            if self.output_feature_q_proba:
                # take the last forwards unmasked feature extractor output and see how it quantized
                output_arrs['feature_q_probas'] = self.cobebook_likelihoods(self.quantizer,
                                                                            out_d['unmasked_features'])['avg_probs_elements'].reshape(bs, -1)

            output_arrs = {k: arr.detach() for k, arr in output_arrs.items()}
            out_d['bce_loss'] = torch.concat(list(output_arrs.values()), dim=1)

            #bce_for_attacker = torch.concat(bce_losses_l, dim=1)
            #probas_for_attacker = torch.concat(codebook_probas, dim=1)
            ## take the last forwards unmasked feature extractor output and see how it quantized
            #featu_q_proba_for_attacker = self.cobebook_likelihoods(self.quantizer,
            #                                                       out_d['unmasked_features'])['avg_probs_elements'].reshape(bs, -1)
            #preds_for_attacker = torch.concat(preds_l, dim=1)

            #input_q, input_q_ix = self.quantizer.forward_idx(out_d['unmasked_features'])
            
            # Shift the second group up by num vars so they are unique indicators
            #input_q_ix[:, :, 1] = input_q_ix[:, :, 1] + self.quantizer.num_vars
            #onehot_q_ix = torch.nn.functional.one_hot(input_q_ix.reshape(bs,-1), 
            #                                          self.quantizer.groups * self.quantizer.num_vars)

            #device = X['signal_arr'].device
            #out_d['bce_loss'] = torch.concat([
            #    -torch.log((
            #        bce_for_attacker + 0.000001
            #               ).detach()),
            #    probas_for_attacker.detach(),
            #    featu_q_proba_for_attacker.detach(),
            #    #preds_for_attacker.detach(),
            #    #self.mask_embedding.expand(bs, self.mask_embedding.shape[0])
            #    ], dim=1)
            #out_d['bce_loss_plus'] = torch.concat([
            #out_d['bce_loss'] = torch.concat([
            #    (out_d['bce_loss_orig'].detach().to(device) / 100) - 2, 
            #    #out_d['mask_indices'].detach().reshape(bs, -1).float().to(device),
            #    #onehot_q_ix.reshape(bs, -1).detach().to(device)
            #    ], dim=1)
            return out_d
        else:
            return self._forward(X, features_only=features_only, mask=mask)
    
    def _forward(self, X, features_only=False, mask=True, mask_indices=None):
        input_d = dict()
        if isinstance(X, dict):
            input_d = X
            X = input_d['signal_arr']

        # TODO: Wave2vec says normalize the raw waveform to zero mean and unit variance - maje sure this happening?
        # Extract features from signal
        # Wave to vec in fair seq either does forward without gradient or with grad multiply
        # - prob grad mult for pre-train
        X = X.squeeze() if self.squeeze_first else X
        X_f = self.feature_model(X)
        if self.feature_grad_mult != 1.0:
            X_f = GradMultiply.apply(X_f, self.feature_grad_mult)

        # Note expected dims: (B)atch, (C)hannel, (T)ime
        B, C, T = X_f.shape

        # Swap C and T  for use with mask and later transformer sequence modeling
        unmasked_features = X_f.transpose(1, 2)
        unmasked_features = self.feature_norm(unmasked_features)

        if mask:
            # Create the mask - what will be hidden from the context model
            if mask_indices is None:
                # fairseq doesn't seem to mask the last timestep - maybe to leave it for a cls token?
                # But we don't have many timesteps and not using cls or other token strategy
                # So tell it we have an extra time dim and then just slice out the always unmasked extra timestep
                mask_indices = _compute_mask_indices((B, T+1), padding_mask=None, mask_prob=self.mask_prob,
                                                     mask_length=self.mask_length, min_masks=1)[:, :-1]
                #mask_indices = _compute_mask_indices((B, T), padding_mask=None, mask_prob=self.mask_prob,
                #                                     mask_length=self.mask_length, min_masks=1)

            # Create inverse of mask to select unmasked values
            #umask_ixes = ~mask_indices

            mask_indices = mask_indices.type(torch.BoolTensor)
            # Select the masked elements as our y, and reshape back
            y = unmasked_features[mask_indices].view(unmasked_features.shape[0], -1, unmasked_features.shape[-1]).contiguous()

            # Go ahead and make a copy of the original data (Same move made in Wave2vec2.py @ line 1021)
            masked_features = torch.clone(unmasked_features)

            # overwrite masked indices with the learnable mask embedding
            masked_features[mask_indices] = self.mask_embedding
            in_to_context = masked_features
        else:
            mask_indices = None
            in_to_context = unmasked_features
            y = unmasked_features

        pos_enc_arr = None
        if 'combined' in self.positional_encoding_method:
            if 'sensor_ras_coord_arr' not in input_d:
                raise KeyError("'sensor_ras_coord_arr' not in batch data"
                               " - to use as pos. emb., use datasets extra_output_keys='sensor_ras_coord_arr'")
            ras_arr = input_d['sensor_ras_coord_arr']
            # Combined will output enough values to populate along channel and time axis - so just reshap to expected
            pos_enc_arr = self.ras_positional_enc(ras_arr).reshape(in_to_context.shape)
            in_to_context = (in_to_context + pos_enc_arr)
            #if self.positional_enc is not None:
            #    in_to_context = self.positional_enc(in_to_context)
        elif self.ras_pos_encoding:
            if 'sensor_ras_coord_arr' not in input_d:
                raise KeyError("'sensor_ras_coord_arr' not in batch data"
                               " - to use as pos. emb., use datasets extra_output_keys='sensor_ras_coord_arr'")
            ras_arr = input_d['sensor_ras_coord_arr']
            ras_sqz_arr = ras_arr.squeeze()
            _, cT, cC = in_to_context.shape

            # RAS encode is constant across time (i.e. not temporally encoding) - outputs a dim with cardinality
            # equal to channel dim i.e. the res output after unsqueeze() that is added is shaped:
            #   (Batch, 1, channels)
            # So will be broadcast along the time dimension
            #in_to_context = (in_to_context + self.ras_positional_enc(ras_arr).unsqueeze(1))
            pos_enc_arr = self.ras_positional_enc(ras_sqz_arr).reshape(B, *([1] * (in_to_context.dim() - 2)), cC)
            in_to_context = (in_to_context + pos_enc_arr)
            #ras_enc_arr.reshape(128, *([1] * (t_ras_arr.dim() - 2)), 128)
        else:
            # Otherwise, use 'normal'/absolute encoding through a module that will do the addition in it's forward pass
            in_to_context = self.positional_enc(in_to_context)

        # Independent temporal encoding if it wasn't already included in the above encoder
        #if self.temporal_pos_encoding and ('combined' not in self.positional_encoding_method):
        #apply_temporal = ((self.temporal_pos_encoding and ('combined' not in self.positional_encoding_method))
        #                  or
        #                  (self.positional_encoding_method == 'combined+'))
        apply_temporal = self.temporal_position_enc is not None
        if apply_temporal:
            # Swap time dimension back to the last dim (dim 2) for temporal pos encoding, then swap result back
            # This module adds the encoding in its forward pass
            #in_to_context = self.temporal_position_enc(in_to_context.transpose(1, 2)).transpose(1, 2)
            #in_to_context = self.temporal_position_enc(in_to_context.permute(1, 0, 2)).permute(1, 0, 2)
            in_to_context = self.temporal_position_enc(in_to_context)

        x = self.context_model(in_to_context)

        # #################################
        # Code block from fairseq wave2vec2
        if features_only:
            d = {
                "x": x,
                #"padding_mask": padding_mask,
                "features": unmasked_features,
                #"layer_results": layer_results,
            }

            #if pos_enc_arr is not None:
            #    d['pos_enc_arr'] = pos_enc_arr

            if mask:
                d['mask_indices'] = mask_indices
                d['mask_features'] = in_to_context
                d['y'] = y
            return d

        # penalize for large features
        # This is in the wave2vec2 fairseq codebase, but this is not an L2 norm...?
        #features_pen = X_f.float().pow(2).mean()
        features_pen = X_f.pow(2).sum().sqrt()
        #if pos_enc_arr is not None:
        #    pos_enc_pen = pos_enc_arr.pow(2).sum().sqrt()

        padding_count = 0

        if self.quantizer:
            if self.negatives_from_everywhere:
                # Don't know if this works
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.projection_q_model(y)

                negs, _ = self.sample_negatives(
                    y,
                    mask_indices[0].sum(),
                    padding_count=padding_count, cross_sample_negatives=self.cross_sample_negatives,
                    n_negatives=self.n_negatives
                )
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"].contiguous()
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.projection_q_model(y)

                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    n_negatives=self.n_negatives, cross_sample_negatives=self.cross_sample_negatives,
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                #raise NotImplementedError
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                # cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            # TODO: project q won't work
            raise NotImplementedError()
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    n_negatives=n_negatives, cross_sample_negatives=cross_sample_negatives,
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    n_negatives=n_negatives, cross_sample_negatives=cross_sample_negatives,
                    padding_count=padding_count,
                )

        # Get outputs from the context model for masked regions - it's estimatino of the quantized output
        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        #
        x = self.projection_out_model(x)
        preds = self.compute_preds(x, y, negs)

        return dict(x=x, y=y, negs=negs, preds=preds,
                    mask_indices=mask_indices,
                    unmasked_features=unmasked_features,
                    features_pen=features_pen,
                    #pos_enc_pen=pos_enc_pen,
                    num_vars=num_vars,
                    code_perplexity=code_ppl, prob_perplexity=prob_ppl,
                    temp=curr_temp)

    def _forward_idx(self, X):
        out_d = self._forward(X, features_only=True, mask=False)

class MultiChannelBrain2Vec(torch.nn.Module):
    def __init__(self, input_shape, c2v_m: Brain2Vec, hidden_encoder='linear',
                 dropout=0., batch_norm=False, linear_hidden_n=16, n_layers=2,
                 dropout_2d=0.,
                 outputs=1):
        super().__init__()
        self.input_shape = input_shape
        self.c2v_m = c2v_m
        self.outputs = outputs
        self.dropout_rate = dropout
        self.dropout_2d_rate = dropout_2d
        self.batch_norm = batch_norm

        self.mc_from_1d = base_ft.MultiChannelFromSingleChannel(self.input_shape, self.c2v_m)

        self.S = self.input_shape[0]
        self.T, self.C = self.c2v_m.T, self.c2v_m.C
        self.h_dim = self.S * self.C

        self.dropout_2d = None
        if self.dropout_2d_rate > 0:
            self.dropout_2d = torch.nn.Dropout2d(p=self.dropout_2d_rate)

        #B, T, S, C = output_arr.shape

        #output_arr_t = output_arr.reshape(B, T, -1)

        #hidden_encoder = 'linear' if hidden_encoder is None else hidden_encoder
        self.hidden_encoder_input = hidden_encoder

        self.classifier_head = torch.nn.Identity()

        def make_linear(outputs, regularize=True, activation=torch.nn.LeakyReLU):
            l = list()
            if regularize and self.dropout_rate > 0.:
                l.append(torch.nn.Dropout(self.dropout_rate))

            l.append(torch.nn.LazyLinear(outputs))
            #torch.nn.init.xavier_uniform_(l[-1].weight)

            if regularize and self.batch_norm:
                l.append(torch.nn.LazyBatchNorm1d(momentum=0.2, track_running_stats=True, affine=True))

            if activation is not None:
                l.append(activation())

            return torch.nn.Sequential(*l)

        if isinstance(hidden_encoder, torch.nn.Module):
            self.hidden_encoder = hidden_encoder
        elif hidden_encoder == 'linear':
            self.lin_dim = self.outputs
            self.hidden_encoder = torch.nn.Sequential(
                *[make_linear(linear_hidden_n) for i in range(n_layers - 1)],
                *make_linear(self.outputs, regularize=False, activation=None)
            )
            if self.outputs == 1:
                self.classifier_head = torch.nn.Sequential(torch.nn.Sigmoid())

            self.feat_arr_reshape = (-1, self.h_dim * self.T)
            #self.classifier_head = torch.nn.Sequential(torch.nn.Sigmoid() if self.outputs == 1
            #                                           else torch.nn.Identity())
        elif hidden_encoder == 'transformer':
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.h_dim, dropout=self.dropout_rate,
                                                             nhead=2, batch_first=True,
                                                             activation="gelu")

            self.hidden_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.feat_arr_reshape = (-1, self.T, self.h_dim)
            self.lin_dim = self.h_dim * self.T

            # h_size = 32
            self.classifier_head = torch.nn.Sequential(*[
                # base.py.Reshape((self.C * self.T,)),
                # torch.nn.Linear(self.lin_dim, h_size),
                # torch.nn.BatchNorm1d(h_size),
                # torch.nn.LeakyReLU(),
                # torch.nn.Linear(h_size, h_size),
                # torch.nn.LeakyReLU(),
                torch.nn.Linear(self.lin_dim, self.outputs),
                # torch.nn.Sigmoid()
            ])
        else:
            raise ValueError(f"Don't understand hidden_encoder = '{hidden_encoder}'")

    def forward(self, input_d: dict):
        feat_d = self.mc_from_1d(input_d)
        feat_arr = feat_d['output']
        if self.dropout_2d is not None:
            feat_arr = self.dropout_2d(feat_arr.permute(0, 2, 1, 3))

        B = feat_arr.shape[0]

        #trf_arr = feat_arr.reshape(B, self.T, self.h_dim)
        trf_arr = feat_arr.reshape(*self.feat_arr_reshape)
        trf_out_arr = self.hidden_encoder(trf_arr)
        lin_in_arr = trf_out_arr.reshape(B, self.lin_dim)

        return self.classifier_head(lin_in_arr)

#    @classmethod
#    def as_transform(cls):
#        kws = dict()
#        def multi_channel_brain2vec_trasnform():
#            kws['signal_arr'] = torch.from_numpy(np_ecog_arr).float()

@attr.s
class Brain2VecTrainer(Trainer):
    input_key = attr.ib('signal_arr')
    ppl_weight = attr.ib(100.)
    feature_pen_weight = attr.ib(1.)
    pos_enc_pen_weight = attr.ib(0.)
    squeeze_first = False
    model_output_logits_key = 'preds'

    def _score(self, epoch_i, dataloader, model_key='model'):
        model = self.model_map[model_key]
        model.eval()

        self.best_cv = getattr(self, 'best_cv', np.inf)

        preds_l, actuals_l, loss_l = list(), list(), list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):
                    _X = _x['signal_arr'].to(self.device)
                    _b_X = _X.select(1, sens_id := 10).unsqueeze(1)
                    out_d = model(_b_X, mask=False, features_only=True)
                    y = model.quantizer(out_d['features'])['x']
                    scores = torch.cosine_similarity(out_d['x'], y, dim=-1)
                    scores_l = 1 - scores.mean()
                    loss_l.append(scores_l.detach().cpu().item())
                    #preds = model(_x['ecog_arr'].to(self.device))
                    #actuals = _x['text_arr'].to(self.device)
                    #loss = self.criterion(preds, actuals)

                    #loss_l.append(loss.detach().cpu().item())

                    pbar.update(1)

                mean_loss = np.mean(loss_l)
                desc = "Mean Eval Loss: %.5f" % mean_loss
                if self.model_regularizer is not None:
                    reg_l = self.model_regularizer(model)
                    desc += (" (+ %.6f reg loss = %.6f)" % (reg_l, mean_loss + reg_l))
                else:
                    reg_l = 0.
                overall_loss = (mean_loss + reg_l)
                if overall_loss < self.best_cv:

                    self.best_model_state = base.copy_model_state(model)
                    self.best_model_epoch = epoch_i
                    self.best_cv = overall_loss
                    desc += "[[NEW BEST]]"

                pbar.set_description(desc)

        self.model_map['model'].train()
        return loss_l

    def eval_on(self, dataloader, cb=None):
        model = self.model_map['model']
        model.eval()
        eval_results_d_l = list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):

                    m_d = model(
                        # Move all the arrays in the input dictionary to the right device
                        # before passing to the model
                        {k: arr.to(self.device) for k, arr in _x.items()}
                    )

                    loss_d = self.loss(m_d, as_tensor=False)
                    score_d = self.score_training(m_d, as_tensor=False)

                    eval_d = dict(**score_d, **loss_d)
                    eval_results_d_l.append(eval_d)

                    pbar.update(1)

                eval_res_df = pd.DataFrame(eval_results_d_l)
                if cb is not None:
                    cb(eval_res_df, pbar)

            return eval_res_df

    def _eval(self, epoch_i, dataloader, model_key='model', primary_eval_key='bce_loss'):
        model = self.model_map[model_key]
        model.eval()
        self.best_cv = getattr(self, 'best_cv', np.inf)

        def cb(_eval_res_df, pbar):
            _eval_res_d = _eval_res_df.mean().to_dict()

            desc = ", ".join(f"{k}={np.round(r, 4)}" for k, r in _eval_res_d.items())
            mean_loss = _eval_res_d[primary_eval_key]

            if self.model_regularizer is not None:
                reg_l = self.model_regularizer(model)
                desc += (" (+ %.6f reg loss = %.6f)" % (reg_l, mean_loss + reg_l))
            else:
                reg_l = 0.
            overall_loss = (mean_loss + reg_l)
            if overall_loss < self.best_cv:

                self.best_model_state = base.copy_model_state(model)
                self.best_model_epoch = epoch_i
                self.best_cv = overall_loss
                desc += "[[NEW BEST]]"

            pbar.set_description(desc)

        eval_res_df = self.eval_on(dataloader, cb=cb)
        eval_res_d = eval_res_df.mean().to_dict()
        eval_res_d['primary_loss'] = eval_res_d[primary_eval_key]

        self.model_map[model_key].train()

        return eval_res_d

    @staticmethod
    def _loss(model_output_d, model_output_logits_key='preds', 
              cross_entropy_reduction='sum',
              ppl_weight=1, feature_pen_weight=1, pos_enc_pen_weight=0.,
              as_tensor=True):
        logits = model_output_d[model_output_logits_key]
        num_vars = model_output_d['num_vars']
        prob_ppl = model_output_d['prob_perplexity']

        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        target = torch.zeros_like(logits)

        loss = F.cross_entropy(
            logits, target[:, 0].long(), reduction=cross_entropy_reduction, #weights, reduction=reduction
        )
        #loss = F.binary_cross_entropy_with_logits(
        #    logits, target.float(), reduction='sum' #weights, reduction=reduction
        #)

        ppl_l = ((num_vars - prob_ppl) / num_vars) * ppl_weight
        fpen_l = model_output_d["features_pen"] * feature_pen_weight

        o = dict(bce_loss=loss, perplexity=ppl_l, feature_pen=fpen_l)
        
        if pos_enc_pen_weight > 0:
            o['pos_pen'] = model_output_d['pos_enc_pen'] * pos_enc_pen_weight

        if not as_tensor:
            o = {k: v.detach().cpu().item() for k, v in o.items()}
        return o

    def loss(self, model_output_d, as_tensor=True):
        return self._loss(model_output_d, model_output_logits_key=self.model_output_logits_key,
                          cross_entropy_reduction='sum', ppl_weight=self.ppl_weight,
                          pos_enc_pen_weight=self.pos_enc_pen_weight,
                          feature_pen_weight=self.feature_pen_weight,
                          as_tensor=as_tensor)
        #logits = model_output_d[self.model_output_logits_key]
        #num_vars = model_output_d['num_vars']
        #prob_ppl = model_output_d['prob_perplexity']

        #logits = logits.transpose(0, 2)
        #logits = logits.reshape(-1, logits.size(-1))
        #target = torch.zeros_like(logits)

        #loss = F.cross_entropy(
        #    logits, target[:, 0].long(), reduction='sum' #weights, reduction=reduction
        #)
        ##loss = F.binary_cross_entropy_with_logits(
        ##    logits, target.float(), reduction='sum' #weights, reduction=reduction
        ##)

        #ppl_l = ((num_vars - prob_ppl) / num_vars) * self.ppl_weight
        #fpen_l = model_output_d["features_pen"] * self.feature_pen_weight

        #o = dict(bce_loss=loss, perplexity=ppl_l, feature_pen=fpen_l)
        #if not as_tensor:
        #    o = {k: v.detach().cpu().item() for k, v in o.items()}
        #return o

    def score_training(self, model_output_d, as_tensor=False):
        """training score metrics that don't require gradient, won't have .backward() called,
        compliments loss for understanding performance"""
        logits = model_output_d[self.model_output_logits_key]

        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))

        with torch.no_grad():
            _max = logits.argmax(-1) == 0
            _min = logits.argmin(-1) == 0

            both = _max & _min
            corr = _max.long().sum().item() - both.long().sum().item()
            count = float(_max.numel())

        acc = (corr / count) if count != 0 else 0
        acc = torch.Tensor([acc]) if as_tensor else acc
        count = torch.Tensor([count]) if as_tensor else count
        return dict(accuracy=acc, n=count)

    def train_inner_step(self, epoch_i, data_batch):
        res_d = dict()

        model = self.model_map['model']
        optim = self.opt_map['model']
        model = model.train()

        model.zero_grad()
        optim.zero_grad()

        m_d = model({k: arr.squeeze().to(self.device) if self.squeeze_first
                     else arr.to(self.device)
                    for k, arr in data_batch.items()})

        loss_d = self.loss(m_d)

        total_loss = sum((loss_d.values()))
        total_loss.backward()

        optim.step()

        model = model.eval()

        score_d = self.score_training(m_d)

        tl = total_loss.detach().cpu().item()

        return dict(
                    # Unpack score floats or tensors
                    **{s_k: s.detach().cpu().item() if isinstance(s, torch.Tensor) else s
                       for s_k, s in score_d.items()},

                    # Total loss right after scores
                    total_loss=tl,

                    # Unpack losses - should be all Tensors?
                    **{l_k: l.detach().cpu().item() if isinstance(l, torch.Tensor) else l
                       for l_k, l in loss_d.items()}
                    )

    def generate_outputs_from_model_inner_step(self, model, data_batch, criterion=None,
                                               input_key='signal_arr', target_key='text_arr', device=None,
                                               ):
        X = data_batch[input_key].to(device)

        if self.squeeze_first:
            X = X.squeeze()

        with torch.no_grad():
            model.eval()
            model.to(device)
            m_d = model(X)

        loss_d = self.loss(m_d)
        score_d = self.score_training(m_d, as_tensor=True)
        eval_d = dict(**score_d, **loss_d)

        return eval_d


@dataclass
class Brain2VecOptions(bmp.ModelOptions):
    model_name: str = 'cog2vec'
    feature_extractor_dropout: float = 0.25
    input_1d_dropout: float = 0.
    feature_grad_mult: float = 1
    negatives_from_everywhere: bool = True
    n_negatives: int = 100
    cross_sample_negatives: int = 0
    codebook_negatives: int = 0
    mask_length: int = 2
    mask_prob: float = 0.3
    n_encoder_heads: int = 4
    n_encoder_layers: int = 6
    encoder_dropout: float = 0.25
    encoder_dim_feedforward: int = 2048
    quant_num_vars: int = 40
    quant_num_groups: int = 2
    quant_weight_proj_factor: int = 2
    quant_weight_proj_depth: int = 1
    feature_extractor_layers: str = '[(128, 7, 2)] + [(64, 3, 2)] * 2 + [(32, 3, 2)] * 2'
    feature_extractor_mode: str = 'layer_norm'
    ras_pos_encoding: bool = True
    ras_batch_norm: bool = False
    ras_noise: float = 0.
    ras_architecture: str = 'simple'
    affine_std: bool = False
    positional_encoding_method: str = 'combined'
    losses_to_output: Optional[str] = None
    squeeze_first: bool = True

    def make_model_kws(self, dataset=None, **kws):
        return dict(
            #input_shape=(1, 256),
            input_shape=(1, dataset.get_feature_shape()[-1]),
            feature_model=None, context_model=None, projection_model=None,
            feature_extractor_dropout=self.feature_extractor_dropout,
            negatives_from_everywhere=self.negatives_from_everywhere,
            feature_grad_mult=self.feature_grad_mult,
            n_negatives=self.n_negatives, codebook_negatives=self.codebook_negatives,
            cross_sample_negatives=self.cross_sample_negatives,
            mask_length=self.mask_length, n_encoder_heads=self.n_encoder_heads,
            context_encoder_dropout=self.encoder_dropout,
            n_encoder_layers=self.n_encoder_layers,
            encoder_dim_feedforward=self.encoder_dim_feedforward,
            quant_num_vars=self.quant_num_vars, quant_num_groups=self.quant_num_groups,
            quant_weight_proj_factor=self.quant_weight_proj_factor,
            quant_weight_proj_depth=self.quant_weight_proj_depth,
            feature_extractor_layers=self.feature_extractor_layers,
            positional_encoding_method=self.positional_encoding_method,
            losses_to_output=self.losses_to_output,
            ras_batch_norm=self.ras_batch_norm,
            ras_noise=self.ras_noise, affine_std=self.affine_std,
            ras_architecture=self.ras_architecture,
            ras_pos_encoding=self.ras_pos_encoding,
            input_1d_dropout=self.input_1d_dropout
        )

    def make_model(self, dataset: Optional[datasets.BaseDataset],
                   in_channels=None, window_size=None):
        model_kws = self.make_model_kws(dataset)
        return Brain2Vec(**model_kws), model_kws


#if __name__ == """__main__""":
#    # Demo the model and trainer for debug/testing - use an experiments interface directly for a full CLI
#    from ecog_speech.experiments import semi_supervised
#    from ecog_speech import utils
#
#    options = utils.build_default_options(semi_supervised.ss_option_kwargs,
#                                          train_sets='UCSD-22',
#                                          device='cpu',
#                                          n_epochs=30)
#
#    results = semi_supervised.run(options)


