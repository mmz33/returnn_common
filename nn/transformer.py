"""
Transformer Modules
"""

from __future__ import annotations
from .base import Module, LayerRef
from .math_ import relu
from ._generated_layers import Linear, layer_norm
from .utils import dropout
from .container import ModuleList
from typing import Optional, Any


class TransformerEncoderLayer(Module):
  """
  Defines one layer of a standard transformer encoder
  """
  def __init__(self, d_model, nhead, dim_feedforward=2048, drop=0.1, act=relu, layer_norm_eps=1e-5,
               norm_first=False) -> None:
    """
    :param d_model: hidden dim
    :param nhead: number heads
    :param dim_feedforward:
    :param drop:
    :param act: activation functional
    :param layer_norm_eps:
    :param norm_first: Whether to do layer norm before or afterwards
    """
    super().__init__()
    self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)  # will change with Attention Modules

    self.linear1 = Linear(dim_feedforward)
    self.linear2 = Linear(d_model)

    self.activation = act
    self.norm_first = norm_first
    self.norm_eps = layer_norm_eps
    self.dropout = drop

  def forward(self, inp: LayerRef) -> LayerRef:
    """
    Two possible forward variants of encoder, defined by self.norm_first
    """
    x = inp
    if self.norm_first:
      x = x + self._sa_block(layer_norm(x, epsilon=self.norm_eps))
      x = x + self._ff_block(layer_norm(x, epsilon=self.norm_eps))
    else:
      x = layer_norm(x + self._sa_block(x), epsilon=self.norm_eps)
      x = layer_norm(x + self._ff_block(x), epsilon=self.norm_eps)

    return x

  # self-attention block
  def _sa_block(self, x: LayerRef) -> LayerRef:
    x = self.self_attn(x, x, x)
    return dropout(x, self.dropout)

  # feed forward block
  def _ff_block(self, x: LayerRef) -> LayerRef:
    x = self.linear2(dropout(self.activation(self.linear1(x)), self.dropout))
    return dropout(x, self.dropout)


class TransformerEncoder(Module):
  """
  Defines the full Encoder of the standard transformer
  """
  def __init__(self, encoder_layer, num_layers, normalization=None, norm_eps: float = 1e-5):
    """
    :param encoder_layer:
    :param num_layers:
    :param normalization:
    :param norm_eps:
    """
    super().__init__()
    import copy
    self.layers = ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    self.num_layers = num_layers
    self.norm = normalization
    self.norm_eps = norm_eps

  def forward(self, inp: LayerRef) -> LayerRef:
    """
    Executes the encoder layer as often as in num_layers defined
    """
    output = inp
    for mod in self.layers:
      output = mod(output)

    if self.norm is not None:
      output = self.norm(output, epsilon=self.norm_eps)

    return output


class TransformerDecoderLayer(Module):
  """
  Defines one layer of a standard transformer decoder
  """
  def __init__(self, d_model, nhead, dim_feedforward=2048, drop=0.1, act=relu, layer_norm_eps=1e-5, norm_first=False):
    """
    :param d_model: hidden dim
    :param nhead: number heads
    :param dim_feedforward:
    :param drop:
    :param act: activation functional
    :param layer_norm_eps:
    :param norm_first: Whether to do layer norm before or afterwards
    """
    super().__init__()
    self.dropout = drop
    self.self_attn = MultiheadAttention(d_model, nhead, dropout=self.dropout)  # will change with Attention Modules
    self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=self.dropout)  # will change with Attention Modules

    self.linear1 = Linear(dim_feedforward)
    self.linear2 = Linear(d_model)

    self.norm_first = norm_first
    self.norm_eps = layer_norm_eps
    self.activation = act

  def forward(self, tgt: LayerRef, memory: LayerRef) -> LayerRef:
    """
    Two possible forward variants of decoder, defined by self.norm_first
    """
    x = tgt
    if self.norm_first:
      x = x + self._sa_block(layer_norm(x, epsilon=self.norm_eps))
      x = x + self._mha_block(layer_norm(x, epsilon=self.norm_eps), memory)
      x = x + self._ff_block(layer_norm(x, epsilon=self.norm_eps))
    else:
      x = layer_norm(x + self._sa_block(x), epsilon=self.norm_eps)
      x = layer_norm(x + self._mha_block(x, memory), epsilon=self.norm_eps)
      x = layer_norm(x + self._ff_block(x), epsilon=self.norm_eps)

    return x

  # self-attention block
  def _sa_block(self, x: LayerRef) -> LayerRef:
    x = self.self_attn(x, x, x)  # will change with Attention Modules
    return dropout(x, self.dropout)

  # multihead attention block
  def _mha_block(self, x: LayerRef, mem: LayerRef) -> LayerRef:
    x = self.multihead_attn(x, mem, mem)  # will change with Attention Modules
    return dropout(x, self.dropout)

  # feed forward block
  def _ff_block(self, x: LayerRef) -> LayerRef:
    x = self.linear2(dropout(self.activation(self.linear1(x)), self.dropout))
    return dropout(x, self.dropout)


class TransformerDecoder(Module):
  """
  Defines the full Decoder of the standard transformer
  """
  def __init__(self, decoder_layer, num_layers, normalization=None, norm_eps: float = 1e-5):
      """
      :param decoder_layer:
      :param num_layers:
      :param normalization:
      :param norm_eps:
      """
      super(TransformerDecoder, self).__init__()
      import copy
      self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
      self.num_layers = num_layers
      self.norm = normalization
      self.norm_eps = norm_eps

  def forward(self, tgt: LayerRef, memory: LayerRef) -> LayerRef:
    """
    Executes the decoder layer as often as in num_layers defined
    """
    output = tgt
    for mod in self.layers:
      output = mod(output, memory)

    if self.norm is not None:
      output = self.norm(output, epsilon=self.norm_eps)

    return output


class Transformer(Module):
  """
  Standard Transformer Module
  """
  def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
               num_decoder_layers: int = 6, dim_feedforward: int = 2048, drop: float = 0.1,
               act=relu, custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
               layer_norm_eps: float = 1e-5) -> None:
    super().__init__()

    if custom_encoder is not None:
      self.encoder = custom_encoder
    else:
      encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, drop, act, layer_norm_eps)
      self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, layer_norm, layer_norm_eps)

    if custom_decoder is not None:
      self.decoder = custom_decoder
    else:
      decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, drop, act, layer_norm_eps)
      self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, layer_norm, layer_norm_eps)

    self.norm_eps = layer_norm_eps
    self.d_model = d_model
    self.nhead = nhead

  def forward(self, src: LayerRef, tgt: LayerRef) -> LayerRef:
    """
    Forward step of Transformer
    """
    memory = self.encoder(src)
    output = self.decoder(tgt, memory)
    return output
