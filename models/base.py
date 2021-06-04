"""
Base interfaces
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List


LayerDictRaw = Dict[str, Any]
LayerRefRaw = str
NetDictRaw = Dict[str, LayerDictRaw]


class Layer:
  """
  Represents a layer and its output, created by :class:`ILayerMaker`.
  """
  def __init__(self, *, maker: ILayerMaker, layer_dict: LayerDictRaw, name_ctx: _NameCtx):
    self.maker = maker
    self.layer_dict = layer_dict
    self.name_ctx = name_ctx
    name_ctx.layer = self

  def get_name(self) -> str:
    """
    Return layer name, valid in the current active name context.
    """
    # TODO extend this to go up and go down, relative names...
    assert _NameCtx.stack and self.name_ctx.parent is _NameCtx.stack[-1]
    assert self.name_ctx.name
    return self.name_ctx.name

  def _sis_hash(self):
    from sisyphus.hash import sis_hash_helper
    return sis_hash_helper(self.layer_dict)


NetDict = Dict[str, Layer]


class ILayerMaker:
  """
  Makes a layer.
  """
  def call(self, *args, **kwargs) -> LayerDictRaw:
    """
    Return layer dict.
    """
    raise NotImplementedError

  def __call__(self, *args, name: Optional[str] = None, **kwargs) -> Layer:
    with _NameCtx(maker=self, name=name) as name_ctx:
      return Layer(maker=self, layer_dict=self.call(*args, **kwargs), name_ctx=name_ctx)


class CopyLayer(ILayerMaker):
  """
  Copy the (single) input.
  """
  def call(self, source: Layer) -> LayerDictRaw:
    """
    Create CopyLayer.
    """
    return {"class": "copy", "from": source.get_name()}


class Module(ILayerMaker):
  """
  Like PyTorch.
  This represents a subnetwork in RETURNN, or the root network.
  Or some other layer which has a subnetwork, like RecLayer.
  """

  def call(self, *args, **kwargs) -> Layer:
    """
    Constructs the output.
    """
    raise NotImplementedError

  def __call__(self, *args, **kwargs) -> Layer:
    with _NameCtx(maker=self) as name_ctx:
      res = self.call(*args, **kwargs)
      CopyLayer()(res, name="output")
      layer_dict = {"class": "subnetwork", "from": [], "subnetwork": name_ctx.make_net_dict()}
      return Layer(maker=self, layer_dict=layer_dict, name_ctx=name_ctx)


class _NameCtx:
  stack = []  # type: List[_NameCtx]

  def __init__(self, *, maker: Optional[ILayerMaker], name: Optional[str] = None):
    self.maker = maker
    self.layer = None  # type: Optional[Layer]
    self.childs = {}  # type: Dict[str, _NameCtx]
    self.parent = self.stack[-1] if self.stack else None
    if self.parent:
      assert self.maker
    else:
      assert not self.maker
    self.name = name if name else (self._get_name() if self.parent else None)
    if self.parent:
      assert self.name not in self.parent.childs
      self.parent.childs[self.name] = self

  def make_net_dict(self) -> NetDict:
    """
    Create net dict.
    """
    net_dict = {}
    for key, value in self.childs.items():
      net_dict[key] = value.layer
    return net_dict

  def __enter__(self):
    if self.parent:
      assert self.stack[-1] is self.parent
    else:
      assert not self.stack
    self.stack.append(self)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert self.stack[-1] is self
    self.stack.pop(-1)

  def _get_name(self) -> str:
    assert self.parent and self.parent.maker
    for key, value in vars(self.parent.maker):
      if value is self.maker:
        return key
    return self._get_unique_name()

  def _get_unique_name(self) -> str:
    name = self.maker.__class__.__name__
    reserved_names = set(vars(self.parent.maker).keys()) | set(self.childs.keys()) | {"output"}
    if name not in reserved_names:
      return name
    i = 0
    while True:
      name_ = f"{name}_{i}"
      if name_ not in reserved_names:
        return name_
      i += 1
