"""
Base interfaces.

The core interfaces for the user are:

* :class:`ILayerMaker`, to directly create a layer dict.
  We recommend using this only for directly wrapping RETURNN layers
  and not for any higher-level logic,
  which should be done as a :class:`Module`.

* :class:`Module`, to write PyTorch-style code, which acts like a subnetwork.
  We recommend using this as the base interface
  for any higher-level interfaces
  (such as a generic decoder interface).

Instances of both objects can be called directly,
and return instances of type :class:`LayerRef`,
which can be thought of as analogue to :class:`torch.Tensor` or :class:`tf.Tensor`.

Use ``x.mark_as_loss()`` to mark some output (layer ref) as a loss.

The root network should be a :class:`Module`,
and then you can use ``make_root_net_dict()``
to get the network dict.
Code example::

    class Network(Module):
      def __init__(self):
        super().__init__()
        self.lstm = Lstm(n_out=1024)

      def forward(self, x):
        y = self.lstm(x)
        return y

    net = Network()
    net_dict = make_root_net_dict(net, "data")

---

Code conventions:

- Usual, as in RETURNN, PEP8, 2-space indents, 120 char line limit.
- Pure interface classes are prefixed with `I`.
  (`Module` is an exception because this is made analogue to PyTorch).

"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union, Set, Iterator
from returnn.util.basic import NotSpecified, OptionalNotImplementedError
from returnn.tf.util.data import DimensionTag
from tensorflow.python.util import nest


LayerDictRaw = Dict[str, Any]
LayerRefRaw = str
NetDictRaw = Dict[str, LayerDictRaw]
RawTensorTypes = Union[int, float, complex, bool, str]


class LayerRef:
  """
  Refers to a layer.

  An instance of this class can be treated very much like a tensor.
  It supports all the common unary and binary math operations such as addition.
  This is the intended view point for the user,
  to treat instances of this class like a tensor.

  For most layers, instead of just having an instance of :class:`LayerRef`,
  you would instead directly have an instance of :class:`Layer`.

  You do not create instances of this object explicitly
  but they are created via :func:`get_special_layer` or :class:`NameCtx.get_child_layer_ref`,
  or layers (:class:`Layer`) via any of the layer makers, modules or other functions.
  """

  def __init__(self, *, name_ctx: NameCtx):
    self.name_ctx = name_ctx
    assert name_ctx.layer_ref is None
    name_ctx.layer_ref = self

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.name_ctx}>"

  def get_name(self) -> str:
    """
    :return: RETURNN layer name, valid in the current active name context.
    """
    return self.name_ctx.get_name_in_current_ctx()

  def get_abs_name(self) -> str:
    """
    :return: absolute RETURNN layer name starting from root context.
    """
    return self.name_ctx.get_abs_name()

  def mark_as_loss(self, loss_scale: Optional[float] = 1.0):
    """
    Mark this as a loss.
    """
    raise TypeError("mark_as_loss can only be called on a layer, not a layer-ref.")

  def __add__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="add", name="add")

  def __sub__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="sub", name="sub")

  def __mul__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="mul", name="mul")

  def __truediv__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="truediv", name="truediv")

  def __radd__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([convert_to_layer_ref(other), self], kind="add", name="add")

  def __rsub__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([convert_to_layer_ref(other), self], kind="sub", name="sub")

  def __rmul__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([convert_to_layer_ref(other), self], kind="mul", name="mul")

  def __rtruediv__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([convert_to_layer_ref(other), self], kind="truediv", name="truediv")

  def __neg__(self) -> LayerRef:
    from . import eval
    return eval(self, eval="-source(0)", name="neg")

  def __invert__(self) -> LayerRef:
    from . import eval
    return eval(self, eval="tf.logical_not(source(0))", name="invert")

  def __pow__(self, other: Union[RawTensorTypes, LayerRef], modulo=None) -> LayerRef:
    assert modulo is None
    from . import eval
    return eval([self, convert_to_layer_ref(other)], eval="tf.math.pow(source(0), source(1))", name="pow")

  def __rpow__(self, other: Union[RawTensorTypes, LayerRef], modulo=None) -> LayerRef:
    assert modulo is None
    from . import eval
    return eval([convert_to_layer_ref(other), self], eval="tf.math.pow(source(0), source(1))", name="pow")

  def __and__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="logical_and", name="logical_and")

  def __or__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="logical_or", name="logical_or")

  def __abs__(self) -> LayerRef:
    from . import eval
    return eval(self, eval="tf.abs(source(0))", name="abs")

  def __ceil__(self) -> LayerRef:
    from . import eval
    return eval(self, eval="tf.math.ceil(source(0))", name="ceil")

  def __floor__(self) -> LayerRef:
    from . import eval
    return eval(self, eval="tf.math.floor(source(0))", name="floor")

  def __floordiv__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from . import eval
    return eval([self, convert_to_layer_ref(other)], eval="tf.math.floordiv(source(0), source(1))", name="floordiv")

  def __eq__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from . import compare
    return compare([self, convert_to_layer_ref(other)], kind="equal", name="equal")

  def __ne__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from . import compare
    return compare([self, convert_to_layer_ref(other)], kind="not_equal", name="not_equal")

  def __lt__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from . import compare
    return compare([self, convert_to_layer_ref(other)], kind="less", name="less")

  def __le__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from . import compare
    return compare([self, convert_to_layer_ref(other)], kind="less_equal", name="less_equal")

  def __gt__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from . import compare
    return compare([self, convert_to_layer_ref(other)], kind="greater", name="greater")

  def __ge__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from . import compare
    return compare([self, convert_to_layer_ref(other)], kind="greater_equal", name="greater_equal")


class Layer(LayerRef):
  """
  Represents a layer and its output, created by :class:`ILayerMaker` or :func:`make_layer`.
  You would not create an instance of this explicitly.
  """

  def __init__(self, *, layer_dict: LayerDictRaw, name_ctx: NameCtx):
    super(Layer, self).__init__(name_ctx=name_ctx)
    assert self.name_ctx.layer is None
    self.name_ctx.layer = self
    self.layer_dict = layer_dict

  def mark_as_loss(self, loss_scale: Optional[float] = 1.0):
    """
    Mark this as a loss.
    """
    assert "loss" not in self.layer_dict
    self.layer_dict["loss"] = "as_is"
    if loss_scale is not None:
      assert "loss_scale" not in self.layer_dict
      self.layer_dict["loss_scale"] = loss_scale

  def _sis_hash(self):
    from sisyphus.hash import sis_hash_helper  # noqa
    return sis_hash_helper(self.layer_dict)


class ILayerMaker:
  """
  Makes a RETURNN layer.

  Also see :func:`make_layer` and :class:`Module`.

  A RETURNN layer also has some specific input and output,
  and usually its own parameters.

  This is in contrast to PyTorch or Keras, where a module or layer
  has params, but getting some output for some input
  requires an additional `forward` call,
  which can be called multiple times.
  Every such call would then share the same module parameters.

  :class:`ILayerMaker` is similar to PyTorch/Keras
  in that it can be called multiple times.
  Every call would create a RETURNN layer,
  where every call after the first would share the params
  with the first layer,
  via the RETURNN ``reuse_params`` layer option.

  A user would create an instance and then call it,
  and get :class:`Layer` instances.
  The naming logic of created layers
  is handled via :class:`NameCtx`.

  A developer which wants to derive its own layer maker
  would overwrite the :func:`make_layer_dict`.
  Usually this is never needed though,
  as all standard RETURNN layers are already wrapped,
  and any potential operation should be possible to be defined
  using the standard RETURNN layers.
  For one-time usages, :func:`make_layer` is probably easier.
  For defining own modules (subnetworks)
  based on existing modules or layers,
  see :class:`Module`.
  """
  has_variables: bool = True
  layer_name_scope = NotSpecified  # type: Union[NotSpecified, str]

  def __init__(self):
    """
    By convention, any options to the layer maker or module are passed to the constructor,
    and potential changing inputs (other layers)
    are passed to :func:`__call__` (:func:`make_layer_dict`).
    """
    # Actually we would want an ordered set for parents, but Python does not provide this.
    # We abuse a dict as a set. This is ordered since Python 3.6, see #43.
    # Also note that the current code does not clean this up when you do delattr later or so.
    self._parents = {}  # type: Dict[Tuple[ILayerMaker, str], None]  # (parent,attrib) -> None
    self.calls = []  # type: List[NameCtx]

  def __repr__(self):
    return f"<{self.__class__.__name__}>"

  def make_layer_dict(self, *args, **kwargs) -> LayerDictRaw:
    """
    :return: layer dict.

    The arguments are usually other layers, via :class:`LayerRef` instances.
    The returned :class:`LayerDictRaw` can reference other layers by using :class:`LayerRef` instances.
    It can further contain subnetworks via :class:`Net` instances,
    although you mostly would not use this in custom implementations
    but use :class:`Module` or :class:`Loop` instead.

    This function is implemented by derived classes.
    This function will be called by :func:`__call__` with all arguments forwarded.
    """
    raise NotImplementedError

  def default_initial_state(self) -> LayerState:
    """
    :return: default initial state, to be used if the module (layer) has recurrent (hidden) state.
      When a module has recurrent state,
      the convention is to return a tuple with instance :class:`LayerState` as the last item,
      and to accept the ``state`` argument with a :class:`LayerState` with the same nested structure.
      This can be a nested structure and should match the structure of the ``state`` argument and returned value.
    """
    raise OptionalNotImplementedError

  def get_default_name(self) -> str:
    """
    Get a default layer name (used when we do not have a Module attribute pointing to this).
    """
    name = self.__class__.__name__
    if name.startswith("_"):
      name = name[1:]
    if name[:1].isupper():
      from returnn.util.basic import camel_case_to_snake_case
      name = camel_case_to_snake_case(name)
    return name

  def _make_layer(self, *args, **kwargs) -> Layer:
    name_ctx = NameCtx.top()
    assert name_ctx.maker is self
    if self.calls:
      name_ctx.is_repeated_call = True
    layer_dict = self.make_layer_dict(*args, **kwargs)
    return make_layer(layer_dict, name_ctx=name_ctx)

  def __call__(self, *args, name: Optional[Union[str, NameCtx]] = None, **kwargs) -> Layer:
    """
    This calls :func:`make_layer_dict` internally and creates a corresponding :class:`Layer` instance.
    """
    with NameCtx.get_from_call(maker=self, name=name):
      return self._make_layer(*args, **kwargs)

  def __setattr__(self, key: str, value):
    super().__setattr__(key, value)
    if isinstance(value, ILayerMaker):
      value._parents[(self, key)] = None

  def parents_with_attr(self) -> Iterator[Tuple[ILayerMaker, str]]:
    """
    Get all (immediate) parent makers, and the attrib name which points to us
    """
    # We rely on deterministic order of dict.
    for parent, attr in self._parents.keys():
      # We currently don't do proper cleanup of _parents via delattr etc,
      # so explicitly check.
      if getattr(parent, attr, None) is self:
        yield parent, attr

  def children(self) -> Iterator[ILayerMaker]:
    """
    Get all (immediate) children makers
    """
    for name, child in self.named_children():
      yield child

  def named_children(self) -> Iterator[Tuple[str, ILayerMaker]]:
    """
    Get all (immediate) children makers
    """
    return iter([])

  def children_deep(self) -> Iterator[ILayerMaker]:
    """
    Get all children (deeply)
    """
    for name, child in self.named_children_deep():
      yield child

  def named_children_deep(self, memo: Optional[Set[ILayerMaker]] = None, prefix: str = ''):
    """
    Get all children (deeply)
    """
    if memo is None:
      memo = set()
    if self not in memo:
      memo.add(self)
      yield prefix, self
      for name, maker in self.named_children():
        if maker is None:
          continue
        sub_prefix = prefix + ('.' if prefix else '') + name
        for m in maker.named_children_deep(memo, sub_prefix):
          yield m


class LayerState(dict):
  """
  Covers all the state of a layer,
  i.e. exactly what needs to be stored and passed into the module or layer maker
  next time you call it as initial state.

  This behaves somewhat like a namedtuple, although we derive from dict.
  """
  def __init__(self, *args, **kwargs):
    if kwargs:
      assert not args
      super().__init__(**kwargs)
    elif args:
      assert len(args) == 1
      super().__init__(state=args[0])
    else:
      super().__init__()

  def __repr__(self):
    return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for (k, v) in self.items())})"

  def __getattr__(self, item):
    if item in self:
      return self[item]
    raise AttributeError(f"{self}.{item}")


# noinspection PyAbstractClass
class _ReturnnWrappedLayerBase(ILayerMaker):
  """
  Base class for all automatically wrapped layers.
  """
  returnn_layer_class: Optional[str] = None
  has_recurrent_state: bool = False

  def _get_recurrent_state(self, layer: Layer) -> LayerState:
    """
    :returns: the recurrent state

    You might override this in case the state is more complex,
    and return some named tuple or any other hierarchical structure.
    """
    assert self.has_recurrent_state
    from ._generated_layers import _get_last_hidden_state
    # Note that this is actually layer specific.
    # We try to use a number of heuristics to get it right for the common cases.
    name = f"{layer.name_ctx.name}_state"
    n_out = layer.layer_dict["n_out"]
    if layer.layer_dict["class"] == "rec" and isinstance(layer.layer_dict["unit"], str):
      if "lstm" in layer.layer_dict["unit"].lower():
        h = _get_last_hidden_state(layer, n_out=n_out, key="h", name=f"{name}_h")
        c = _get_last_hidden_state(layer, n_out=n_out, key="c", name=f"{name}_c")
        return LayerState(h=h, c=c)
    return LayerState(_get_last_hidden_state(layer, n_out=n_out, name=name))

  def __call__(self, *args,
               name: Optional[Union[str, NameCtx]] = None,
               **kwargs
               ) -> Union[Layer, Tuple[Layer, LayerState]]:
    with NameCtx.get_from_call(maker=self, name=name):
      layer = self._make_layer(*args, **kwargs)
      if not self.has_recurrent_state:
        return layer
    state = self._get_recurrent_state(layer)
    return layer, state

  def default_initial_state(self) -> LayerState:
    """
    :return: default initial state
    """
    assert self.has_recurrent_state
    # Match the logic of _get_recurrent_state above.
    if self.returnn_layer_class == "rec":
      unit = getattr(self, "unit")
      if isinstance(unit, str):
        if "lstm" in unit.lower():
          return LayerState(h=0, c=0)  # TODO get real shape... how to get batch dim?
    raise NotImplementedError(f"{self}.default_initial_state")


def make_layer(layer_dict: LayerDictRaw, *,
               name: Optional[str] = None, name_ctx: Optional[NameCtx] = None) -> Layer:
  """
  Creates the layer. This also registers the layer instance in the top name ctx.
  When no name is given, this assumes that the top name ctx corresponds to this layer maker.

  This is used internally via :class:`ILayerMaker`
  but might also be used to wrap simple RETURNN layers.
  If a layer has params and you want the param sharing logic,
  you should instead derive a new class from :class:`ILayerMaker`.
  Usually, you do not need either of these,
  as all standard layers should already be wrapped,
  and it should be possible to define any possible logic
  using that.
  (If this is not the case, please report an issue.)

  :param LayerDictRaw layer_dict: can contain :class:`LayerRef` instances
  :param str|None name: (suggested) layer name. if given, will create a new :class:`NameCtx`
  :param NameCtx|None name_ctx: if given, will use this name ctx.
    You can either pass ``name_ctx`` or ``name`` but not both.
  """
  if name:
    assert not name_ctx
    assert isinstance(name, str)
    name_ctx = NameCtx(suggested_name=name)
    return make_layer(layer_dict=layer_dict, name_ctx=name_ctx)
  if name_ctx:
    assert isinstance(name_ctx, NameCtx)
    if NameCtx.top() is name_ctx:
      pass  # go on
    else:
      with name_ctx:
        return make_layer(layer_dict=layer_dict)
  else:
    name_ctx = NameCtx.top()
  assert not name_ctx.layer_ref and not name_ctx.layer  # not yet assigned
  layer_dict = layer_dict.copy()

  if name_ctx.maker and name_ctx.maker.has_variables:
    # We must check whether the RETURNN abs layer name is consistent with our module naming hierarchy,
    # and make it consistent if not (https://github.com/rwth-i6/returnn_common/issues/25).
    if name_ctx.is_root:
      pass  # nothing to do
    else:
      # The parent name ctx RETURNN layer will also have the right name_scope set,
      # so this layers name scope default is simply based on that.
      layer_abs_name_scope_parent = name_ctx.parent.layer_abs_name_scope
      if layer_abs_name_scope_parent:
        layer_abs_name_scope_parent += "/"
      layer_abs_name_scope_default = layer_abs_name_scope_parent + name_ctx.name
      if layer_abs_name_scope_default != name_ctx.layer_abs_name_scope:  # default does not match what we require
        assert "name_scope" not in layer_dict
        if name_ctx.layer_abs_name_scope == name_ctx.parent.layer_abs_name_scope:
          layer_dict["name_scope"] = ""
        elif name_ctx.layer_abs_name_scope.startswith(layer_abs_name_scope_parent):  # can use relative
          layer_dict["name_scope"] = name_ctx.layer_abs_name_scope[len(layer_abs_name_scope_parent):]
        else:  # must use absolute
          layer_dict["name_scope"] = "/" + name_ctx.layer_abs_name_scope

  name_ctx.is_subnet_ctx = False
  if name_ctx.maker and name_ctx.maker.calls:
    name_ctx.is_repeated_call = True
  layer = Layer(layer_dict=layer_dict, name_ctx=name_ctx)
  if name_ctx.maker:
    name_ctx.maker.calls.append(name_ctx)
  return layer


def convert_to_layer_ref(x: Union[LayerRef, int, float, complex, bool, str]) -> LayerRef:
  """
  In case it is not a layer ref yet, it will make some constant.
  """
  if isinstance(x, LayerRef):
    return x
  from . import constant
  return constant(value=x)


class Module(ILayerMaker):
  """
  This represents a subnetwork in RETURNN, or the root network.

  You can write PyTorch-like code here, like::

      class MyModule(Module):

       def __init__(self, dim: int, activation=tanh):
         super().__init__()
         self.linear = Linear(dim)
         self.activation = activation

       def forward(self, x: LayerRef) -> LayerRef:
         x_ = x
         x = layer_norm(x)
         x = self.linear(x)
         x = self.activation(x)
         return x_ + x

  """

  def forward(self, *args, **kwargs) -> LayerRef:
    """
    Constructs the output.
    You can write PyTorch-style code here.
    """
    raise NotImplementedError

  def __call__(self, *args, name: Optional[Union[str, NameCtx]] = None, **kwargs) -> Union[Layer, Any]:
    from . import copy
    with NameCtx.get_from_call(maker=self, name=name) as name_ctx:
      name_ctx.is_subnet_ctx = True
      res = self.forward(*args, **kwargs)
      if name_ctx.parent is None:  # root
        # special logic, no output layers, no subnetwork layer needed
        self.calls.append(name_ctx)
        return res
      if isinstance(res, LayerRef):
        copy(res, name=name_ctx.get_child("output"))
      else:
        # we return more than one layer (thus also working on other layers of the subnet, that are not output)
        # by convention: first layer is the output layer
        res_flat = nest.flatten(res)
        copy(res_flat[0], name=name_ctx.get_child("output"))
      subnet_layer = self._make_layer()
    if isinstance(res, LayerRef):
      return subnet_layer  # maybe nicer to return subnet layer
    return res

  def make_layer_dict(self) -> LayerDictRaw:
    """
    Make subnet layer dict.
    """
    name_ctx = NameCtx.top()
    assert name_ctx.maker is self
    return {"class": "subnetwork", "from": [], "subnetwork": name_ctx.make_net()}

  def named_children(self) -> Iterator[Tuple[str, ILayerMaker]]:
    """
    Get all (immediate) children makers
    """
    # We rely on deterministic order of dict and vars.
    for key, value in vars(self).items():
      if isinstance(value, ILayerMaker):
        yield key, value

  @property
  def has_variables(self):
    """
    Whether this module has variables
    """
    for maker in self.children():
      if maker.has_variables:
        return True
    return False


def make_root_net_dict(model: Module, *args, **kwargs) -> NetDictRaw:
  """
  Make net dict, to be used as the main RETURNN network, not within a subnetwork.
  Any passed arguments are keys of extern data,
  and are forwarded to the module.
  """
  assert isinstance(model, Module)
  from . import copy
  with NameCtx(maker=model, parent=None) as name_ctx:
    name_ctx.is_subnet_ctx = True
    args = tuple(get_extern_data(arg) for arg in args)
    kwargs = {key: get_extern_data(value) for (key, value) in kwargs.items()}
    res = model.forward(*args, **kwargs)
    if "output" not in name_ctx.children:
      if isinstance(res, LayerRef):
        copy(res, name=name_ctx.get_child("output"))
      else:
        res_list = nest.flatten(res)
        assert res_list and isinstance(res_list[0], LayerRef)
        copy(res_list[0], name=name_ctx.get_child("output"))
    net = name_ctx.make_net()
  return net.make_net_dict_raw()


class Loop:
  """
  This represents a RecLayer subnetwork in RETURNN,
  i.e. where the calculation per step is defined explicitly.

  (For RecLayer with a predefined unit, see :class:`Rec`.
   Or for example :class:`Lstm`.)

  To define a loop like this pseudo Python code::

    x  # given, shape (batch, time, dim)
    h = Zeros([batch,dim])()  # initial state, shape (batch,dim)
    out = []
    for t in range(x.max_seq_len):
      x_lin = Linear(dim)(x[t])
      h_prev = h
      h = Linear(dim)(x_lin + h_prev)
      out.append(h)

    h  # final state
    out  # shape (time, batch, h_dim)

  You would write::

    with Loop() as loop:
      x_t = loop.unstack(x)
      x_lin = Linear(dim)(x_t)
      loop.state.h = State(shape=[batch,dim], initial=0)  # optional
      loop.state.h = Linear(dim)(x_lin + loop.state.h)
      out = loop.stack(loop.state.h)

  ``state`` is :class:`Loop._StateHolder` and manages the recurrent state.

  This code must be run within a :func:`Module.forward`
  or with some active global name context (:class:`NameCtx`).

  This API is currently in development, and might change.
  See: https://github.com/rwth-i6/returnn_common/issues/16
  """

  def __init__(self, *,
               max_seq_len: Optional[Union[str, int]] = NotSpecified,
               optimize_move_layers_out: Optional[bool] = NotSpecified,
               cheating: bool = NotSpecified,
               unroll: bool = NotSpecified,
               back_prop: Optional[bool] = NotSpecified,
               use_global_rec_step_offset: bool = NotSpecified,
               include_eos: bool = NotSpecified,
               debug: Optional[bool] = NotSpecified,
               name: str = "loop"
               ):
    super(Loop, self).__init__()
    self.extra_opts = {
      key: value for (key, value) in locals().items()
      if value is not NotSpecified and key not in {"self", "__class__", "name"}}
    self.layer_maker = _LoopLayerMaker(loop=self)
    self.name_ctx = NameCtx(maker=self.layer_maker, suggested_name=name, parent=NameCtx.current_ctx())
    self.name_ctx.is_subnet_ctx = True
    self.name_ctx.extend_reserved_names({"output", "end"})
    self.state = _StateHolder(loop=self)
    self.unstacked_refs = []  # type: List[LayerRef]
    self.outputs = []  # type: List[LayerRef]
    self.end_ref = None  # type: Optional[LayerRef]

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.name_ctx.get_abs_name_repr()}>"

  def __enter__(self) -> Loop:
    self.name_ctx.__enter__()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    try:
      if not exc_type:
        if not self.outputs:  # stack or last was called at least once, so we have some output
          raise Exception(f"{self}: call `stack` or `last` at least once to define some output")
        if not self.end_ref and not self.unstacked_refs:
          raise Exception(f"{self}: call `unstack` or `end` at least once to define the loop length")
        # Make sure there is an "output" layer. (Similar as for Module with subnetwork.)
        if "output" not in self.name_ctx.children:
          from . import copy
          copy(self.outputs[0], name=self.name_ctx.get_child("output"))
    finally:
      self.name_ctx.__exit__(exc_type, exc_val, exc_tb)
    if not exc_type:
      self.layer_maker(name=self.name_ctx)

  def unstack(self, source: LayerRef, *, axis: Union[str, DimensionTag], name: Optional[str] = None) -> LayerRef:
    """
    Unrolls over the specified axis, and provides each frame in each loop iteration.
    """
    from . import rec_unstack
    res = rec_unstack(source, axis=axis, name=name)
    self.unstacked_refs.append(res)
    return res

  def stack(self, source: LayerRef, *, name: Optional[str] = None) -> LayerRef:
    """
    Accumulates the frames of source within the loop,
    to make it accessible outside the loop.
    """
    from . import copy
    if not name and "output" not in self.name_ctx.children:
      name = self.name_ctx.get_child("output")
    res = copy(source, name=name)
    assert isinstance(res, Layer)
    if res.name_ctx.name != "output":
      res.layer_dict["is_output_layer"] = True
    self.outputs.append(res)
    return res

  def last(self, source: LayerRef, *, name: Optional[str] = None) -> LayerRef:
    """
    Gets the last value from source.
    """
    # TODO ...
    raise NotImplementedError("Loop.last not implemented yet...")

  def end(self, source: LayerRef) -> LayerRef:
    """
    For loops with dynamic ending condition (which might not use unstack),
    this defines the ending condition.
    """
    assert not self.end_ref  # do not call this multiple times
    from . import copy
    self.end_ref = copy(source, name=self.name_ctx.get_child("end"))
    return self.end_ref


class _LoopLayerMaker(ILayerMaker):
  layer_name_scope = ""

  def __init__(self, loop: Loop):
    super(_LoopLayerMaker, self).__init__()
    self.loop = loop

  def make_layer_dict(self) -> LayerDictRaw:
    """
    Makes layer dict for this loop, i.e. a RecLayer.
    """
    name_ctx = NameCtx.top()
    assert name_ctx.maker is self
    return {"class": "rec", "from": [], "unit": name_ctx.make_net(), **self.loop.extra_opts}

  def named_children(self) -> Iterator[Tuple[str, ILayerMaker]]:
    """
    Children
    """
    # We rely on deterministic order of dict.
    for name, sub_name_ctx in self.loop.name_ctx.children.items():
      if sub_name_ctx.maker:
        yield name, sub_name_ctx.maker

  @property
  def has_variables(self):
    """
    Whether this module has variables
    """
    for maker in self.children():
      if maker.has_variables:
        return True
    return False


class PrevLayerRef(LayerRef):
  """
  Refers to a layer from the previous loop iteration.
  """
  @classmethod
  def get_prev_ref(cls, *, cur_layer_name_ctx: NameCtx) -> PrevLayerRef:
    """
    Create prev ref.
    """
    parent_name_ctx = cur_layer_name_ctx.parent
    prev_layer_name_ctx = parent_name_ctx.get_child(f"prev:{cur_layer_name_ctx.name}")
    if prev_layer_name_ctx.layer_ref:
      prev_layer_ref = prev_layer_name_ctx.layer_ref
      assert isinstance(prev_layer_ref, PrevLayerRef)
      assert prev_layer_ref.cur_layer_name_ctx is cur_layer_name_ctx
    else:
      prev_layer_ref = PrevLayerRef(name_ctx=prev_layer_name_ctx, cur_layer_name_ctx=cur_layer_name_ctx)
      assert prev_layer_name_ctx.layer_ref is prev_layer_ref
    return prev_layer_ref

  def __init__(self, *, name_ctx: NameCtx, cur_layer_name_ctx: NameCtx):
    # At the time we instantiate this, cur_layer_name_ctx.layer probably does not exist yet.
    super().__init__(name_ctx=name_ctx)
    self.cur_layer_name_ctx = cur_layer_name_ctx


class _StateHolder:
  def __init__(self, loop: Loop):
    self._loop = loop
    self._state = {}  # type: Dict[str, State]

  def __repr__(self):
    return f"{self._loop}.state"

  def _get_state(self, name: str) -> State:
    if name in self._state:
      return self._state[name]
    raise AttributeError(f"{self}: Unknown state attrib {name!r}. Assign the initial state first.")

  def __getattr__(self, item):
    return self._get_state(item).get()

  def __setattr__(self, key, value):
    if key in {"_state", "_loop"}:
      return super().__setattr__(key, value)
    if isinstance(value, State):
      # noinspection PyProtectedMember
      value._set_name_and_loop(name=key, loop=self._loop)
      self._state[key] = value
      return
    self._get_state(key).assign(value)


class State:
  """
  Represents some recurrent state, to be used with :class:`Loop`.
  It can also represent some nested hierarchy of states.
  """

  def __init__(self, *, initial: Union[LayerRef, Any]):
    """
    :param initial: some layer-ref, or any kind of nested structure of layers.
    """
    super(State, self).__init__()
    assert initial is not None
    self.initial = initial
    self.loop = None  # type: Optional[Loop]
    self.name = None  # type: Optional[str]
    self.name_ctx = None  # type: Optional[Union[NameCtx, Any]]  # same nested structure as initial
    self.assigned_value = None

  def _set_name_and_loop(self, *, name: str, loop: Loop):
    """
    Assigns the name (internally on first assignment).
    """
    if self.name == name and self.loop is loop:
      return
    assert not self.loop and not self.name and not self.name_ctx  # not yet assigned
    self.loop = loop
    self.name = name
    self.name_ctx = nest.map_structure_with_tuple_paths(
      lambda path, ref: NameCtx(suggested_name='.'.join(str(key) for key in ('state', name) + path)),
      self.initial)

  def assign(self, value):
    """
    Assign the new value for the current iteration.
    """
    assert self.name_ctx is not None
    assert value is not None
    assert self.assigned_value is None, (
      f"Cannot assign the rec state {self.loop}/{self.name} multiple times, "
      f"assigned previously to {self.assigned_value}, now to {value}")
    nest.assert_same_structure(self.initial, value)
    nest.assert_same_structure(self.name_ctx, value)
    self.assigned_value = value

    def _map_ref_to_name_ctx(layer_ref: LayerRef, name_ctx: NameCtx, initial: Any):
      assert isinstance(layer_ref, LayerRef)
      assert isinstance(name_ctx, NameCtx)

      # Potential optimization for RETURNN layers.
      # See _ReturnnWrappedLayerBase._get_recurrent_state.
      if isinstance(layer_ref, Layer):
        if layer_ref.layer_dict["class"] == "get_last_hidden_state":
          used_state_eliminate_optimization = False
          key = layer_ref.layer_dict.get("key", "state")
          src = layer_ref.layer_dict["from"]
          assert isinstance(src, Layer)
          layer_dict_opt_name = "state"
          if layer_dict_opt_name not in src.layer_dict:
            # TODO this should be cleaned up. currently we only really use initial_state in generated layers...
            #   https://github.com/rwth-i6/returnn/issues/732
            #   This is actually incorrect, but not here but in the generated layers...
            #     https://github.com/rwth-i6/returnn_common/issues/31
            layer_dict_opt_name = "initial_state"
          src_state_opt = src.layer_dict.get(layer_dict_opt_name)
          if isinstance(src_state_opt, LayerState):
            src_state_for_key = src_state_opt.get(key)
            if isinstance(src_state_for_key, PrevLayerRef):
              if src_state_for_key.cur_layer_name_ctx is name_ctx:
                # The 'state' argument of the rec layer refers to "prev:..." of the state.
                # So we don't need to pass it now.
                used_state_eliminate_optimization = True
                src_state_opt[key] = None
                # We need to pass the initial_state instead though.
                src_initial_state_opt = src.layer_dict.setdefault("initial_state", LayerState())
                src_initial_state_opt[key] = initial
                # If there is any other code which refers to this state, it can access the passed layer.
                # So anyway pass through.

          if not used_state_eliminate_optimization:
            raise NotImplementedError(
              f"{self}.assign to {layer_ref} on {src}:"
              f" We need https://github.com/rwth-i6/returnn_common/issues/31"
              f" and https://github.com/rwth-i6/returnn/issues/732.")

      _move_layer_ref_to_new_name_ctx(layer_ref=layer_ref, name_ctx=name_ctx)

    nest.map_structure(_map_ref_to_name_ctx, value, self.name_ctx, self.initial)

  @staticmethod
  def _map_name_ctx_to_prev_layer_ref(name_ctx: NameCtx) -> PrevLayerRef:
    assert isinstance(name_ctx, NameCtx)
    return PrevLayerRef.get_prev_ref(cur_layer_name_ctx=name_ctx)

  def get(self):
    """
    Return prev or current value
    """
    assert self.name_ctx is not None
    if self.assigned_value is None:  # not yet assigned
      # Return prev value
      return nest.map_structure(self._map_name_ctx_to_prev_layer_ref, self.name_ctx)
    return self.assigned_value


def get_root_extern_data(data_key: str) -> LayerRef:
  """
  Get extern data from root.
  """
  scope = NameCtx.top()  # must exist
  scope_abs = scope.get_abs_name_ctx_list()
  root_scope = scope_abs[0]
  root_layer_name = f"data:{data_key}"
  return get_special_layer(root_layer_name, scope=root_scope)


def get_extern_data(data_key: str) -> LayerRef:
  """
  Get extern data from current scope.
  """
  assert isinstance(data_key, str)
  return get_special_layer(f"data:{data_key}")


def get_special_layer(name: str, *, scope: Optional[NameCtx] = None) -> LayerRef:
  """
  Special layer can be "data:..." or whatever.
  """
  if not scope:
    scope = NameCtx.current_ctx()  # must exist
  return scope.get_child_layer_ref(name)


def get_sub_layer(layer: LayerRef, name: str) -> LayerRef:
  """
  Like the "{layer}/{name}" syntax in RETURNN.
  Normally this should only be needed for internal usage.
  """
  return layer.name_ctx.get_child_layer_ref(name)


class Net:
  """
  Represents a RETURNN (sub) network.
  It can create a net dict when needed.
  """
  def __init__(self, *, name_ctx: NameCtx):
    self.name_ctx = name_ctx

  def _map_elem_resolve(self, obj: Any) -> Any:
    if isinstance(obj, LayerRef):
      return obj.name_ctx.get_name_in_ctx(ctx=self.name_ctx)
    if isinstance(obj, Net):
      return obj.make_net_dict_raw()
    return obj

  def make_net_dict_raw(self) -> NetDictRaw:
    """
    Create raw net dict, not containing any :class:`LayerRef` or :class:`Net` instances anymore.
    """
    net_dict = {}
    for key, value in self.name_ctx.children.items():
      if value.layer:
        layer_dict = value.layer.layer_dict
        layer_dict = nest.map_structure(self._map_elem_resolve, layer_dict)
        net_dict[key] = layer_dict
    return net_dict


class NameCtx:
  """
  This is a helper class to keep track of the current name context when creating layers.
  Usually you do not need to access this directly.
  """

  stack = []  # type: List[NameCtx]
  _ReservedNames = {"data", "output"}

  @classmethod
  def top(cls) -> NameCtx:
    """
    Return the top of the stack.
    Assumes that it exists.
    """
    assert cls.stack
    return cls.stack[-1]

  @classmethod
  def current_ctx(cls) -> NameCtx:
    """
    Return the current context.
    This is the top from the stack with is_subnet_ctx.
    """
    top = cls.top()
    if not top.is_subnet_ctx:
      assert top.parent and top.parent.is_subnet_ctx
      return top.parent
    assert top.is_subnet_ctx
    return top

  @classmethod
  def new_root(cls) -> NameCtx:
    """
    Create new root name context
    """
    ctx = NameCtx(parent=None)
    ctx.is_subnet_ctx = True
    return ctx

  def __init__(self, *,
               maker: Optional[ILayerMaker] = None,
               suggested_name: Optional[str] = None,
               name: Optional[str] = None,
               parent: Optional[NameCtx] = NotSpecified):
    self.maker = maker
    self.layer_ref = None  # type: Optional[LayerRef]
    self.layer = None  # type: Optional[Layer]
    self._layer_abs_name_scope = None  # type: Optional[str]
    self.is_subnet_ctx = False
    self.is_repeated_call = False
    self.children = {}  # type: Dict[str, NameCtx]
    self.parent = parent if parent is not NotSpecified else (self.current_ctx() if self.stack else None)
    self.name = name  # early assign such that debug repr works later
    if not name:
      if suggested_name:
        name = self._get_unique_name(suggested_name)
      elif self.parent:
        name = self._get_unique_name()
    self.name = name
    if self.parent:
      self.parent._add_child(self)

  @classmethod
  def get_from_call(cls, *, name: Optional[Union[str, NameCtx]], maker: ILayerMaker) -> NameCtx:
    """
    This is used e.g. for user module or maker calls.
    The name argument can either be a predefined name ctx, or a suggested name.
    """
    if isinstance(name, NameCtx):
      if name.maker is None:
        name.maker = maker
      else:
        assert name.maker is maker
      return name
    assert not name or isinstance(name, str)
    return NameCtx(maker=maker, suggested_name=name)

  def __repr__(self):
    return f"<{self.__class__.__name__} maker:{self.maker} name:{self.get_abs_name_repr()}>"

  @property
  def root(self) -> NameCtx:
    """
    :return: root name ctx
    """
    root = self
    while root.parent:
      root = root.parent
    return root

  @property
  def is_root(self) -> bool:
    """
    :return: whether this is a root ctx
    """
    return not self.parent

  def extend_reserved_names(self, names: Set[str]):
    """
    Extend reserved child names.
    """
    # Do not update inplace because we want an own instance on self.
    self._ReservedNames = self._ReservedNames | names

  def make_net(self) -> Net:
    """
    Create new (sub) net, an instance of :class:`Net`.
    """
    return Net(name_ctx=self)

  def make_default_output(self, ref: LayerRef) -> LayerRef:
    """
    Assume this is a subnet, and make a default output.
    """
    from . import copy
    assert self.is_subnet_ctx
    assert "output" not in self.children
    return copy(ref, name=self.get_child("output"))

  def get_abs_name_ctx_list(self) -> List[NameCtx]:
    """
    Return list [root name ctx, ..., self].
    """
    ls = []
    cur = self
    while cur:
      ls.append(cur)
      cur = cur.parent
    return list(reversed(ls))

  def get_abs_name(self) -> str:
    """
    :return: absolute RETURNN layer name starting from root context.
    """
    ls = self.get_abs_name_ctx_list()
    assert len(ls) >= 2 and not ls[0].name and ls[-1] is self and ls[-1].name
    return "/".join(ctx.name for ctx in ls[1:])

  def get_abs_name_repr(self) -> str:
    """
    :return: Some repr for our absolute name.
    """
    ls = self.get_abs_name_ctx_list()
    if len(ls) == 0:
      debug_name = "???"
    elif len(ls) == 1 and ls[0].name is None:
      debug_name = "/"
    else:
      debug_name = "/".join(repr(ctx.name) if i > 0 or ctx.name is not None else '' for i, ctx in enumerate(ls))
    return debug_name

  @property
  def layer_abs_name_scope(self) -> str:
    """
    :return: layer abs name scope, i.e. the TF name scope of variables
    """
    if self._layer_abs_name_scope is not None:
      return self._layer_abs_name_scope
    assert self.maker
    if self.maker.layer_name_scope is not NotSpecified:
      assert isinstance(self.maker.layer_name_scope, str)
      if self.maker.layer_name_scope == "":
        self._layer_abs_name_scope = self.parent.layer_abs_name_scope
      else:
        parent_prefix = self.parent.layer_abs_name_scope
        if parent_prefix:
          parent_prefix += "/"
        self._layer_abs_name_scope = parent_prefix + self.maker.layer_name_scope
    else:
      self._layer_abs_name_scope = self._get_abs_canonical_name()
    return self._layer_abs_name_scope

  def _get_abs_canonical_name(self, join_str="/") -> str:
    """
    :param str join_str: maybe "." is more common for attrib chains.
      however, we use "/" as default, to make this consistent to :func:`get_abs_name`.
    :return: unique absolute layer name for the maker (module) hierarchy.
      https://github.com/rwth-i6/returnn_common/issues/25
    """
    assert self.maker, f"{self} is not assigned to a maker (module)"
    root = self.root
    root_maker = root.maker  # might be None
    assert root_maker, f"root name ctx {self.root} is not assigned to a maker (module)"
    if root_maker is self.maker:
      return ""  # special case
    # Do a breadth-first search through the parents, starting from self.maker, until we find root_maker.
    queue = [self.maker]
    cache = {}  # maker -> full name
    while queue:
      maker = queue.pop(0)
      postfix = (join_str + cache[maker]) if maker in cache else ""
      for parent, attr in maker.parents_with_attr():
        if parent in cache:
          continue
        for call in parent.calls:
          if call.root is root:  # same name ctx hierarchy
            assert call.is_root or call.layer_abs_name_scope is not None
            if call.is_root or call.layer_abs_name_scope == "":
              return attr + postfix
            assert call.layer_abs_name_scope
            return call.layer_abs_name_scope + join_str + attr + postfix
        cache[parent] = attr + postfix
        queue.append(parent)
      if root_maker in cache:
        break
    if root_maker not in cache:
      err_msgs = []
      for maker, name in cache.items():
        err_msgs.append(f"  {maker}: {name}\n")
      if not err_msgs:
        err_msgs.append(f"  (None, {self.maker} has no parent modules)\n")
      raise Exception(
        f"{self}: no abs canonical name found."
        f" Found partial names:\n{''.join(err_msgs)}"
        f"There must be a path of attribs from the root {root_maker} to {self.maker}.")
    return cache[root_maker]

  def get_name_in_ctx(self, ctx: NameCtx) -> str:
    """
    Get layer name valid in given scope.
    """
    if self.parent is ctx:  # fast path
      return self.name
    ctx_scope_abs = ctx.get_abs_name_ctx_list()
    self_name_abs = self.get_abs_name_ctx_list()
    assert ctx_scope_abs[0] is self_name_abs[0]  # same root
    common_len = 0
    max_common_len = min(len(ctx_scope_abs), len(self_name_abs))
    while common_len < max_common_len and ctx_scope_abs[common_len] is self_name_abs[common_len]:
      common_len += 1
    prefix = "base:" * (len(ctx_scope_abs) - common_len)
    postfix = "/".join([ctx.name for ctx in self_name_abs[common_len:]])
    return prefix + postfix

  def get_name_in_current_ctx(self) -> str:
    """
    Get layer name valid for current scope.
    """
    return self.get_name_in_ctx(ctx=NameCtx.current_ctx())

  def _add_child(self, child: NameCtx):
    assert child.name
    assert child.parent is self
    assert self.is_subnet_ctx
    assert child.name not in self.children
    self.children[child.name] = child

  def get_child(self, name: str) -> NameCtx:
    """
    Makes sure the child exists.
    """
    if name in self.children:
      return self.children[name]
    else:
      return NameCtx(name=name, parent=self)  # also registers in self.children

  def get_child_with_layer_ref(self, name: str) -> NameCtx:
    """
    Makes sure the child exists, including a corresponding layer ref.
    Creates the child together with a layer ref if it does not exist yet.
    """
    child = self.get_child(name)
    if not child.layer_ref:
      layer_ref = LayerRef(name_ctx=child)
      assert child.layer_ref is layer_ref
    return child

  def get_child_layer_ref(self, name: str) -> LayerRef:
    """
    Get child layer ref. Makes sure it exists.
    """
    return self.get_child_with_layer_ref(name).layer_ref

  def __enter__(self):
    self.stack.append(self)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert self.stack[-1] is self, f"{self}.__exit__: stack {self.stack} top is not self"
    self.stack.pop(-1)

  def _get_suggested_name(self) -> str:
    assert self.maker
    reserved_names = set(self.parent.children.keys()) | self._ReservedNames
    if self.parent.maker:
      # Check parent name scope maker, any attrib from there to self.maker.
      # Do a breadth-first search through the parents, starting from self.maker,
      # until we find self.parent.maker.
      queue = [self.maker]
      cache = {}  # parent -> full attrib
      while queue:
        maker = queue.pop(0)
        postfix = f".{cache[maker]}" if maker in cache else ""
        for parent, attr in maker.parents_with_attr():
          if parent in cache:
            if cache[parent] in reserved_names:
              cache[parent] = attr + postfix  # anyway overwrite
            continue
          cache[parent] = attr + postfix
          queue.append(parent)
        if self.parent.maker in cache:
          break
      if self.parent.maker in cache:
        return cache[self.parent.maker]
    # Check parent maker (or module), and use this attrib name.
    # First check if we can find any attr which is not yet reserved.
    for parent, attr in self.maker.parents_with_attr():
      if attr not in reserved_names:
        return attr
    # Now again, to just use any.
    for parent, attr in self.maker.parents_with_attr():
      return attr
    # Check potential previous calls, and reuse their name.
    for call in self.maker.calls:
      if call is self:
        continue  # ignore this
      if call.parent is self.parent:
        return call.name
    # Fallback to the canonical name.
    return self.maker.get_default_name()

  def _get_unique_name(self, suggested_name: Optional[str] = None) -> str:
    name = suggested_name or self._get_suggested_name()
    reserved_names = set(self.parent.children.keys()) | self._ReservedNames
    if self.parent.maker:
      # Also reserve all attrib names of the parent maker.
      # However, we allow to use the name if it is the attrib itself.
      if self.maker and name not in reserved_names and getattr(self.parent.maker, name, None) is self.maker:
        return name
      reserved_names |= set(vars(self.parent.maker).keys())
    if name not in reserved_names:
      return name
    i = 0
    while True:
      name_ = f"{name}_{i}"
      if name_ not in reserved_names:
        return name_
      i += 1


def _move_layer_ref_to_new_name_ctx(*, layer_ref: LayerRef, name_ctx: NameCtx):
  """
  Moves an existing layer ref (with assigned name ctx)
  to another name ctx (without assigned layer or layer ref).

  This assumes that there are no other references to layer_ref.name_ctx
  because those would become invalid.
  References to layer_ref itself should be fine.
  """
  assert not name_ctx.layer and not name_ctx.layer_ref  # none yet assigned

  # Remove layer_ref.name_ctx from its parent name ctx.
  _remove_name_ctx_from_parent(layer_ref.name_ctx)

  # Now reassign.
  layer_ref.name_ctx = name_ctx
  name_ctx.layer_ref = layer_ref
  if isinstance(layer_ref, Layer):
    name_ctx.layer = layer_ref


def _remove_name_ctx_from_parent(name_ctx: NameCtx):
  old_name_ctx = name_ctx.parent.children.pop(name_ctx.name)
  assert old_name_ctx is name_ctx
