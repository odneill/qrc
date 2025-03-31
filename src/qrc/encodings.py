"""
Implements encoding functions for the QRC, to be used in building networks
"""

import numpy as np
import sympy as sp

from . import utils

EncodingRepr = dict


def _general_encoding_raw(x, N, K, F, G, port=0):
  """periodic in x in [-1,1]"""
  x = np.mod(x + 1, 2) - 1
  N = N * (K * port + 1)
  sx = G * (x > 0).astype(float)
  q1 = F * (1 + 2 * x - 4 * x * sx)
  h1 = F * sx + 2 * F * N * x * (sx - 1) + x * N
  return np.array([q1, h1]) * np.pi / 4


# ------------------------------------- x ------------------------------------ #

register_encoding_handler, get_encoding_handler = utils.new_registry("encoding")


class Encoding:
  """
  Takes a data vector to a reservoir compatible input
  Reservoir provides the number of inputs and meta-data about those inputs,
  which encodings can use.
  """

  name: str = ""

  def __repr__(self):
    return f"{type(self).__name__}: {self.name}"

  def __call__(self, data, lookup: dict[sp.Symbol, int]): ...

  def forward_num(self, num_in):
    return None

  def backward_num(self, num_out):
    return None

  def solve_params(self, nin, nout):
    expr = self.forward_num(nin)
    if isinstance(expr, sp.Expr):
      assert len(expr.free_symbols) == 1
      vals = sp.solve(expr - nout, expr.free_symbols)
      lookup = dict(zip(expr.free_symbols, vals, strict=True))
    else:
      lookup = {}
      assert expr == nout, "Encoding must match number of free parameters in reservoir"
    return lookup


def parse_encoding(type: str, params: dict) -> Encoding:
  return get_encoding_handler(type)(**params)


@register_encoding_handler("linear_one_to_many")
class _(Encoding):
  name: str = "linear_one_to_many"

  def __init__(
    self,
    *,
    name: str | None = None,
  ):
    self.name = name

    self._N = sp.symbols("n" + str(id(self))[:4], real=True)

  def __call__(self, data, lookup: dict[sp.Symbol, int]):
    assert data.size == 1, "Input must be a single value"
    N = lookup[self._N]
    out = np.repeat(data, N, axis=0)
    return out

  def forward_num(self, num_in):
    return num_in * self._N

  def backward_num(self, num_out):
    return num_out / self._N


@register_encoding_handler("n_pad_end")
class _(Encoding):
  """
  Pad the end of the data vector with `value` until shape matches the
  reservoir free params.
  """

  name: str = "n_pad_end"
  value: float

  def __init__(
    self,
    *,
    name: str | None = None,
    value: float = 0.0,
  ):
    self.name = name
    self.value = value

    self._N = sp.symbols("n" + str(id(self))[:4], real=True)

  def __call__(self, data, lookup: dict[sp.Symbol, int]):
    N = lookup[self._N]
    out = np.concatenate([data, self.value * np.ones(N)], axis=0)
    return out

  def forward_num(self, num_in):
    return num_in + self._N

  def backward_num(self, num_out):
    return num_out - self._N


@register_encoding_handler("linear_match_ranges")
class _(Encoding):
  """linearly scale input from range 1 to match range 2"""

  name: str = "n_pad_end"

  def __init__(
    self,
    *,
    name: str | None = None,
    in_min: float = 0.0,
    in_max: float = 1.0,
    out_min: float = 0.0,
    out_max: float = 1.0,
  ):
    self.name = name
    self.range = [in_min, in_max, out_min, out_max]

  def __call__(self, data, lookup: dict[sp.Symbol, int]):
    data -= self.range[0]
    data *= (self.range[3] - self.range[2]) / (self.range[1] - self.range[0])
    data += self.range[2]
    return data

  def forward_num(self, num_in):
    return num_in

  def backward_num(self, num_out):
    return num_out


@register_encoding_handler("compose")
class _(Encoding):
  name: str = "compose"

  def __init__(
    self,
    encodings: list[EncodingRepr | Encoding] = (),
    *,
    name: str | None = None,
  ):
    self._encodings = [
      e if isinstance(e, Encoding) else parse_encoding(**e) for e in encodings
    ]

    self.name = name if name is not None else self._encodings[-1].name

  def __call__(self, data, lookup: dict[sp.Symbol, int]):
    out = data
    for e in self._encodings:
      out = e(out, lookup)
    return out

  def forward_num(self, num_in):
    k = num_in
    for e in self._encodings:
      k = e.forward_num(k)
    return k

  def backward_num(self, num_out):
    k = num_out
    for e in self._encodings[::-1]:
      k = e.backward_num(k)
    return k


@register_encoding_handler("general_pol_encoding")
class _(Encoding):
  """
  assumes the couplings we feed into each have 2 params, Q and then H angles

  """

  num_in: int = 1
  name: str = "general_pol_encoding"

  def __init__(
    self,
    seed: int,
    N: float = 2,
    F: float = 1,
    K: float = 0,
    G: float = 0,
    P: float = 0,
    num_in: int = None,
    num_out: int = None,
    *,
    name: str | None = None,
  ):
    """Periodic -1, 1

    N : number of periods, valid [-1, inf)
    K : Port proportionality
    F : Flag to include circularity, valid in {0,1}. F=0 corresponds to states
    only on the equator i.e. lin pol.
    G : Flag to determine the direction of the spiral over the second half of
    the trajectory, valid in {0,1}.
    P : Phase mutliplier of the encoding
    """

    self.name = name
    self.num_out = num_out
    self.num_in = num_in

    self.seed = seed
    self.params = {"N": N, "F": F, "K": K, "G": G, "P": P}

  def __call__(self, data, lookup: dict[sp.Symbol, int]):
    N = len(data)
    phases = np.random.default_rng(self.seed).uniform(0, 1, N)

    x = data + phases

    out = []
    for i, xv in enumerate(x):
      out.append(
        _general_encoding_raw(
          xv,
          N=self.params["N"],
          K=self.params["K"],
          F=self.params["F"],
          G=self.params["G"],
          port=i,
        )
      )

    out = np.concatenate(out)

    return out

  def forward_num(self, num_in):
    return 2 * num_in

  def backward_num(self, num_out):
    return num_out // 2
