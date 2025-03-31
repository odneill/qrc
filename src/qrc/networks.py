"""
Utils for building QRC networks
"""

import functools as ft
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import utils

ReservoirRepr = dict

# -------------------------- Network building utils -------------------------- #


def Q(theta):
  # return utils.rmat(theta) @ np.diag([1, 1j]) @ utils.rmat(-theta)
  # To match perceval
  return (
    utils.rmat(theta) @ np.diag([1, -1j]) @ utils.rmat(-theta) * np.exp(1j * np.pi / 4)
  )


def H(theta):
  # return utils.rmat(theta) @ np.diag([1, -1]) @ utils.rmat(-theta)
  # To match perceval
  return 1j * utils.rmat(theta) @ np.diag([1, -1]) @ utils.rmat(-theta)


def BS(theta):
  # perceval
  return utils.rmat(theta / 2) @ np.diag([1, -1])


def PR(theta):
  return np.diag([1, np.exp(1j * theta)])


def P(theta):
  return np.array([[np.exp(1j * theta)]])


def tri_couplings(m):
  outs = []
  for i in range(0, m - 1):
    if i % 2 == 0:
      outs.append([(2 * j, 2 * j + 1) for j in range(0, i // 2 + 1)])
    else:
      outs.append([(2 * j + 1, 2 * j + 2) for j in range(0, (i + 1) // 2)])
  [outs.append(outs[i]) for i in range(m - 3, -1, -1)]
  outs = [v for o in outs for v in o]
  return outs


def mesh_couplings(m, d):
  outs = []
  for i in range(0, d):
    if i % 2 == 0:
      outs.append([(2 * j, 2 * j + 1) for j in range(0, (m) // 2)])
    else:
      outs.append([(2 * j + 1, 2 * j + 2) for j in range(0, (m + i % 2) // 2 - 1)])
  outs = [v for o in outs for v in o]
  return outs


def layer_couplings(m):
  return list(range(m))


# ------------------------------------- x ------------------------------------ #


register_coupling_handler, get_coupling_handler = utils.new_registry("coupling")


@runtime_checkable
class Coupling(Protocol):
  seed: int
  modes: int
  lossy: bool
  polarised: bool
  num_free_params: int

  def __call__(
    self,
    data=None,
    **kws: Any,
  ): ...

  def validate_params(self, data):
    if data is not None:
      assert data.shape[0] == self.num_free_params, (
        f"Data must have {self.num_free_params} rows"
      )
    else:
      assert self.num_free_params == 0, "Data must be provided"


@register_coupling_handler("qhp_qhp_(theta_2x2)_all_uniform")
class _(Coupling):
  """This should EXACTLY match the original paper"""

  seed: int
  modes: int = 2
  lossy: bool = False
  polarised: bool = True
  num_free_params: int = 0

  def __init__(
    self,
    seed: int,
  ):
    self.seed = seed

  def __call__(self, data=None, **kws):
    self.validate_params(data)
    rng = np.random.default_rng(self.seed)

    q1 = Q(rng.uniform(0, np.pi / 2))
    h1 = H(rng.uniform(0, np.pi / 2))

    q2 = Q(rng.uniform(0, np.pi / 2))
    h2 = H(rng.uniform(0, np.pi / 2))

    p1 = P(rng.uniform(0, np.pi * 2))
    p2 = P(rng.uniform(0, np.pi * 2))

    bs = BS(rng.uniform(0, np.pi * 2))

    ms = [
      utils.embed(q1, 4, [0, 1], [0, 1]),
      utils.embed(h1, 4, [0, 1], [0, 1]),
      utils.embed(p1, 4, [0], [0]),
      utils.embed(p1, 4, [1], [1]),
      #
      utils.embed(q2, 4, [2, 3], [2, 3]),
      utils.embed(h2, 4, [2, 3], [2, 3]),
      utils.embed(p2, 4, [2], [2]),
      utils.embed(p2, 4, [3], [3]),
      #
      utils.embed(bs, 4, [0, 2], [0, 2]),
      utils.embed(bs, 4, [1, 3], [1, 3]),
    ]

    return ms


@register_coupling_handler("qhp_qhp_(theta_2x2)_parameterised")
class _(Coupling):
  """Parameterised pol coupling"""

  seed: int
  modes: int = 2
  lossy: bool = False
  polarised: bool = True
  num_free_params: int = 1

  def __init__(
    self,
    seed: int,
  ):
    self.seed = seed

  def __call__(self, data=None, **kws):
    self.validate_params(data)
    rng = np.random.default_rng(self.seed)

    q1 = Q(rng.uniform(0, np.pi / 2))
    h1 = H(rng.uniform(0, np.pi / 2))

    q2 = Q(rng.uniform(0, np.pi / 2))
    h2 = H(rng.uniform(0, np.pi / 2))

    p1 = P(rng.uniform(0, np.pi * 2))
    p2 = P(rng.uniform(0, np.pi * 2))

    bs = BS(data[0])

    ms = [
      utils.embed(q1, 4, [0, 1], [0, 1]),
      utils.embed(h1, 4, [0, 1], [0, 1]),
      utils.embed(p1, 4, [0], [0]),
      utils.embed(p1, 4, [1], [1]),
      #
      utils.embed(q2, 4, [2, 3], [2, 3]),
      utils.embed(h2, 4, [2, 3], [2, 3]),
      utils.embed(p2, 4, [2], [2]),
      utils.embed(p2, 4, [3], [3]),
      #
      utils.embed(bs, 4, [0, 2], [0, 2]),
      utils.embed(bs, 4, [1, 3], [1, 3]),
    ]

    return ms


@register_coupling_handler("qh_1x1")
class _(Coupling):
  seed: int
  modes: int = 1
  lossy: bool = False
  polarised: bool = True
  num_free_params: int = 2

  def __init__(
    self,
    seed: int = None,
  ):
    """qwp and then hwp, on one spatial mode"""
    pass

  def __call__(self, data=None, **kws):
    self.validate_params(data)
    q1 = Q(data[0])
    h1 = H(data[1])

    ms = [
      utils.embed(q1, 2, [0, 1], [0, 1]),
      utils.embed(h1, 2, [0, 1], [0, 1]),
    ]

    return ms


# ------------------------------------- x ------------------------------------ #

register_reservoir_handler, get_reservoir_handler = utils.new_registry("reservoir")


class Reservoir:
  num_spatial_modes: int
  polarised: bool
  lossy: bool
  num_free_params: int
  _m: int

  def _unitaries(self, data=None): ...

  def get_unitary(self, data=None):
    ms = self._unitaries(data)
    m = ft.reduce(lambda a, b: b @ a, ms, np.eye(self._m))
    if self.lossy:
      m = utils.construct_lossy_unitary(m)
    return m

  def __repr__(self):
    return (
      f"{type(self).__name__}:\n  seed:{self.seed}\n"
      + f"  m:{self.num_spatial_modes}\n  p:{self.polarised}\n  l:{self.lossy}"
    )

  def upproject_matrix(self, matrix, m, p1, p2, c):
    """Takes matrix and projects onto modes, with offset c, and considering
    polarisations"""
    N = len(matrix)
    if p2:
      N //= 2

    if p1 and p2:
      c1 = [v for i in range(N) for v in [2 * (c + i), 2 * (c + i) + 1]]
    elif p1 and not p2:
      c1 = [2 * (c + i) for i in range(N)]
      c2 = [2 * (c + i) + 1 for i in range(N)]
    else:
      c1 = [c + i for i in range(N)]

    u1 = utils.embed(
      matrix,
      m,
      c1,
      c1,
    )
    if p1 and not p2:
      u1 = u1 @ utils.embed(
        matrix,
        m,
        c2,
        c2,
      )
    return u1


def parse_reservoir(
  type: str,
  params: dict | None = None,
) -> Reservoir:
  return get_reservoir_handler(type)(**params)


@register_reservoir_handler("layer")
class LayerReservoir(Reservoir):
  seed: int
  _coupling_fn: str
  _cf: Coupling
  _m: int

  def __init__(
    self,
    seed: int,
    num_spatial_modes: int,
    coupling_fn: str,
    *,
    cf_params: dict = None,
    polarised: bool = False,
    lossy: bool = False,
    name: str | None = None,
  ):
    self.name = name
    self.seed = seed
    self.num_spatial_modes = num_spatial_modes
    self.polarised = polarised
    self.lossy = lossy
    self._coupling_fn = coupling_fn

    self._m = self.num_spatial_modes
    if self.polarised:
      self._m *= 2

    cf_params = cf_params or {}
    self._cf = lambda seed: get_coupling_handler(self._coupling_fn)(seed, **cf_params)
    cf0 = self._cf(0)

    assert cf0.lossy == self.lossy, "Coupling function must match lossy setting"
    assert self.polarised or not cf0.polarised, (
      "Coupling function must match lossy setting"
    )

    self._mesh = layer_couplings(
      self.num_spatial_modes,
    )
    self.num_free_params = len(self._mesh) * cf0.num_free_params

  def _unitaries(self, data=None):
    cf0 = self._cf(0)

    rng = np.random.default_rng(self.seed)
    seeds = rng.integers(0, 2**32, len(self._mesh))

    di = 0
    matrices = []
    for c, s in zip(self._mesh, seeds, strict=True):
      dslice = data[di : di + cf0.num_free_params]
      di += cf0.num_free_params
      us = self._cf(s)(dslice, ports=c)
      us1 = [
        self.upproject_matrix(u, self._m, self.polarised, cf0.polarised, c) for u in us
      ]
      matrices.extend(us1)

    assert di == len(data)

    return matrices


@register_reservoir_handler("mesh")
class MeshReservoir(Reservoir):
  seed: int
  _coupling_fn: str
  _cf: Coupling
  _depth: int
  _m: int

  def __init__(
    self,
    seed: int,
    num_spatial_modes: int,
    depth: int,
    coupling_fn: str,
    *,
    cf_params: dict = None,
    polarised: bool = False,
    lossy: bool = False,
    name: str | None = None,
  ):
    self.name = name
    self.seed = seed
    self.num_spatial_modes = num_spatial_modes
    self.polarised = polarised
    self.lossy = lossy
    self._depth = depth
    self._coupling_fn = coupling_fn

    self._m = self.num_spatial_modes
    if self.polarised:
      self._m *= 2

    cf_params = cf_params or {}
    self._cf = lambda seed: get_coupling_handler(self._coupling_fn)(seed, **cf_params)
    cf0 = self._cf(0)

    assert self.lossy or not cf0.lossy, "Coupling function must match lossy setting"
    assert self.polarised or not cf0.polarised, (
      "Coupling function must match lossy setting"
    )

    self._mesh = mesh_couplings(
      self.num_spatial_modes,
      self._depth,
    )
    self.num_free_params = len(self._mesh) * cf0.num_free_params

  def _unitaries(self, data=None):
    cf0 = self._cf(0)

    rng = np.random.default_rng(self.seed)
    seeds = rng.integers(0, 2**32, len(self._mesh))

    di = 0
    matrices = []
    for c, s in zip(self._mesh, seeds, strict=True):
      dslice = data[di : di + cf0.num_free_params]
      di += cf0.num_free_params
      us = self._cf(s)(dslice, ports=c)
      us1 = [
        self.upproject_matrix(u, self._m, self.polarised, cf0.polarised, c[0])
        for u in us
      ]
      matrices.extend(us1)

    assert di == len(data)

    return matrices


@register_reservoir_handler("tri")
class TriReservoir(Reservoir):
  seed: int
  _coupling_fn: str
  _cf: Coupling
  _m: int

  def __init__(
    self,
    seed: int,
    num_spatial_modes: int,
    coupling_fn: str,
    *,
    cf_params: dict = None,
    polarised: bool = False,
    lossy: bool = False,
    name: str | None = None,
  ):
    self.name = name
    self.seed = seed
    self.num_spatial_modes = num_spatial_modes
    self.polarised = polarised
    self.lossy = lossy
    self._coupling_fn = coupling_fn

    self._m = self.num_spatial_modes
    if self.polarised:
      self._m *= 2

    cf_params = cf_params or {}
    self._cf = lambda seed: get_coupling_handler(self._coupling_fn)(seed, **cf_params)
    cf0 = self._cf(0)

    assert cf0.lossy == self.lossy, "Coupling function must match lossy setting"
    assert self.polarised or not cf0.polarised, (
      "Coupling function must match lossy setting"
    )

    self._mesh = tri_couplings(
      self.num_spatial_modes,
    )
    self.num_free_params = len(self._mesh) * cf0.num_free_params

  def _unitaries(self, data=None):
    cf0 = self._cf(0)

    di = 0
    matrices = []

    seeds = np.arange(len(self._mesh)) + self.seed
    for c, seed in zip(self._mesh, seeds, strict=True):
      dslice = data[di : di + cf0.num_free_params]
      di += cf0.num_free_params
      us = self._cf(seed)(dslice, ports=c)
      us1 = [
        self.upproject_matrix(u, self._m, self.polarised, cf0.polarised, c[0])
        for u in us
      ]
      matrices.extend(us1)

    assert di == len(data)

    return matrices


@register_reservoir_handler("stack")
class StackReservoir(Reservoir):
  _reservoirs: list[Reservoir]
  _offsets: list[int] | None

  def __init__(
    self,
    reservoirs: list[ReservoirRepr | Reservoir],
    offsets: list[int] | None = None,
    *,
    name: str | None = None,
  ):
    self._reservoirs = [
      r if isinstance(r, Reservoir) else parse_reservoir(**r) for r in reservoirs
    ]

    if offsets is None:
      self._offsets = [
        0,
      ] * len(self._reservoirs)
    else:
      assert len(offsets) == len(self._reservoirs)
      self._offsets = offsets

    max_modes = {
      r.num_spatial_modes + o
      for o, r in zip(self._offsets, self._reservoirs, strict=True)
    }
    self.num_spatial_modes = max(max_modes)
    self.lossy = True in {r.lossy for r in self._reservoirs}
    self.polarised = True in {r.polarised for r in self._reservoirs}

    self.num_free_params = sum(r.num_free_params for r in self._reservoirs)
    self.name = name

    self._m = self.num_spatial_modes
    if self.polarised:
      self._m *= 2

  def __repr__(self):
    s = f"{type(self).__name__}: offsets: {self._offsets}, [\n"
    for i, e in enumerate(self._reservoirs):
      lines = str(e).split("\n")
      s += f"{i}: {lines[0]}\n"
      for line in lines[1:]:
        s += f"  {line}\n"
    return s + "]"

  def _unitaries(self, data=None):
    m = self.num_spatial_modes
    if self.polarised:
      m *= 2

    di = 0
    matrices = []
    for r, c in zip(self._reservoirs, self._offsets, strict=True):
      dslice = data[di : di + r.num_free_params]
      di += r.num_free_params
      us = r._unitaries(dslice)
      us1 = [self.upproject_matrix(u, m, self.polarised, r.polarised, c) for u in us]
      matrices.extend(us1)

    assert di == len(data)

    return matrices
