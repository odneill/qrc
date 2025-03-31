"""
Quantum states and state conversions
"""

import numpy as np
import perceval as pcvl
import scipy
from exqalibur import FockState

from . import utils

BasicState = FockState

StateRepr = dict

# ----------------------------- State generation ----------------------------- #


def coherent_state_amplitudes(alpha, trunc=10):
  n = np.arange(trunc + 1)[:, None]
  flag = False
  if isinstance(alpha, np.ndarray):
    alpha = alpha[None, :]
  else:
    flag = True
    alpha = np.array([[alpha]])
  coefs = (
    np.exp(-0.5 * abs(alpha) ** 2) * alpha**n / np.sqrt(scipy.special.factorial(n))
  )

  if flag:
    coefs = coefs.ravel()
  return coefs


def coherent_state(alpha, trunc=10, dist=False):
  coefs = coherent_state_amplitudes(alpha, trunc)
  s = pcvl.StateVector()
  for n, a in enumerate(coefs):
    s[pcvl.BasicState(f"|{n}>")] = a
  return s


def map_state(
  fn,
  state: BasicState | pcvl.StateVector | pcvl.SVDistribution,
) -> BasicState | pcvl.StateVector | pcvl.SVDistribution:
  if isinstance(state, pcvl.BasicState):
    return fn(state)
  elif isinstance(state, pcvl.StateVector):
    sv = pcvl.StateVector()
    for a, b in state.items():
      sv[fn(a)] = b
    return sv
  elif isinstance(state, pcvl.SVDistribution):
    return pcvl.SVDistribution({map_state(fn, sv): prob for sv, prob in state.items()})


# ------------------------------------- x ------------------------------------ #


register_state_handler, get_state_handler = utils.new_registry("state")


class State:
  _pol: bool = False
  _modes: int
  _lossy: bool = False
  _dist: bool = False
  _state: BasicState | pcvl.StateVector | pcvl.SVDistribution
  name: str | None = None

  def __repr__(self):
    return f"{type(self).__name__}: {self._state}"

  def get_perceval(self):
    assert self._state is not None, "State must be initialised"
    if not self._lossy:
      return self._state

    def loss_modes(s: BasicState):
      r = str(s)[1:-1].split(",")
      o = "|"
      for c in r:
        o += c + ","
      for _ in r:
        o += "0,"
      o = o[:-1] + ">"
      return pcvl.BasicState(o)

    return map_state(loss_modes, self._state)


def parse_state(type: str, params: dict) -> State:
  return get_state_handler(type)(**params)


@register_state_handler("photon_added_coherent")
class PACState(State):
  """"""

  alphas: list[float]
  adds: list[int]
  truncation: int

  def __init__(
    self,
    alphas: list[float],
    adds: list[int],
    truncation: int,
    *,
    name: str | None = "pac",
  ):
    self.name = name
    self.alphas = alphas
    self.adds = adds
    self.truncation = truncation
    alphas = np.array(alphas)
    truncation: int = int(truncation)

    csses = []
    for a, b in zip(alphas, adds, strict=True):
      ns = np.arange(0, truncation + 1)
      cscoefs = coherent_state_amplitudes(a, truncation)
      paccoefs = cscoefs * np.sqrt(
        scipy.special.factorial(ns + b) / scipy.special.factorial(ns)
      )
      norm = np.sqrt(np.sum(np.abs(paccoefs) ** 2))
      paccoefs /= norm
      pac = {pcvl.BasicState(f"|{n + b}>"): paccoefs[n] for n in ns}
      pacsv = pcvl.StateVector()
      for k, v in pac.items():
        pacsv[k] = v
      csses.append(pacsv)

    self._state = pcvl.tensorproduct(csses)
    self._modes = len(alphas)


@register_state_handler("polarised")
class PolarisedState(State):
  state: State
  axis: int
  _pol: bool = True

  def __init__(
    self,
    state: StateRepr | State,
    axis: int,
    *,
    name: str | None = None,
  ):
    self.axis = axis
    assert axis in (0, 1)
    self.state = state if isinstance(state, State) else parse_state(**state)
    self.name = name if name is not None else self.state.name

    assert not self.state._pol, "Cannot polarise a polarised state"

    def polarise(s: BasicState):
      r = str(s)[1:-1].split(",")
      o = "|"
      for i, c in enumerate(r):
        if axis == 0:
          o += f"{c},0,"
        else:
          o += f"0,{c},"
      o = o[:-1] + ">"

      return pcvl.BasicState(o)

    if hasattr(self.state, "_state"):  # If perceval state
      self._state = map_state(polarise, self.state._state)
    if hasattr(self.state, "alphas"):  # coherent state
      if axis == 0:
        self.alphas = [v for a in self.state.alphas for v in (a, 0)]
      else:
        self.alphas = [v for a in self.state.alphas for v in (0, a)]
    self._modes = 2 * self.state._modes


@register_state_handler("fock")
class FockState(State):
  occupancies: list[int]

  def __init__(
    self,
    occupancies: list[int],
    *,
    name: str | None = "fock",
  ):
    self.name = name
    self.occupancies = occupancies
    s = np.array(occupancies)
    self._state = pcvl.StateVector(s)
    self._modes = len(occupancies)


@register_state_handler("dist")
class DistState(State):
  labels: str

  def __init__(
    self,
    labels: str,
    *,
    name: str | None = "dist",
  ):
    self.name = name
    self.labels = labels
    self._dist = True
    self._state = pcvl.StateVector("|" + labels + ">")
    self._modes = len(labels.split(","))


@register_state_handler("coherent")
class CoherentState(State):
  alphas: list[float]
  truncation: int

  def __init__(
    self,
    alphas: list[float],
    truncation: int,
    *,
    name: str | None = "coherent",
  ):
    self.name = name
    self.alphas = alphas
    self.truncation = truncation
    alphas = np.array(alphas)
    truncation: int = int(truncation)

    csses = [coherent_state(a, truncation) for a in alphas]

    self._state = pcvl.tensorproduct(csses)
    self._modes = len(alphas)


@register_state_handler("classical_coherent")
class ClassicalCoherentState(State):
  alphas: list[float]

  def __init__(
    self,
    alphas: list[float],
    *,
    name: str | None = "classical_coherent",
  ):
    self.name = name
    self.alphas = alphas
    alphas = np.array(alphas)
    self._modes = len(alphas)
