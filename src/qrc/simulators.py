from typing import TYPE_CHECKING

import numpy as np
import perceval as pcvl
import scipy.special
from loguru import logger
from tqdm.auto import tqdm

from . import runtime, states, utils

if TYPE_CHECKING:
  from . import experiments as defs

SimulatorRepr = dict


def restructure_split_job_modes(jobs):
  """Array of jobs, where each job is (probs [M_n, B_n], modes[M_n])

  This is much more efficient - key improvements
  working with the basicstates directly, allows dictionary lookup, much faster

  returns arrays probs [M x B], modes [M]

  for B = sum(B_n) and M = cardinality |{M_n, for all n}|"""

  allmodes = {k: None for j in jobs for k in j["modes"]}
  allmodes = {k: i for i, k in enumerate(allmodes.keys())}
  nd = sum(n["probs"].shape[1] for n in jobs)

  probs = np.zeros([len(allmodes), nd])
  i = 0
  for j in jobs:
    inds = np.array(list(map(allmodes.get, j["modes"])))
    probs[inds, i : i + j["probs"].shape[1]] = j["probs"]
    i += j["probs"].shape[1]

  return {"probs": probs, "modes": allmodes}


def filtered_tensorproduct(ss, filter_fn=None):
  """
  Filtered tensor product
  """
  if filter_fn is None:

    def filter_fn(k, v):
      return True

  if len(ss) == 1:
    return ss[0]
  if len(ss) > 2:
    return filtered_tensorproduct(
      [filtered_tensorproduct(ss[:2], filter_fn), *ss[2:]], filter_fn
    )
  else:
    s = pcvl.tensorproduct([
      ss[0],
      ss[1],
    ])

    out = pcvl.StateVector()
    for k, v in s.items():
      if filter_fn(k, v):
        out[k] = v

    return out


# ------------------------------------- x ------------------------------------ #


register_simulator_handler, get_simulator_handler = utils.new_registry("simulator")


class Simulator:
  name: str | None = None

  def __repr__(self):
    return f"{type(self).__name__}: {self.name}"


def parse_simulator(type: str, params: dict) -> Simulator:
  return get_simulator_handler(type)(**params)


def max_photons(state):
  k = []

  def mapper(bs):
    k.append((bs.n, bs.m))
    return bs

  states.map_state(mapper, state)
  k = np.array(k)
  return np.unique(k[:, 0]), np.unique(k[:, 1])


def num_modes(n, m, lower=False):
  k = 0
  if lower:
    for p in range(n + 1):
      k += scipy.special.binom(p + m - 1, m - 1)
    return k
  else:
    return scipy.special.binom(n + m - 1, m - 1)


@register_simulator_handler("SLOS_probs")
class SLOSProbsSimulator(Simulator):
  def __init__(
    self,
    name: str | None = None,
    workers: int = 1,
    split_factor: int = 1,
    **params,
  ):
    self.name = name
    self.workers = workers
    self.split_factor = split_factor

  def run(self, expt: "defs.Experiment"):
    state = expt.state.get_perceval()
    n, m = max_photons(state)
    assert len(m) == 1
    m = expt.reservoir.num_spatial_modes
    n_modes = sum(num_modes(p, m) for p in n)
    est_t = 3e-6 * n_modes
    if est_t > 1:
      est_t = f"{est_t:.2f}s/it"
    else:
      est_t = f"{1 / est_t:.2f}it/s"

    logger.info(f"{expt.name}, {n}, {m}, {n_modes}, {est_t}")

    batched = expt.dataset.inputs
    static = expt
    if self.workers == 0:
      out = slos_runner(([batched], static), True)
      out = [out]
    else:
      out = runtime.batched_multi_parfor(
        [([batched], static)],
        slos_runner,
        workers=self.workers,
        pbars=None,
        split_factor=self.split_factor,
      )[0]
    # don't use postfunc,only one job, so no parallel advantage
    # But has to transfer memory
    out = restructure_split_job_modes(out)
    return out


def slos_runner(a, pbar=False):
  dataset = a[0][0]
  expt = a[1]

  assert isinstance(expt.simulator, SLOSProbsSimulator)

  # Solve for free variables in encoding

  lookup = expt.encoding.solve_params(dataset.shape[1], expt.reservoir.num_free_params)

  state = expt.state.get_perceval()
  enc_data = expt.encoding(dataset[0], lookup)
  unitary = expt.reservoir.get_unitary(enc_data)
  m = len(unitary)
  sm = max_photons(state)[1][0]
  if sm < m:
    d = m - sm
    state = states.map_state(
      lambda x: pcvl.tensorproduct([
        x,
        pcvl.BasicState(
          [
            0,
          ]
          * d
        ),
      ]),
      state,
    )

  probs = []

  if pbar:
    dataset = tqdm(dataset)

  for data in dataset:
    enc_data = expt.encoding(data, lookup)
    unitary = expt.reservoir.get_unitary(enc_data)

    _src = pcvl.components.source.Source(1, 0, 1, 0, "indistinguishable")
    c = pcvl.Processor("SLOS", len(unitary), _src)
    wrapped_unitary = pcvl.components.Unitary(pcvl.utils.Matrix(unitary))
    c.add(0, wrapped_unitary)
    c.with_input(state)
    probs.append(c.probs())

  modes = list({v for p in probs for v in p["results"].keys()})
  ps = np.array([[p["results"].get(m, 0) for p in probs] for m in modes])

  return {"probs": ps, "modes": modes}


@register_simulator_handler("Efficient_quantum_coherent")
class EfficientCoherentSimulator(Simulator):
  def __init__(
    self,
    name: str | None = None,
    workers: int = 1,
    split_factor: int = 1,
    min_prob: float = None,
  ):
    self.name = name
    self.workers = workers
    self.split_factor = split_factor
    self.min_prob = min_prob

  def run(self, expt: "defs.Experiment"):
    assert isinstance(expt.simulator, EfficientCoherentSimulator)
    dataset = expt.dataset.inputs

    static = expt
    if self.workers == 0:
      out = eff_coh_runner(([dataset], static), True)
      out = [out]
    else:
      out = runtime.batched_multi_parfor(
        [([dataset], static)],
        eff_coh_runner,
        workers=self.workers,
        pbars=None,
        split_factor=self.split_factor,
      )[0]
    # don't use postfunc,only one job, so no parallel advantage
    # But has to transfer memory
    out = restructure_split_job_modes(out)
    return out


def eff_coh_runner(a, pbar=False):
  dataset = a[0][0]
  expt = a[1]

  assert isinstance(expt.simulator, EfficientCoherentSimulator)

  # Solve for free variables in encoding
  lookup = expt.encoding.solve_params(dataset.shape[1], expt.reservoir.num_free_params)
  csclassical = expt.state.alphas

  truncation = None
  s = expt.state
  # unwrap the state
  while not truncation:
    if hasattr(s, "truncation"):
      truncation = s.truncation
    else:
      s = s.state

  ntot = truncation * (np.abs(csclassical) > 0).sum()

  m = len(csclassical)
  classical_alphas = np.zeros([m, dataset.shape[0]], dtype=complex)

  if pbar:
    dataset = tqdm(dataset)
  for i, data in enumerate(dataset):
    enc_data = expt.encoding(data, lookup)
    unitary = expt.reservoir.get_unitary(enc_data)
    classical_alphas[:, i] = unitary @ csclassical

  min_p = expt.simulator.min_prob
  if min_p is None:
    min_p = pcvl.utils.global_params["min_p"]

  # iterate over modes, get each coherent state (as func of data)
  # Then tensor all modes together
  # We filter for only modes with total occupation <= truncation
  classical_state = filtered_tensorproduct(
    [states.coherent_state(a, ntot) for a in classical_alphas],
    filter_fn=lambda k, v: k.n <= ntot and np.any(np.abs(v) ** 2 >= min_p),
  )
  classical_ampls = np.array(list(classical_state.values()))
  classical_probs = np.abs(classical_ampls) ** 2
  modes = list(classical_state.keys())

  return {"probs": classical_probs, "modes": modes, "amplitudes": classical_ampls}


@register_simulator_handler("classical")
class ClassicalSimulator(Simulator):
  def __init__(
    self,
    name: str | None = None,
    workers: int = 0,
    split_factor: int = 1,
    **params,
  ):
    self.name = name
    self.workers = workers
    self.split_factor = split_factor

  def run(self, expt: "defs.Experiment"):
    assert isinstance(expt.simulator, ClassicalSimulator)
    dataset = expt.dataset.inputs

    static = expt
    if self.workers == 0:
      out = classical_runner(([dataset], static), True)
    else:
      out = runtime.batched_multi_parfor(
        [([dataset], static)],
        classical_runner,
        workers=self.workers,
        pbars=None,
        split_factor=self.split_factor,
      )[0]

    out = {"amplitudes": np.concatenate([v["amplitudes"] for v in out], axis=-1)}

    return out


def classical_runner(a, pbar=False):
  dataset = a[0][0]
  expt = a[1]

  assert isinstance(expt.simulator, ClassicalSimulator)

  # Solve for free variables in encoding
  lookup = expt.encoding.solve_params(dataset.shape[1], expt.reservoir.num_free_params)
  state = np.array(expt.state.alphas)

  m = len(state)
  amplitudes = np.zeros([m, dataset.shape[0]], dtype=complex)

  if pbar:
    dataset = tqdm(dataset)
  for i, data in enumerate(dataset):
    enc_data = expt.encoding(data, lookup)
    unitary = expt.reservoir.get_unitary(enc_data)
    amplitudes[:, i] = unitary @ state

  return {"amplitudes": amplitudes}
