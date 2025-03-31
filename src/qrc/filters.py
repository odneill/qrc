"""
Anything to do with filtering output, postselection, distribution transforms
(i.e. noise modeling)
"""

import numpy as np
import perceval as pcvl
import scipy
from loguru import logger


def get_noise_dist(
  n_max,
  expected_darkcounts=100 * 50e-9,
  detector_efficiency=0.86,
  PTHRESH=1e-20,
):
  """Gets the detector noise distribution for a given number of modes

  n_max should be the maximum expected input photons in a single mode

  returns a [p,q] matrix, where p is the maximum number of expected photons in
  a mode, and q is the maximum expected output photons in a mode.
  """
  assert expected_darkcounts < 10, "Too many darkcounts"
  noise_dist = scipy.stats.poisson.pmf(np.arange(100), expected_darkcounts)
  maxind = np.argwhere(noise_dist > PTHRESH).max() + 1
  noise_dist = noise_dist[:maxind]

  expected_ms = np.arange(n_max + 1)
  detection_dist = scipy.stats.binom.pmf(
    expected_ms[None, :], expected_ms[:, None], detector_efficiency
  )

  full_dist = scipy.signal.convolve(noise_dist[None, :], detection_dist)
  full_dist = full_dist[:, full_dist.sum(0) >= PTHRESH]

  return full_dist


def convolve_mode(modes, probs, conv_dist, mode=0, PTHRESH=1e-10):
  """Given a set of modes [M, m] for M events, m modes

  and probabilities [M, D] for D distributions.

  Convolve the mode indexed `mode` with the convolutional distribution
  `conv_dist`, and return the new modes and probabilities

  To perform for every mode, should sequentially call this with mode = 0, 1,
  2, ... m-1.

  """
  ps_m = conv_dist[modes[:, mode]]

  ind = ps_m >= PTHRESH
  # keys is a list of pairs. first is the index of a row in modes/ probs.
  # Second is the new photon count we map to
  keys = np.argwhere(ind)
  newprobs = ps_m[ind]
  newprobs = newprobs[:, None] * probs[keys[:, 0]]

  newmodes = np.concatenate(
    [
      modes[keys[:, 0]][:, :mode],
      keys[:, 1][:, None],
      modes[keys[:, 0]][:, mode + 1 :],
    ],
    axis=1,
  )

  unique_modes, inds = np.unique(newmodes, axis=0, return_inverse=True)

  finalprobs = np.zeros((len(unique_modes), newprobs.shape[1]), dtype=newprobs.dtype)
  np.add.at(finalprobs, inds, newprobs)

  return unique_modes, finalprobs


def parse_state(state):
  if isinstance(state, pcvl.StateVector):
    inprobs = np.array(list(state.values()))
    inmodes = np.array(list(state.keys()))
  else:
    inprobs, inmodes = state

  if len(inprobs.shape) == 1:
    inprobs = inprobs[:, None]

  return inmodes, inprobs


def noise_model_v3(state: pcvl.StateVector, conv_dist, PTHRESH=1e-20):
  """
  Much more efficient noise model.

  Assumes the conv_dist is the same for all modes, and that each mode is independent.
  This allows us to convolve each mode independently, and then combine the results.
  """
  inmodes, inprobs = parse_state(state)
  outmodes = inmodes
  outprobs = inprobs
  for i in range(inmodes.shape[-1]):
    outmodes, outprobs = convolve_mode(
      outmodes, outprobs, conv_dist, mode=i, PTHRESH=PTHRESH
    )

  return outmodes, outprobs


def noise_model(
  state: pcvl.StateVector,
  expected_darkcounts=100 * 50e-9,
  detector_efficiency=0.86,
  PTHRESH=1e-20,
):
  inmodes, inprobs = parse_state(state)

  conv_dist = get_noise_dist(
    inmodes.max(),
    expected_darkcounts=expected_darkcounts,
    detector_efficiency=detector_efficiency,
    PTHRESH=PTHRESH,
  )

  return noise_model_v3(state, conv_dist, PTHRESH=PTHRESH)


def sample_dist(dists, *, N=0, f=1000, seeds=None):
  # return sample_dist_v1(dists, N=N, f=f, seeds=seeds)
  return sample_dist_v2(dists, N=N, f=f, seeds=seeds)


def sample_dist_v2(dists, *, N=0, f=1000, seeds=None):
  """
  Given a series of X distributions with in an array with shape [M, X], sample
  over the M modes N times, returning a distribution with shape [M, X]

  If f is not None then N = f * M + 1
  """
  M, X = dists.shape
  if f is not None:
    N = int(f * M) + 1

  if seeds is None:
    seeds = [None] * X

  samples = np.zeros((M, X), dtype=np.float64)

  for i, (dist, seed) in enumerate(zip(dists.T, seeds, strict=True)):
    rng = np.random.default_rng(seed)
    sample_dist = rng.multinomial(N, dist) / N
    samples[:, i] = sample_dist

  return samples


def sample_dist_v1(dists, *, N=0, f=1000, seeds=None):
  """
  Deprecated, use v2 for speed
  Maintained for backwards compatibility
  """
  M, X = dists.shape
  if f is not None:
    N = int(f * M) + 1

  if seeds is None:
    seeds = [None] * X

  samples = np.zeros((M, X), dtype=np.float64)

  for i, (dist, seed) in enumerate(zip(dists.T, seeds, strict=True)):
    rng = np.random.default_rng(seed)
    choices = rng.choice(M, N, p=dist)
    sample_dist, _ = np.histogram(choices, np.arange(M + 1)) / N
    samples[:, i] = sample_dist

  return samples


# --------------------------- Postselection filters -------------------------- #


def specific_range_filter(modes, lower=1, upper=2):
  mask = (modes.sum(1)[..., None] >= lower) * (modes.sum(1)[..., None] <= upper)
  total = modes * mask + (0 * modes - 1) * (1 - mask)
  unique, inds = np.unique(total, axis=0, return_inverse=True)
  return unique, inds


def pol_filter(modes):
  total = modes.reshape(len(modes), -1, 2).sum(-1)
  unique, inds = np.unique(total, axis=0, return_inverse=True)
  return unique, inds


def postselect(probs, modes, filter):
  """filter should preserve probability. Modes which are ignored should be
  mapped to the dummmy mode -1, equal to `outmodes[0]*0 - 1`"""

  flag = False
  if len(probs.shape) == 1:
    flag = True
    probs = probs[:, None]
  outputmodes, inds = filter(modes)

  outprobs = np.zeros((len(outputmodes), probs.shape[1]), dtype=probs.dtype)
  np.add.at(outprobs, inds, probs)

  if flag:
    outprobs = outprobs[:, 0]

  return outputmodes, outprobs


def filter_dummy(probs, modes):
  inds = np.logical_not(np.prod(modes == (modes[0] * 0 - 1), axis=-1))
  outprobs = probs[inds]
  outmodes = modes[inds]
  return (outprobs, outmodes), probs[np.logical_not(inds)]


def dummy_renormalisation(probs, modes, *, PTHRESH=1e-20):
  norm = probs.sum(0)
  delta = 1 - norm

  if np.any(delta > PTHRESH):
    logger.error(
      f"Probabilities do not sum to 1, within threshold: {delta.max()} > {PTHRESH}"
    )

  if np.any(delta > 0):
    inds = np.prod(modes == (modes[0] * 0 - 1), axis=-1).astype(bool)
    if not np.any(inds):
      outmodes = np.concatenate([modes, (modes[0] * 0 - 1)[None]], axis=0)
      outprobs = np.concatenate([probs, delta[None]], axis=0)
    else:
      assert sum(inds) == 1, f"Only one dummy mode supported: {sum(inds)}"
      outprobs = probs.copy()
      outprobs[inds] += delta
      outmodes = modes.copy()

  return (outprobs, outmodes), norm


def renormalisation(probs, modes, *, PTHRESH=1e-20):
  outprobs = probs.copy()
  outmodes = modes.copy()

  norm = outprobs.sum(0)
  if np.any(1 - norm > PTHRESH):
    logger.error(
      f"Probabilities do not sum to 1, within threshold: {np.max(1 - norm)} > {PTHRESH}"
    )
  outprobs /= norm

  return (outprobs, outmodes), norm
