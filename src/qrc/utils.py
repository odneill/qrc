"""
Misc utilities
"""

from typing import Any, Callable

import numpy as np

# -------------------------- Tensor / Mapping Utils -------------------------- #


def tensor_map(f: Callable, *args: list[tuple[Any]], iterate: bool = False, **kwargs):
  """Maps a function over the tensor product of the arguments"""

  _args = list(args)
  _args.extend(kwargs.values())
  ls = [len(a) for a in _args]
  outs = np.ones(ls, dtype=object)
  inds = np.argwhere(outs)
  for i in inds:
    _a = [_args[j][k] for j, k in enumerate(i)]
    a = _a[: len(args)]
    k = dict(zip(kwargs.keys(), _a[len(args) :], strict=True))
    if iterate:
      outs[*i] = f(*a, **k, index=i)
    else:
      outs[*i] = f(*a, **k)
  return outs


def dict_tree_update(d1, d2):
  for k, v in d2.items():
    if isinstance(v, dict):
      d1[k] = dict_tree_update(d1.get(k, {}), v)
    else:
      d1[k] = v
  return d1


# --------------------------------- Unitaries -------------------------------- #


def construct_lossy_unitary(S):
  """
  Following https://arxiv.org/pdf/2108.12160

  Construct a 2M unitary from an M non-unitary
  """

  U, ss, Vt = np.linalg.svd(S)

  assert np.all(abs(ss) <= 1 + 1e-14), "Input matrix must have all SVs <= 1"
  if np.allclose(abs(ss), 1):
    L = np.concatenate(
      [
        np.concatenate([S, 0 * np.eye(len(ss))], axis=0),
        np.concatenate([0 * np.eye(len(ss)), S], axis=0),
      ],
      axis=1,
    )
    return L

  sa = np.sqrt(1 - abs(ss) ** 2)

  A = U @ np.diag(sa) @ Vt

  L = np.concatenate(
    [
      np.concatenate([S, -A], axis=0),
      np.concatenate([A, S], axis=0),
    ],
    axis=1,
  )

  # This is a final safety check, can catch numeric errors
  Lsvd = np.linalg.svd(L)
  L = Lsvd.U @ Lsvd.Vh

  return L


def embed(matrix, m, inmodes, outmodes):
  im = np.array(inmodes)
  om = np.array(outmodes)
  out = np.eye(m, dtype=matrix.dtype)
  out[np.ix_(om, im)] = matrix
  return out


def rmat(t):
  return np.array([
    [np.cos(t), -np.sin(t)],
    [np.sin(t), np.cos(t)],
  ])


# -------------------------------- Registries -------------------------------- #


_registries = {}


def new_registry(name):
  handlers: dict[str, Any] = {}

  def register_decorator(key: str):
    def decorator(fn):
      assert key not in handlers, f"Key {key} already registered"
      handlers[key] = fn
      return fn

    return decorator

  def getter(key):
    return handlers[key]

  _registries[name] = handlers

  return register_decorator, getter


def validate_repr(a, A, parser, optional=False):
  """
  Validates the input and calls the given parser as needed.

  This allows us to pass in either preinstantiated objects, reprs of these, or
  lists of reprs of these.
  """
  if isinstance(a, A):
    return a
  elif isinstance(a, list):
    return [validate_repr(aa, A, parser, optional) for aa in a]
  elif a is not None:
    return parser(**a)
  elif optional:
    return None
  else:
    raise ValueError(f"{A.__name__} must be provided")


# ---------------------------- Confusion Matrices ---------------------------- #


def confusion_matrix(preds, labels, nc):
  """
  Produces an unnormalised confusion matrix, where each row is a true label,
  and each column is a prediction.
  """
  cm = np.zeros([nc, nc])
  inds, vals = np.unique(
    np.stack([labels, preds]),
    axis=1,
    return_counts=True,
  )
  cm[inds[:1, :], inds[1:, :]] = vals
  return cm


def cm_to_pc_acc(cm):
  """Takes confusion matrix, produces per class accuracy."""
  return np.diag(cm) / cm.sum(1)


def cm_to_acc(cm):
  """Takes confusion matrix, produces overall accuracy."""
  return np.diag(cm).sum() / cm.sum()


# ------------------------------ Data Processing ----------------------------- #


def rescale(data, outmin, outmax):
  """Rescales data to [outmin, outmax]"""
  return (data - data.min()) / (data.max() - data.min()) * (outmax - outmin) + outmin


def onehot_encode(labels, classes):
  N = len(labels)
  ys = np.zeros([N, classes], dtype=np.uint8)
  ys[np.arange(N), labels.ravel()] = 1
  return ys


# ------------------------------ Reservoir tools ----------------------------- #


def train_reservoir(train_dat, train_labels, rcond=None):
  """Takes in data in [D, N] for N features, D data samples"""

  jtrain_data = np.array(train_dat)
  jtrain_labels = np.array(train_labels)

  rcond = rcond if rcond is not None else 1e-15
  jm = np.linalg.pinv(jtrain_data.T @ jtrain_data, rcond=rcond) @ jtrain_data.T
  jw = jm @ jtrain_labels

  w = np.array(jw)
  m = np.array(jm)

  return w, m


def eval_reservoir(dat, w):
  """Takes in data in [N x D] for N features, D data samples"""
  fit = dat @ w
  return fit
