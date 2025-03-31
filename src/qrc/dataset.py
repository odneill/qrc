import functools as ft
from enum import Enum

import numpy as np

from . import utils
from ._datasets import medmnist, mnist

DSPostFilterRepr = dict
TaskRepr = dict
DatasetRepr = dict

# ----------------------------------- Tasks ---------------------------------- #

register_task_handler, get_task_handler = utils.new_registry("task")


class Task:
  name: str

  def __init__(self, **params):
    self.params = params

  def domain(self, x, y=None) -> bool:
    """filter function on dataset elements. Can be applied to batches

    Optionally takes labels too

    Defaults to allow all values

    Shapes:
    [N,d]: float
    N: batch
    d: features
    returns:
    [N]: bool
    """
    return np.ones(x.shape[0], dtype=bool)

  def __call__(self, x, y=None):
    """Map inputs to outputs, ground truth model"""

  def metrics(self, x, y_pred, y_true):
    return {"MSE": (abs(y_pred - y_true) ** 2).mean()}


def parse_task(type: str, params: dict) -> Task:
  return get_task_handler(type)(**params)


class Task01(Task):
  def domain(self, x, y=None):
    return np.all(np.logical_and(x <= 1, x >= 0), axis=1)


@register_task_handler("sinc")
class SincTask(Task01):
  name = "sinc"

  def __call__(self, x, y=None):
    return np.sinc(15 * (x - 0.75))


@register_task_handler("tanh")
class TanhTask(Task01):
  name = "tanh"

  def __call__(self, x, y=None):
    return np.tanh(10 * (x - 0.5))


@register_task_handler("rect")
class RectTask(Task01):
  name = "rect"

  def __call__(self, x, y=None):
    return (np.array(x) > 0.5).astype(float)


@register_task_handler("random_freq")
class RandTask(Task01):
  """
  Generates functions on the domain [0,1] which are ~reasonable~.
  Lots of magic numbers

  Parameters
  ----------
  P : int, optional
    The number of functions to generate. Default is 16.
  seed : int, optional
    Seed for the random number generator. Default is None.
  ret_freqs : bool, optional
    Whether to return the frequencies, amplitudes and phases of the generated functions. Default is False.

  Returns
  -------
  tuple
    A tuple containing the generated functions and optionally the frequencies, amplitudes and phases of the functions.
    If ret_freqs is False, the tuple contains two elements:
      - An np.ndarray representing the domain x.
      - A list of np.ndarray representing the generated functions.
    If ret_freqs is True, the tuple contains three elements:
      - An np.ndarray representing the domain x.
      - A list of np.ndarray representing the generated functions.
      - A list of tuples, where each tuple contains three np.ndarray representing the frequencies, amplitudes and phases of a function.

  Notes
  -----

  The implementation of these random functions is weird - it is designed to maintain backward compatibility with the old random function generation, which was in `qrc.eval`

  """

  def __init__(self, index, seed):
    self.index = index
    self.seed = seed
    self.name = f"random_{index}"

    # For backward compat, do not change RNG
    indiv_seed = np.random.default_rng(self.seed).integers(0, 2**32, index + 1)[index]

    M = 50
    ml = 0.5
    rng = np.random.default_rng(indiv_seed)
    fu = abs(rng.normal(0, 60)) + ml
    denom = abs(rng.normal(3, 15))

    frequencies = np.linspace(ml, fu, M)
    dk = frequencies[1] - frequencies[0]
    amplitudes = rng.uniform(0, 1, M) * np.exp(-frequencies / denom)
    self._amplitudes = amplitudes / np.sqrt(dk * np.sum(amplitudes**2))
    self._phases = rng.uniform(0, 2 * np.pi, M)
    self._frequencies = frequencies

  def __call__(self, x, y=None):
    values = tuple(zip(self._amplitudes, self._frequencies, self._phases, strict=True))
    return sum(a * np.sin(f * x * np.pi + p) for a, f, p in values)


@register_task_handler("classify_one_hot")
class ClassifyTask(Task):
  name = "classify"

  def __call__(self, x, y):
    return y

  def metrics(self, x, y_pred, y_true):
    nc = y_true.shape[1]
    conf_matrix = utils.confusion_matrix(
      np.argmax(y_pred, axis=1), np.argmax(y_true, axis=1), nc
    )
    return {
      "confmat": conf_matrix,
      "pc_accuracy": utils.cm_to_pc_acc(conf_matrix),
      "accuracy": utils.cm_to_acc(conf_matrix),
    }


# ------------------------------- Data loaders ------------------------------- #

register_dataset_generator_handler, get_dataset_generator_handler = utils.new_registry(
  "dataset_generator"
)


@register_dataset_generator_handler("1xn_linspace_n1-p1_generator")
def dataset_linspace(
  split,
  seed,
  n: int = 500,
  start: float = -1,
  end: float = 1,
  endpoint: bool = True,
):
  inputs = np.linspace(start, end, n, endpoint=endpoint)[:, None]

  out = {
    "inputs": inputs,
    "labels": None,
    "splits_fn": _get_random_splitsfn(split, seed),
    "meta": {},
  }
  return out


@register_dataset_generator_handler("mnist_v1")
def dataset_mnist(**kwargs):
  """Version 1 MNIST dataset

  `alg` is deprecated argument maintained for compatibility
  """
  ds = mnist.mnist()

  data = {
    "inputs": np.concatenate(
      [
        ds["train"]["images"],
        ds["test"]["images"],
      ],
      axis=0,
    ),
    "labels": np.concatenate(
      [
        ds["train"]["labels"],
        ds["test"]["labels"],
      ],
      axis=0,
    ),
    "splits_fn": _get_sequential_splitsfn(len(ds["train"]["images"])),
    "meta": {
      "numclasses": 10,
    },
  }

  return data


@register_dataset_generator_handler("breast_v1")
def _(**kwargs):
  """Version 1 BreastMNIST dataset"""
  data = _medmnist_image_classification("BreastMNIST", **kwargs)
  return data


@register_dataset_generator_handler("blood_v1")
def _(**kwargs):
  """Version 1 BloodMNIST dataset"""
  data = _medmnist_image_classification("BloodMNIST", **kwargs)
  return data


@register_dataset_generator_handler("derma_v1")
def _(**kwargs):
  """Version 1 DermaMNIST dataset"""
  data = _medmnist_image_classification("DermaMNIST", **kwargs)
  return data


@register_dataset_generator_handler("path_v1")
def _(**kwargs):
  """Version 1 PathMNIST dataset"""
  data = _medmnist_image_classification("PathMNIST", **kwargs)
  return data


@register_dataset_generator_handler("organc_v1")
def _(**kwargs):
  """Version 1 OrganCMNIST dataset"""
  data = _medmnist_image_classification("OrganCMNIST", **kwargs)
  return data


@register_dataset_generator_handler("oct_v1")
def _(**kwargs):
  """Version 1 OCTMNIST dataset"""
  data = _medmnist_image_classification("OCTMNIST", **kwargs)
  return data


def _medmnist_image_classification(key: str, **kwargs):
  """Prepare image datasets in the MedMNIST collection"""

  ds_train = getattr(medmnist, key)(split="train")
  ds_test = getattr(medmnist, key)(split="test")

  data = {
    "inputs": np.concatenate(
      [
        ds_train.imgs,
        ds_test.imgs,
      ],
      axis=0,
    ),
    "labels": np.concatenate(
      [
        ds_train.labels,
        ds_test.labels,
      ],
      axis=0,
    ),
    "splits_fn": _get_sequential_splitsfn(len(ds_train.imgs)),
    "meta": {
      "numclasses": len(ds_train.info["label"]),
    },
  }

  return data


# ---------------------------- Dataset postfilter ---------------------------- #

register_dspf_handler, get_dspf_handler = utils.new_registry("dspf")


class DSPostFilter:
  def __call__(self, ds):
    """Perform transformation on dataset"""


def parse_dspf(type: str, params: dict) -> DSPostFilter:
  return get_dspf_handler(type)(**params)


@register_dspf_handler("classification")
class ClassifyDSPF(DSPostFilter):
  """prep image dataset for classification"""

  def __call__(self, ds):
    labels = utils.onehot_encode(ds["labels"], ds["meta"]["numclasses"])

    out = {
      "inputs": ds["inputs"],
      "labels": labels,
      "splits_fn": ds["splits_fn"],
      "meta": ds["meta"],
    }

    return out


class _SubsampleStrategy(Enum):
  RANDOM = "random"
  FIRST = "first"
  LAST = "last"


@register_dspf_handler("subsample")
class SubsampleDSPF(DSPostFilter):
  """Subsample the total dataset"""

  def __init__(self, n_test, n_train, strategy, seed=None):
    self._strategy = _SubsampleStrategy(strategy)
    self._n_test = n_test
    self._n_train = n_train
    self._seed = seed

  def __call__(self, ds):
    if self._strategy == _SubsampleStrategy.RANDOM:
      rng = np.random.default_rng(self._seed)
      rng2 = np.random.default_rng(self._seed + 1)

      splits = ds["splits_fn"](ds["inputs"])
      splits = instantiate_split_masks(
        splits, ds["inputs"]
      )  # got to make sure boolean masks, not slices

      train_inds = splits["train_inds"]
      N = train_inds.sum()
      sub_train_inds = rng.choice(N, min(N, self._n_train), replace=False)

      test_inds = splits["test_inds"]
      N = test_inds.sum()
      sub_test_inds = rng2.choice(N, min(N, self._n_test), replace=False)

    elif self._strategy == _SubsampleStrategy.FIRST:
      raise NotImplementedError("FIRST strategy not implemented")
    elif self._strategy == _SubsampleStrategy.LAST:
      raise NotImplementedError("Last strategy not implemented")
    else:
      raise ValueError(
        f"Unknown dataset postfilter subsampling strategy: {self._strategy}"
      )

    train_xs = ds["inputs"][train_inds][sub_train_inds]
    test_xs = ds["inputs"][test_inds][sub_test_inds]
    train_ys = ds["labels"][train_inds][sub_train_inds]
    test_ys = ds["labels"][test_inds][sub_test_inds]

    xs = np.concatenate([train_xs, test_xs], axis=0)
    ys = np.concatenate([train_ys, test_ys], axis=0)

    out = {
      "inputs": xs,
      "labels": ys,
      "splits_fn": _get_sequential_splitsfn(len(train_xs)),
      "meta": ds["meta"],
    }
    return out


@register_dspf_handler("pca")
class PCADSPF(DSPostFilter):
  """PCA on the train data, reduce entire DS"""

  def __init__(self, n_components):
    self._n_components = n_components

  def __call__(self, ds):
    inputs = ds["inputs"]

    splits_fn = ds["splits_fn"]
    splits = splits_fn(inputs)

    jarr = np.array(inputs)
    # Normalisation needs to be done on the whole dataset
    jarr = _preprocess_input(jarr)

    # PCA only on the training data
    kernel = _get_pca_kernel(jarr[splits["train_inds"]])

    drjarr = jarr @ kernel[:, : self._n_components]
    drjarr = utils.rescale(drjarr, 0, 1)
    xs = np.array(drjarr)

    # Don't change the order of the inputs/ labels
    out = {
      "inputs": xs,
      "labels": ds["labels"],
      "splits_fn": ds["splits_fn"],
      "meta": ds["meta"],
    }

    return out


# ------------------------------ Split functions ----------------------------- #


def _get_random_splitsfn(split, seed):
  def inner(*inputs):
    N = len(inputs[0])
    rng = np.random.default_rng(seed)
    train_inds = np.sort(rng.choice(N, int(split * N), False))
    test_inds = np.delete(np.arange(N), train_inds)

    out = {
      "train_inds": train_inds,
      "test_inds": test_inds,
    }
    return out

  return inner


def _get_sequential_splitsfn(N):
  def inner(*inputs):
    out = {
      "train_inds": slice(None, N),
      "test_inds": slice(N, None),
    }
    return out

  return inner


def apply_splits(splits, *inputs):
  assert len({len(i) for i in inputs}) == 1, "All inputs must have the same length"

  train_inds = splits["train_inds"]
  test_inds = splits["test_inds"]

  out = {
    "train": [i[train_inds] for i in inputs],
    "test": [i[test_inds] for i in inputs],
    "valid": list(inputs),
  }
  return out


def instantiate_mask(slc, arr):
  """
  Instantiate a boolean mask for the given array, from a slice-like object.
  If scl is an ndarry then it is treated as a mask directly

  Parameters:
  slc (slice): The slice object to convert.
  arr (numpy.ndarray): The array for which the mask is generated.

  Returns:
  numpy.ndarray: A boolean mask with True values corresponding to the slice.
  """
  if isinstance(slc, slice):
    mask = np.zeros(arr.shape[0], dtype=bool)
    mask[slc] = True
  elif isinstance(slc, np.ndarray):
    if slc.dtype == bool:
      mask = slc
    elif slc.dtype == int:
      mask = np.zeros(arr.shape[0], dtype=bool)
      mask[slc] = True
    else:
      raise ValueError(f"Unknown mask dtype: {slc.dtype}")
  else:
    raise ValueError(f"Unknown slice/ mask type: {type(slc)}")
  return mask


def instantiate_split_masks(splits, arr):
  return {k: instantiate_mask(splits[k], arr) for k in splits}


# --------------------------------- DS utils --------------------------------- #


def _get_pca_kernel(data, dimred=None):
  W = data.T @ data
  evs = np.linalg.eig(W)[1].astype(float)

  K = evs.shape[1] if dimred is None else dimred

  kernel = evs[:, :K]
  return kernel


def _preprocess_input(inp):
  jarr = np.array(inp.reshape([inp.shape[0], -1]).astype(float))
  jarr = utils.rescale(jarr, 0, 1)
  return jarr


# --------------------------------- Datasets --------------------------------- #

register_dataset_handler, get_dataset_handler = utils.new_registry("dataset")


class Dataset:
  inputs: np.ndarray
  labels: np.ndarray | None
  name: str | None = None
  tasks: list[Task]

  def _init_common(
    self,
    name: str | None = None,
    tasks: list[TaskRepr] | Task = None,
    postfilters: list[DSPostFilterRepr] | DSPostFilter = None,
    **params,
  ):
    if postfilters is None:
      postfilters = []
    elif isinstance(postfilters, DSPostFilter):
      postfilters = [postfilters]

    self.name = name if name is not None else params.pop("key", None)
    self.tasks = utils.validate_repr(tasks, Task, parse_task, optional=True)
    self.postfilters = utils.validate_repr(
      postfilters, DSPostFilter, parse_dspf, optional=True
    )
    return params

  def __repr__(self):
    return f"{type(self).__name__}: {self.name}"

  def get_splits(self, *inputs): ...


def parse_dataset(type: str, params: dict) -> Dataset:
  return get_dataset_handler(type)(**params)


@register_dataset_handler("func")
class FuncDataset(Dataset):
  def __init__(
    self,
    key: str,
    **params,
  ):
    self._labels = False
    self._inputs = False
    self._key = key
    self._data = None
    self._params = self._init_common(**params, key=key)

  def _lazy_load(self):
    data = get_dataset_generator_handler(self._key)(**self._params)
    self._data = ft.reduce(lambda x, f: f(x), self.postfilters, data)
    self._inputs = self._data["inputs"]
    self._labels = self._data["labels"]

    if self._labels is not None:
      lsh = self._labels.shape
      assert len(lsh) == 2
    ish = self._inputs.shape
    assert len(ish) == 2

  def get_splits(self, *inputs):
    if self._data is None:
      self._lazy_load()
    return self._data["splits_fn"](*inputs)

  @property
  def inputs(self):
    if self._inputs is False:
      self._lazy_load()
    return self._inputs

  @property
  def labels(self):
    if self._labels is False:
      self._lazy_load()
    return self._labels
