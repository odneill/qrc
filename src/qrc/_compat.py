"""
Compatibility layer for Perceval
"""

import copyreg

import numpy as np
from perceval.utils import BasicState, BSDistribution, StateVector
from perceval.utils.globals import global_params


def reduce_SVlike(s):
  return type(s), (), None, None, s.__reduce__()[4]


def reduce_BasicState(s):
  return BasicState, (str(s),)


"""New vectorial normalization method for state vectors"""


def normalize(self):
  r"""Normalize a state vector"""
  if not self._normalized:
    # norm = 0
    norm = 0 * list(self.values())[0]
    to_remove = []
    for key in self.keys():
      # if (isinstance(self[key], (complex, float, int))
      #     and abs(self[key]) < global_params["min_complex_component"]) or self[key] == 0:
      if (
        isinstance(self[key], (complex, float, int, np.ndarray))
        and np.sum(abs(self[key])) < global_params["min_complex_component"]
      ) or np.all(self[key] == 0):
        to_remove.append(key)
      else:
        norm += abs(self[key]) ** 2
    for key in to_remove:
      del self[key]
    norm = norm**0.5
    nkey = len(self.keys())
    for key in self.keys():
      if nkey == 1:
        # self[key] = 1
        self[key] = 1 + 0 * self[key]
      else:
        self[key] /= norm
    self._normalized = True


def patch():
  """Register the custom pickling functions"""
  copyreg.pickle(StateVector, reduce_SVlike)
  copyreg.pickle(BSDistribution, reduce_SVlike)
  copyreg.pickle(BasicState, reduce_BasicState)

  # Monkey patch the normalize method
  StateVector.normalize = normalize
