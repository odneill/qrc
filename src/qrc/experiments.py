from loguru import logger

from . import __version__ as qrc_version
from . import dataset as ds
from . import encodings, networks, states, utils
from . import simulators as sims

ExperimentRepr = dict

# ------------------------------------- x ------------------------------------ #


def tensor_experiments_to_list(tensor_expt: dict):
  logger.info(f"library version: {qrc_version}")

  name = tensor_expt.pop("name", None)
  simulator = tensor_expt.pop("simulator", None)

  _tensor_expt = {k[:-1]: v for k, v in tensor_expt.items()}

  experiments = utils.tensor_map(
    lambda index, **kwargs: dict(
      name=name + "_" + str(index),
      simulator=simulator,
      **kwargs,
    ),
    **_tensor_expt,
    iterate=True,
  )
  return experiments


class Experiment:
  name: str
  state: states.State
  dataset: ds.Dataset
  reservoir: networks.Reservoir
  encoding: encodings.Encoding
  _dataset: ds.Dataset = None

  def __init__(
    self,
    name: str = "",
    state: states.StateRepr | states.State = None,
    simulator: sims.SimulatorRepr | sims.Simulator = None,
    dataset: ds.DatasetRepr | ds.Dataset = None,
    reservoir: networks.ReservoirRepr | networks.Reservoir = None,
    encoding: encodings.EncodingRepr | encodings.Encoding = None,
  ):
    repr_ = {
      "name": name,
      "state": state,
      "simulator": simulator,
      "dataset": dataset,
      "reservoir": reservoir,
      "encoding": encoding,
    }
    self._init(repr_)

  def _init(self, repr_):
    self.repr = repr_
    self.name = repr_["name"]
    self.state = utils.validate_repr(repr_["state"], states.State, states.parse_state)
    self.reservoir = utils.validate_repr(
      repr_["reservoir"], networks.Reservoir, networks.parse_reservoir
    )
    self.encoding = utils.validate_repr(
      repr_["encoding"], encodings.Encoding, encodings.parse_encoding
    )
    self.simulator = utils.validate_repr(
      repr_["simulator"], sims.Simulator, sims.parse_simulator, True
    )

    assert self.state._pol == self.reservoir.polarised, (
      "State and reservoir must have matching polarisation"
    )
    m = self.reservoir.num_spatial_modes
    if self.state._pol:
      m *= 2
    assert self.state._modes <= m, (
      "State and reservoir must have matching number of modes"
    )

    if self.reservoir.lossy:
      self.state._lossy = True

  @property
  def dataset(self) -> ds.Dataset:
    """
    Dataset is lazy loaded as this may be heavy.
    Not necessary to repeat in subprocess during runtime if batching is used.
    """
    if self._dataset is None:
      self._dataset = utils.validate_repr(
        self.repr["dataset"], ds.Dataset, ds.parse_dataset
      )

    return self._dataset

  def __repr__(self):
    return (
      f"Experiment: {self.name} \n{self.state} \n{self.dataset} "
      + f"\n{self.reservoir} \n{self.encoding}"
    )

  def __getstate__(self):
    """Pickle support for multiprocess runtimes"""
    return self.repr

  def __setstate__(self, repr_):
    """Pickle support for multiprocess runtimes"""
    self._init(repr_)
