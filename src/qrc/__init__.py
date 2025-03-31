from importlib.metadata import PackageNotFoundError, version

try:
  # Get version in editable installation
  import setuptools_scm

  __version__ = setuptools_scm.get_version(root="../../", relative_to=__file__)
except Exception:
  try:
    __version__ = version("qrc")
  except PackageNotFoundError:
    __version__ = "unknown"

from . import _compat

_compat.patch()

# flake8: noqa: E402
from . import dataset as dataset
from . import encodings as encodings
from . import experiments as experiments
from . import filters as filters
from . import networks as networks
from . import runtime as runtime
from . import states as states
from . import ui as ui
from . import utils as utils
