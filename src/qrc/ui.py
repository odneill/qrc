import argparse
import importlib.util
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from qrc import __version__ as qrc_version


def cli(*, parser_cb=None):
  curdir = Path(os.getcwd()).absolute()

  def abspath(pathstr):
    return Path(pathstr).absolute()

  def relpath(pathstr):
    return Path(pathstr)

  parser = argparse.ArgumentParser(
    description="Run experiments from a definition.py file"
  )
  parser.add_argument(
    "-d",
    "--definition",
    dest="definition",
    type=abspath,
    default=str(curdir / "definition.py"),
    help="Definition file to load experiments from",
  )
  parser.add_argument(
    "-i",
    "--datadir",
    dest="datadir",
    type=abspath,
    default=str(curdir),
    help="Optional directory where data files should be loaded from",
  )
  parser.add_argument(
    "-o",
    "--outdatadir",
    dest="outdatadir",
    type=abspath,
    default=str(curdir),
    help="Optional directory where data files should be written to",
  )
  parser.add_argument(
    "-l",
    "--logfile",
    dest="logfile",
    type=relpath,
    default="./out.log",
    help="Optional output log file, absolute or relative to outdatadir",
  )
  if parser_cb is not None:
    parser_cb(parser)
  args = parser.parse_args()
  return args


def setup():
  args = cli()

  logger.remove()
  logger.add(args.outdatadir / args.logfile, enqueue=True)
  logger.add(
    sys.stdout,
    filter=lambda record: not (
      "qrc.runtime" in record["name"] and record["level"].no <= logger.level("INFO").no
    ),
  )

  logger.info("Args: " + str(args))

  cfg = {}
  cfg["outdir"] = args.outdatadir
  cfg["indir"] = args.datadir
  cfg["deffile"] = args.definition
  cfg["defdir"] = args.definition.parent

  sys.path.insert(0, str(cfg["defdir"]))
  definition = importlib.import_module(cfg["deffile"].stem)

  return definition, cfg


def gen_meta(dire):
  meta = {}
  meta["version"] = qrc_version
  meta["dir"] = str(dire)
  meta["timestamp"] = datetime.now().isoformat()
  return meta


def _load_npz_items(path, mmap_mode=None):
  with zipfile.ZipFile(path, "r") as zf:
    files = [
      f.filename[: -(len(".npy"))] for f in zf.filelist if f.filename.endswith(".npy")
    ]
  return {name: _load_npy_from_npz(path, name, mmap_mode=mmap_mode) for name in files}


def _load_npy_from_npz(zippath, item, mmap_mode=None, allow_nonmemmap=True):
  with zipfile.ZipFile(zippath, "r") as zf:
    if mmap_mode is not None and zf.getinfo(item + ".npy")._compresslevel is None:
      with zf.open(item + ".npy") as npy_file:
        zoffset = npy_file._orig_compress_start
        version = np.lib.format.read_magic(npy_file)
        shape, fortran_order, dtype = np.lib.format._read_array_header(
          npy_file, version
        )
        foffset = npy_file.tell()
      if dtype.hasobject:
        mmap_mode = None
    else:
      # "Compressed npz not supported"
      mmap_mode = None

    if mmap_mode is None:
      if not allow_nonmemmap:
        raise ValueError("Object arrays not supported")

      with zf.open(item + ".npy") as npy_file:
        array = np.lib.format.read_array(npy_file, allow_pickle=True)
    else:
      if fortran_order:
        order = "F"
      else:
        order = "C"
      offset = zoffset + foffset

      array = np.memmap(
        zippath, dtype=dtype, mode=mmap_mode, shape=shape, order=order, offset=offset
      )

  return array


def save_data(path, **data):
  np.savez(path, **data, meta=gen_meta(path.parent))


def save_data_flat(path, **data):
  """Saves a data_dict as individual arrays"""
  outdata = flatten_tree(data)
  assert "meta" not in outdata
  # outdata["meta"] = gen_meta(path.parent)
  np.savez(path, **outdata)


def load_data_flat(path, *, mmap_mode=None):
  """Saves a data_dict as individual arrays"""

  data = _load_npz_items(path, mmap_mode=mmap_mode)
  data = unflatten_tree(data)
  return data


def flatten_tree(data):
  flat = {}
  if isinstance(data, dict):
    for k, v in data.items():
      for kk, vv in flatten_tree(v).items():
        flat[f"{k}.{kk}"] = vv
  elif isinstance(data, list):
    for i, v in enumerate(data):
      for kk, vv in flatten_tree(v).items():
        flat[f"_list_{i}.{kk}"] = vv
  elif isinstance(data, tuple):
    for i, v in enumerate(data):
      for kk, vv in flatten_tree(v).items():
        flat[f"_tuple_{i}.{kk}"] = vv
  else:
    flat["_fv_"] = data
  return flat


def unflatten_tree(data):
  if not all("_fv_" in k for k in data.keys()):
    """If already unflattened, return"""
    return data

  h_keys = list({k.split(".", 1)[0] for k in data.keys()})
  if h_keys[0] == "_fv_":
    assert len(h_keys) == 1
    unflat = data["_fv_"]
  elif all(h[: len("_list_")] == "_list_" for h in h_keys):
    unflat = [0] * len(h_keys)
    for h in h_keys:
      subdata = {
        k.split(".", 1)[1]: v for k, v in data.items() if k.split(".", 1)[0] == h
      }
      i = int(h[len("_list_") :])
      unflat[i] = unflatten_tree(subdata)
  elif all(h[: len("_tuple_")] == "_tuple_" for h in h_keys):
    unflat = [0] * len(h_keys)
    for h in h_keys:
      subdata = {
        k.split(".", 1)[1]: v for k, v in data.items() if k.split(".", 1)[0] == h
      }
      i = int(h[len("_tuple_") :])
      unflat[i] = unflatten_tree(subdata)
    unflat = tuple(unflat)
  else:
    unflat = {}
    for h in h_keys:
      subdata = {
        k.split(".", 1)[1]: v for k, v in data.items() if k.split(".", 1)[0] == h
      }
      unflat[h] = unflatten_tree(subdata)
  return unflat
