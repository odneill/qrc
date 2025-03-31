import numpy as np
from loguru import logger
from tqdm.auto import tqdm

import qrc
import qrc.experiments as defs
import qrc.simulators as sims

"""
Run this from a terminal in a directory, with a definition.py file.
This should provide a get_experiments() function which returns a list of tensor
experiment definitions in repr form.

Must have already generated the raw data with run_experiments.py

This script produces a .postprocess.npz for every experiment, running the
pipeline defined below. The output saves intermediate data and metrics for each
step.


"""


_default_ppcfg = {
  "detector": {
    "darkcounts": 0,
    "detector_efficiency": 0.9,
    "pthresh": 1e-20,  # must be greater than, else filtered out
  },
  "runtime": {
    "workers": 25,
    "split_factor": 1,
  },
  "sampling": {
    "seed": None,
    "f": 10_000_000,
  },
  "renorm": {
    "remove_dummy": True,
    "pthresh": 1e-20,  # for summing to 1, takes up slack from perceval threshold
    "remove_dummy_final": True,
  },
  "postsel": {
    "lower": 1,
    "upper": 4,
  },
}

runtime_cfg = {}
runtime_cfg.update(_default_ppcfg["runtime"])


def get_metrics(data: dict):
  p = np.array(data["probs"])

  if p.shape[0] < p.shape[1]:
    D = p @ p.T
  else:
    D = p.T @ p

  svd = np.linalg.svd(D)
  svs = svd[1]

  return {"svs": np.array(svs), "D": np.array(D)}


def apply_noise(data, darkcounts, detector_efficiency, pthresh=1e-20):
  dists = data["probs"]
  modes = np.array(data["modes"])

  noisemodes, noiseprobs = qrc.filters.noise_model(
    (dists, modes),
    expected_darkcounts=darkcounts,
    detector_efficiency=detector_efficiency,
    PTHRESH=pthresh,
  )
  return {"probs": noiseprobs, "modes": noisemodes}


def depolarise(data):
  dists = data["probs"]
  modes = np.array(data["modes"])

  outmodes, outprobs = qrc.filters.postselect(dists, modes, qrc.filters.pol_filter)

  return {"probs": np.array(outprobs), "modes": outmodes}


def postselect(data, flt):
  dists = data["probs"]
  modes = np.array(data["modes"])

  outmodes, outprobs = qrc.filters.postselect(dists, modes, flt)

  return {"probs": np.array(outprobs), "modes": outmodes}


def renormalise(data, *, pthresh):
  dists = data["probs"]
  modes = np.array(data["modes"])

  (outprobs, outmodes), norm = qrc.filters.renormalisation(
    dists, modes, PTHRESH=pthresh
  )

  return {"probs": np.array(outprobs), "modes": outmodes, "norm": norm}


def dummy_renormalise(data, *, pthresh):
  dists = data["probs"]
  modes = np.array(data["modes"])

  (outprobs, outmodes), norm = qrc.filters.dummy_renormalisation(
    dists, modes, PTHRESH=pthresh
  )

  return {"probs": np.array(outprobs), "modes": outmodes, "norm": norm}


def filter_dummy(data):
  dists = data["probs"]
  modes = np.array(data["modes"])

  (outprobs, outmodes), delta = qrc.filters.filter_dummy(dists, modes)

  return {"probs": np.array(outprobs), "modes": outmodes, "delta": delta}


def _par_sample_runner(arg):
  batched, static = arg
  fdm = static
  probs = batched[0].T
  seeds = batched[1]
  outprobs = qrc.filters.sample_dist(probs, f=fdm, seeds=seeds)
  return outprobs


def par_sample(data, seed, f):
  dists = data["probs"]
  modes = np.array(data["modes"])

  M, N = dists.shape
  rng = np.random.default_rng(seed)
  if seed is None:
    seed = rng.integers(0, 2**32)
    rng = np.random.default_rng(seed)
  seeds = rng.integers(0, 2**32, size=N)

  static = f / M
  batched = (dists.T, seeds)
  out = qrc.runtime.batched_multi_parfor(
    [(batched, static)],
    _par_sample_runner,
    workers=runtime_cfg["workers"],
    pbars=False,
    split_factor=runtime_cfg["split_factor"],
  )[0]

  outprob = np.concatenate(out, axis=1)

  return {"probs": outprob, "modes": modes}, seed


def quantum_post_process(data, ppcfg):
  logger.info("Starting quantum post process")
  # data_metrics = get_metrics(data)
  data_metrics = None
  logger.info("Done data metrics")

  depol_data = depolarise(data)
  logger.info("Done depol")
  depol_data_metrics = get_metrics(depol_data)
  logger.info("Done depol metrics")

  noisy_data = apply_noise(
    depol_data,
    darkcounts=ppcfg["detector"]["darkcounts"],
    detector_efficiency=ppcfg["detector"]["detector_efficiency"],
    pthresh=ppcfg["detector"]["pthresh"],
  )
  logger.info("Done noise")
  noisy_data_metrics = get_metrics(noisy_data)
  logger.info("Done noise metrics")
  noisy_data_metrics["darkcounts"] = ppcfg["detector"]["darkcounts"]
  noisy_data_metrics["detector_efficiency"] = ppcfg["detector"]["detector_efficiency"]

  postselect_data = postselect(
    noisy_data,
    lambda m: qrc.filters.specific_range_filter(
      m, ppcfg["postsel"]["lower"], ppcfg["postsel"]["upper"]
    ),
  )
  logger.info("Done postsel")
  postselect_data_metrics = get_metrics(postselect_data)
  logger.info("Done postsel metrics")
  postselect_data_metrics["filter"] = (
    f"number,{ppcfg['postsel']['lower']}-{ppcfg['postsel']['upper']}"
  )

  # If remove_dummy
  #   We remove it, and then renormalise to 1, then sample
  #   This is equivalent to measuring until N valid events
  # Else
  #   We renormalise, by adding extra prob to the dummy, sample, then remove
  #   This is measuring N total frames, then taking the smaller subset of valid events

  if ppcfg["renorm"]["remove_dummy"]:
    _postselect_data = filter_dummy(postselect_data)
    renormalised_data = renormalise(
      _postselect_data, pthresh=ppcfg["renorm"]["pthresh"]
    )
  else:
    renormalised_data = dummy_renormalise(
      postselect_data, pthresh=ppcfg["renorm"]["pthresh"]
    )
  logger.info("Done renorm")
  renormalised_data_metrics = get_metrics(renormalised_data)
  logger.info("Done renorm metrics")

  sampled_data_w_dummy, seed = par_sample(
    renormalised_data, seed=ppcfg["sampling"]["seed"], f=ppcfg["sampling"]["f"]
  )
  logger.info("Done sample")
  if ppcfg["renorm"].get("remove_dummy_final", True):
    sampled_data = filter_dummy(sampled_data_w_dummy)
  else:
    sampled_data = sampled_data_w_dummy

  sampled_data_metrics = get_metrics(sampled_data)
  logger.info("Done sample metrics")
  sampled_data_metrics["seed"] = seed
  sampled_data_metrics["f"] = ppcfg["sampling"]["f"]

  outdata = {
    "data_metrics": data_metrics,
    "depol_data": depol_data,
    "depol_data_metrics": depol_data_metrics,
    "noisy_data": noisy_data,
    "noisy_data_metrics": noisy_data_metrics,
    "postselect_data": postselect_data,
    "postselect_data_metrics": postselect_data_metrics,
    "renormalised_data": renormalised_data,
    "renormalised_data_metrics": renormalised_data_metrics,
    "sampled_data": sampled_data,
    "sampled_data_metrics": sampled_data_metrics,
  }
  return outdata


def classical_post_process(data):
  data_metrics = get_metrics(data)

  depol_data = depolarise(data)
  depol_data_metrics = get_metrics(depol_data)

  outdata = {
    "data_metrics": data_metrics,
    "depol_data": depol_data,
    "depol_data_metrics": depol_data_metrics,
  }
  return outdata


def post_process(e: defs.Experiment, cfg, ppcfg):
  runtime_cfg.update(ppcfg["runtime"])

  outpath = (cfg["outdir"] / e.name).with_suffix(".postprocess.npz")
  if not outpath.exists():
    inpath = (cfg["indir"] / e.name).with_suffix(".npz")

    _data = qrc.ui.load_data_flat(inpath, mmap_mode="r")
    if isinstance(e.simulator, sims.ClassicalSimulator):
      data = {
        "probs": abs(_data["amplitudes"]) ** 2,
        "modes": np.eye(_data["amplitudes"].shape[0]),
      }
      del _data
      logger.info(f"Processing {e.name}")
      outdata = classical_post_process(data)
      logger.info(f"Done {e.name}")
    elif isinstance(
      e.simulator, (sims.SLOSProbsSimulator, sims.EfficientCoherentSimulator)
    ):
      modes = _data["modes"]
      if hasattr(modes.ravel()[0], "keys"):
        modes = np.array(list(modes.ravel()[0].keys()))
      data = {
        "modes": modes,
        "probs": _data["probs"],
      }
      logger.info(f"Processing {e.name}")
      outdata = quantum_post_process(data, ppcfg)
      logger.info(f"Done {e.name}")

    qrc.ui.save_data(outpath, data=outdata)
  else:
    logger.info(f"skipping {e.name}")


def do_postprocess():
  definition, cfg = qrc.ui.setup()

  expts = definition.get_experiments()
  try:
    ppcfg = definition.get_postprocess()
  except AttributeError:
    logger.warning("No postprocess config found, using default")
    ppcfg = {}
  ppcfg = qrc.utils.dict_tree_update(_default_ppcfg, ppcfg)

  shaped_experiments = [defs.tensor_experiments_to_list(te) for te in expts]
  experiments = [e for se in shaped_experiments for e in se.ravel()]
  expt_objs = [defs.Experiment(**e) for e in experiments]

  for ex in tqdm(expt_objs):
    try:
      post_process(ex, cfg, ppcfg)
      qrc.runtime.trim_memory()
    except FileNotFoundError:
      continue


def main():
  do_postprocess()


if __name__ == "__main__":
  main()
