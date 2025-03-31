import numpy as np
from tqdm.auto import tqdm

import qrc
from qrc import experiments as defs
from qrc import simulators as sims


def main():
  definition, cfg = qrc.ui.setup()

  expts = definition.get_experiments()

  shaped_experiments = [defs.tensor_experiments_to_list(te) for te in expts]
  experiments = [e for se in shaped_experiments for e in se.ravel()]
  expt_objs = [defs.Experiment(**e) for e in experiments]

  states = {
    "fock": "fock",
    "dist": "dist",
    "photon_added_coherent": "hybrid",
    "coherent": "coherent_pnr",
    "classical_coherent": "coherent_classical",
  }
  encodings = {
    "uniform_linear_rand_phase": "lin",
    "multilinear_rand_phase": "multilin",
    "multispiral_overlap_rand_phase": "spiral",
  }

  N = len(expt_objs[0].dataset.inputs.ravel())
  xs = np.linspace(-1, 1, N, endpoint=False)
  assert np.all(xs == expt_objs[0].dataset.inputs.ravel())

  outdata = {}

  for i, e in enumerate(tqdm(expt_objs)):
    state = e.state.name
    enc = e.encoding.name
    res = e.reservoir._reservoirs[1].seed

    if not (state in states and enc in encodings):
      continue

    file = (cfg["indir"] / e.name).with_suffix(".tasks.npz")
    if not file.exists():
      continue
    indata = np.load(file, allow_pickle=True)["data"].ravel()

    file = (cfg["indir"] / e.name).with_suffix(".postprocess.npz")
    if not file.exists():
      continue
    ppdata = np.load(file, allow_pickle=True)["data"].ravel()[0]

    key = None
    if isinstance(e.simulator, sims.ClassicalSimulator):
      key = "depol"
    elif isinstance(
      e.simulator, (sims.SLOSProbsSimulator, sims.EfficientCoherentSimulator)
    ):
      key = "sampled"

    Rank = np.linalg.matrix_rank(ppdata[key + "_data_metrics"]["D"])

    inds = (xs >= 0) * (xs <= 1)
    probs = ppdata[key + "_data"]["probs"][:, inds]

    m = probs @ probs.T

    svs = np.linalg.svd(m)[1]
    Rank2 = np.linalg.matrix_rank(m)

    if states[state] not in outdata:
      outdata[states[state]] = {}
    if encodings[enc] not in outdata[states[state]]:
      outdata[states[state]][encodings[enc]] = {}
    for task in indata:
      name = task["task"]
      if name not in outdata[states[state]][encodings[enc]]:
        outdata[states[state]][encodings[enc]][name] = {}
      if res not in outdata[states[state]][encodings[enc]][name]:
        outdata[states[state]][encodings[enc]][name][res] = {}

      outdata[states[state]][encodings[enc]][name][res]["R"] = Rank
      outdata[states[state]][encodings[enc]][name][res]["R2"] = Rank2
      outdata[states[state]][encodings[enc]][name][res]["svs2"] = svs
      outdata[states[state]][encodings[enc]][name][res]["mse"] = task["test_mse_err"]
      outdata[states[state]][encodings[enc]][name][res]["gtx"] = task["ground_truth_x"]
      outdata[states[state]][encodings[enc]][name][res]["gty"] = task["ground_truth_y"]
      outdata[states[state]][encodings[enc]][name][res]["trainx"] = task[
        "ground_truth_x"
      ][task["train_inds"]]
      outdata[states[state]][encodings[enc]][name][res]["trainy"] = task["train_fit"]
      outdata[states[state]][encodings[enc]][name][res]["testx"] = task[
        "ground_truth_x"
      ][task["test_inds"]]
      outdata[states[state]][encodings[enc]][name][res]["testy"] = task["test_fit"]

  qrc.ui.save_data(cfg["outdir"] / "task_interp_data.npz", data=outdata)


if __name__ == "__main__":
  main()
