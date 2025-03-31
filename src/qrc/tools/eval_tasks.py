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

Must have already generated the processed data with postprocess.py

This script produces a .tasks.npz for every experiment, running
evaluating on all tasks defined below.

"""


_default_evcfg = {
  "training": {
    "rcond": 2e-5,
  },
  "runtime": {
    "workers": 25,
  },
}

runtime_cfg = {}
runtime_cfg.update(_default_evcfg["runtime"])


def train_and_eval(
  train_probs,
  test_probs,
  train_labels,
  test_labels,
  train_xs,
  test_xs,
  task,
  *,
  name,
  rcond,
):
  w, m = qrc.utils.train_reservoir(
    train_probs,
    train_labels,
    rcond=rcond,
  )
  train_fit = qrc.utils.eval_reservoir(train_probs, w)
  test_fit = qrc.utils.eval_reservoir(test_probs, w)

  train_metrics = task.metrics(train_xs, train_fit, train_labels)
  test_metrics = task.metrics(test_xs, test_fit, test_labels)
  logger.info(f"Task {name} {task} train metrics: {train_metrics}")
  logger.info(f"Task {name} {task} test metrics: {test_metrics}")

  return w, m, train_fit, test_fit, train_metrics, test_metrics


def task_eval(wrapped_task):
  task_ind, e, probs, evcfg = wrapped_task

  task = e.dataset.tasks[task_ind]
  xs = e.dataset.inputs
  ys = e.dataset.labels
  if ys is None:
    ys = np.array([None] * len(xs))

  mask = task.domain(xs, ys)
  xs, ys, probs = xs[mask], ys[mask], probs.T[mask]

  # TODO careful! This could cause issues where we have a subset domain - it
  # won't be applied at dataset initialisation
  # This is a direct result of need for backward compat. Ideally we'd scrap
  # domains or switch order of split and domain
  splits = e.dataset.get_splits(xs, ys, probs)
  split_vars = qrc.dataset.apply_splits(splits, xs, ys, probs)

  train_xs, train_ys, train_probs = split_vars["train"]
  test_xs, test_ys, test_probs = split_vars["test"]

  train_labels = task(train_xs, train_ys)
  test_labels = task(test_xs, test_ys)

  gt_labels = task(xs, ys)

  w, m, train_fit, test_fit, train_metrics, test_metrics = train_and_eval(
    train_probs,
    test_probs,
    train_labels,
    test_labels,
    train_xs,
    test_xs,
    task,
    name=e.name,
    rcond=evcfg["training"]["rcond"],
  )
  test_mse_err = (abs(test_fit - test_labels) ** 2).mean()

  out = {
    "task": task.name,
    #
    "train_inds": splits["train_inds"],
    "test_inds": splits["test_inds"],
    #
    "ground_truth_x": xs,
    "ground_truth_y": gt_labels,
    #
    "m": m,
    "w": w,
    #
    "train_fit": train_fit,
    "test_fit": test_fit,
    "test_mse_err": test_mse_err,
    "train_metrics": train_metrics,
    "test_metrics": test_metrics,
  }

  if isinstance(task, qrc.dataset.ClassifyTask):
    # argmax_classifier
    classes = np.argmax(train_labels, axis=1)
    bins = np.bincount(classes.ravel())
    bestclass = np.argmax(bins)

    am_train_fit = 0 * train_labels
    am_train_fit[:, bestclass] = 1
    am_train_metrics = task.metrics(train_xs, am_train_fit, train_labels)
    am_test_fit = 0 * test_labels
    am_test_fit[:, bestclass] = 1
    am_test_metrics = task.metrics(test_xs, am_test_fit, test_labels)

    logger.info(
      f"Task {e.name} argmax_classifier {task} train metrics: "
      + f"\n 'accuracy': {am_train_metrics.get('accuracy')}"
      + "}"
    )
    logger.info(
      f"Task {e.name} argmax_classifier {task} test metrics: "
      + f"\n 'accuracy': {am_test_metrics.get('accuracy')}"
      + "}"
    )
    out.update({
      "am_train_metrics": am_train_metrics,
      "am_test_metrics": am_test_metrics,
    })

  return out


def do_task_eval():
  definition, cfg = qrc.ui.setup()

  expts = definition.get_experiments()
  try:
    evcfg = definition.get_eval()
  except AttributeError:
    logger.warning("No eval config found, using default")
    evcfg = {}
  evcfg = qrc.utils.dict_tree_update(_default_evcfg, evcfg)
  runtime_cfg.update(evcfg["runtime"])

  shaped_experiments = [defs.tensor_experiments_to_list(te) for te in expts]
  experiments = [e for se in shaped_experiments for e in se.ravel()]
  expt_objs = [defs.Experiment(**e) for e in experiments]

  _, workers = qrc.runtime.multi_parfor(
    [],
    lambda x: x,
    workers=runtime_cfg["workers"],
    ret_workers=True,
  )

  for e in tqdm(expt_objs):
    outpath = cfg["outdir"] / f"{e.name}.tasks.npz"

    if outpath.exists():
      logger.info(f"Skipping {e.name}, already exists")
      continue

    file = (cfg["indir"] / e.name).with_suffix(".postprocess.npz")
    if not file.exists():
      continue
    data = np.load(file, allow_pickle=True)["data"].ravel()[0]

    probs = None
    if isinstance(e.simulator, sims.ClassicalSimulator):
      probs = data["depol_data"]["probs"]
    elif isinstance(
      e.simulator, (sims.EfficientCoherentSimulator, sims.SLOSProbsSimulator)
    ):
      probs = data["sampled_data"]["probs"]

    logger.info(f"Processing {e.name}")

    """iterate tasks"""
    task_stats, workers = qrc.runtime.multi_parfor(
      [(t_ind, e, probs, evcfg) for t_ind in range(len(e.dataset.tasks))],
      task_eval,
      workers=workers,
      ret_workers=True,
    )

    logger.info(f"Done {e.name}")

    qrc.ui.save_data(outpath, data=task_stats)

  # Close process pool
  qrc.runtime.multi_parfor(
    [],
    lambda x: x,
    workers=workers,
  )


def main():
  do_task_eval()


if __name__ == "__main__":
  main()
