import numpy as np
from loguru import logger
from tqdm.auto import tqdm

import qrc
import qrc.ui
from qrc import experiments as defs


def mcc(cm):
  c = np.diag(cm).sum()
  s = cm.sum()
  p = cm.sum(axis=0)
  t = cm.sum(axis=1)
  if (c * s - t @ p) == 0:
    return 0
  return (c * s - t @ p) / (np.sqrt(s**2 - p @ p) * np.sqrt(s**2 - t @ t))


def task_eval(wrapped_task):
  task_ind, e, data = wrapped_task

  task = e.dataset.tasks[task_ind]

  trainmcc = mcc(data["train_metrics"]["confmat"])
  testmcc = mcc(data["test_metrics"]["confmat"])
  logger.info(f"Task {e.name} {task} train metrics: \n 'mcc': {trainmcc}")
  logger.info(f"Task {e.name} {task} test metrics: \n 'mcc': {testmcc}")

  trainmcc = mcc(data["am_train_metrics"]["confmat"])
  testmcc = mcc(data["am_test_metrics"]["confmat"])
  logger.info(f"Task {e.name} argmax {task} train metrics: \n 'mcc': {trainmcc}")
  logger.info(f"Task {e.name} argmax {task} test metrics: \n 'mcc': {testmcc}")

  return {}


def main():
  definition, cfg = qrc.ui.setup()

  expts = definition.get_experiments()

  shaped_experiments = [defs.tensor_experiments_to_list(te) for te in expts]
  experiments = [e for se in shaped_experiments for e in se.ravel()]
  expt_objs = [defs.Experiment(**e) for e in experiments]

  for e in tqdm(expt_objs):
    file = (cfg["indir"] / e.name).with_suffix(".tasks.npz")
    if not file.exists():
      continue
    data = np.load(file, allow_pickle=True)["data"].ravel()[0]

    logger.info(f"Processing {e.name}")

    """iterate tasks"""
    qrc.runtime.multi_parfor(
      [(t_ind, e, data) for t_ind in range(len(e.dataset.tasks))],
      task_eval,
      workers=0,
      ret_workers=True,
    )

    logger.info(f"Done {e.name}")


if __name__ == "__main__":
  main()
