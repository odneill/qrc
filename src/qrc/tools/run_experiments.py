from loguru import logger

import qrc
import qrc.experiments as defs

"""
Run this from a terminal in a directory, with a definition.py file.
This should provide a get_experiments() function which returns a list of tensor
experiment definitions in repr form.

For each experiment in the tensor product, we write an npz file with the name
and nd index with data "probs" and "modes".
"""


def do_experiment(e, name, outfile):
  """
  Returns 1 for failure condition
  """

  try:
    E = defs.Experiment(**e)
  except Exception as excpt:
    logger.info(f"failed loading experiment {name}: {excpt}")
    return 1

  logger.info(f"Starting {name}")
  try:
    result = E.simulator.run(E)
    qrc.ui.save_data(outfile, **result)

  except Exception as excpt:
    logger.info(f"failed running experiment {e.get('name', 'unknown')}: {excpt}")
    return 1

  logger.info(f"Ended {name}")
  return 0


def do_optics_simulation():
  definition, cfg = qrc.ui.setup()

  expts = definition.get_experiments()

  shaped_experiments = [defs.tensor_experiments_to_list(te) for te in expts]
  experiments = [e for se in shaped_experiments for e in se.ravel()]
  logger.info(f"running {len(experiments)} experiments")
  logger.info(str([se.shape for se in shaped_experiments]))
  es = list(experiments)
  fails = {}
  while True:
    if len(es) == 0:
      break
    e = es.pop(0)
    name = e.get("name", "unknown")
    if fails.get(name, 0) > 3:
      continue

    outfile = cfg["outdir"] / f"{name}.npz"
    if (outfile).exists():
      logger.info(f"skipping {name}")
      continue

    fail = do_experiment(e, name, outfile)
    qrc.runtime.trim_memory()

    if fail:
      es.append(e)
      fails[name] = fails.get(name, 0) + 1
      continue


def main():
  do_optics_simulation()


if __name__ == "__main__":
  main()
