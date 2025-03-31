import argparse
import os

import numpy as np

expt = {
  "[0 0 0 0]": "mnist",
  "[0 5 0 0]": "organc",
  "[0 2 0 0]": "blood",
  "[0 4 0 0]": "path",
  "[0 6 0 0]": "oct",
  "[0 1 0 0]": "breast",
  "[0 3 0 0]": "derma",
}

datasets = [
  "Digit",
  "OrganC",
  "Blood",
  "Path",
  "OCT",
  "Breast",
  "Derma",
]

state = {
  "argmax": "argmax",
  "_classical": "cint",
  "_semiclassical": "cpnr",
  "_quantum": "fock",
}

sets = [
  "Train",
  "Test",
]

classifiers = [
  "Fock",
  "Coherent PNR",
  "Coherent Intensity",
  "Majority",
]


def do_parse():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", dest="file", default="out")
  args = parser.parse_args()

  cmd = f"grep -oE \"'accuracy':.*$|'mcc':.*$|Task.*metrics:\" {args.file}.log > {args.file}.filtered.log"
  os.system(cmd)
  print(cmd)
  with open(args.file + ".filtered.log") as f:
    lines = f.readlines()

  out = np.zeros([2 * len(expt), len(state)]) - 2
  mccout = np.zeros([2 * len(expt), len(state)]) - 2
  for i, line in enumerate(lines):
    k = m = jj = None
    for e in expt.keys():
      if e in line:
        k = expt[e]
        break
    if k is None:
      continue
    for e in state.keys():
      if e in line:
        m = state[e]
        break
    if "test" in line:
      jj = 1
    else:
      jj = 0

    if None in (k, jj, m):
      continue

    p = list(expt.values()).index(k) * 2 + jj
    q = len(state) - 1 - list(state.values()).index(m)
    s = lines[i + 1]
    if "accuracy" in s:
      val = float(s.split(" ")[-1].split("}")[0])
      out[p, q] = val
    elif "mcc" in s:
      val = float(s.split(" ")[-1].split("}")[0])
      mccout[p, q] = val
    else:
      continue

  print_out(out)
  print_out(mccout)

  np.savez(
    "accuracies.npz",
    **{
      "acc": out,
      "mcc": mccout,
      "labels": datasets,
      "classifiers": classifiers,
      "sets": sets,
    },
  )


def print_out(out):
  print("\t\t" + "\t".join(list(state.values())[::-1]))
  with np.printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)}):
    lines = (
      np.array2string(out.round(3), precision=3, separator="\t")
      .replace("[", "")
      .replace("]", "")
      .replace(" ", "")
      .split("\n")
    )
    for i, line in enumerate(lines):
      if i % 2 == 0:
        pref = list(expt.values())[i // 2]
      else:
        pref = ""
      print(pref + "\t" + ["train", "test"][i % 2] + "\t" + line)


def main():
  do_parse()


if __name__ == "__main__":
  main()
