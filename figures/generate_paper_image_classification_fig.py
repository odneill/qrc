from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figutil import util


def plotfig(data, sets, datasets, classifiers):
  # Create x-ticks positions
  x_positions = np.arange(len(data))
  x2_positions = np.linspace(0.5, 12.5, 7)
  xlabels = sets * len(datasets)

  colors = ["r", "b", "y", "k"]
  markers = ["o", "+", "x", "_"]

  plt.hlines(np.linspace(0, 1, 11), -0.5, 13.5, color="gray", alpha=0.2)

  # Plot scatter points
  for i in range(len(data.T)):
    plt.plot(
      x_positions,
      data[:, i],
      label=classifiers[i],
      marker=markers[i],
      linestyle="None",
      color=colors[i],
      markerfacecolor="none",
      markerfacecoloralt="none",
    )

  plt.xticks(x_positions, xlabels)

  sec = plt.gca().secondary_xaxis(location=0)
  sec.set_xticks(x2_positions, labels=datasets, fontsize=util.FONT_MEDIUM)
  sec.tick_params(axis="x", pad=20)
  sec.tick_params("x", length=0)

  sec2 = plt.gca().secondary_xaxis(location=0)
  sec2.set_xticks(np.linspace(-0.5, 15.5, 9), labels=[])
  sec2.tick_params("x", length=30, width=1)

  plt.xlim(-0.5, 13.5)


def main():
  data = np.load(
    Path(__file__).parent.parent / "./experiments/image_classification/accuracies.npz",
    allow_pickle=True,
  )
  accuracies = data["acc"]
  mccs = data["mcc"]
  datasets = list(data["labels"])
  classifiers = list(data["classifiers"])
  sets = list(data["sets"])

  plt.rcParams.update(util.get_default_params())
  plt.rcParams.update({
    "font.size": 10,
    "ytick.labelsize": util.FONT_SMALL,
    "legend.fontsize": util.FONT_SMALL,
  })

  plt.figure(figsize=(5.25, 5))

  plt.subplot(2, 1, 1)
  plotfig(accuracies, sets, datasets, classifiers)
  plt.ylim(0, 1)
  plt.ylabel("Accuracy")
  plt.legend(
    loc="upper right",
    ncol=4,
    borderaxespad=0.2,
    columnspacing=0.5,
    handletextpad=0.1,
  )

  plt.subplot(2, 1, 2)
  plotfig(mccs, sets, datasets, classifiers)
  plt.ylim(-0.2, 1)
  plt.ylabel("MCC")
  plt.legend(
    loc="upper right",
    ncol=4,
    borderaxespad=0.2,
    columnspacing=0.5,
    handletextpad=0.1,
  )

  plt.tight_layout()

  plt.savefig(Path(__file__).parent / "image_classification.pdf")


if __name__ == "__main__":
  main()
