import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figutil import colours, util
from matplotlib.legend_handler import HandlerLine2D

import qrc.ui


def draw_figure(axes, cfg):
  loaded = np.load(
    cfg["indir"] / "task_interp_data.npz",
    allow_pickle=True,
  )
  data = loaded["data"].ravel()[0]

  rcols = list(colours.colours["plot"].values())
  cols = [
    "red",
    "k",
    "green",
    "blue",
    "orange",
  ]

  ename_lookup = {
    "spiral": "Spiral",
    "multilin": "Multilinear",
    "lin": "Uniform linear",
  }
  enames = list(ename_lookup.values())

  sname_lookup = {
    "fock": "Fock",
    "dist": "Dist.",
    "coherent_pnr": "Coherent PNR",
    "coherent_classical": "Coherent Intensity",
    "hybrid": "Hybrid",
  }

  datasets = {}

  dx = 0.17
  Ns = 5
  axes[0, 0].text(
    -0.7, 10, "\\(\\langle R_c \\rangle\\)", ha="center", fontsize=util.FONT_SMALL
  )
  axes[0, 0].text(
    0, 8 * 10**-6, "[", rotation=90, ha="center", fontsize=80, color="#777777"
  )
  axes[0, 0].text(
    1, 8 * 10**-6, "[", rotation=90, ha="center", fontsize=80, color="#777777"
  )
  axes[0, 0].text(
    2, 8 * 10**-6, "[", rotation=90, ha="center", fontsize=80, color="#777777"
  )

  def filled_marker_style(col):
    return {
      "marker": "o",
      "linestyle": "None",
      "markersize": 3,
      "color": col,
      "markerfacecolor": col,
      "markerfacecoloralt": col,
      "markeredgecolor": "none",
    }

  for i, (sname, state) in enumerate(list(data.items())):
    sname = sname_lookup[sname]
    for j, (ename, enc) in enumerate(state.items()):
      ename = ename_lookup[ename]
      j = enames.index(ename)

      datasets[(i, j)] = []
      for k, (_, tasks) in enumerate(enc.items()):
        mses = []
        Rcs = []
        for p, (_, task) in enumerate(tasks.items()):
          svs = task["svs2"]
          svs = svs / svs.sum()
          Rc = int((svs > 3 * 10**-7).sum())
          mses.append(task["mse"])
          Rcs.append(Rc)

        for p, (_, task) in enumerate(tasks.items()):
          if i + j + p == 0:
            ax = axes[0, 1]
            ax.plot(
              task["gtx"], k + task["gty"], alpha=0.4, color=rcols[k % len(rcols)]
            )

          ax = axes[0, 0]
          ax.plot(
            [j + dx * i - (Ns - 1) * dx / 2 + (p - 4.5) * 0.01],
            task["mse"],
            **filled_marker_style(cols[i]),
            alpha=0.1,
            label=sname if k + j + p == 0 else "_nolegend_",
          )
          datasets[(i, j)].append(task["mse"])

        if k == 0:
          axes[0, 0].text(
            j + dx * i - (Ns - 1) * dx / 2,
            10,
            rf"\({np.array(Rcs).mean():.1f}\)",
            ha="center",
            fontsize=util.FONT_SMALL,
          )

      plt.sca(axes[0, 0])
      bp = plt.boxplot(
        datasets[(i, j)],
        positions=[j + dx * i - (Ns - 1) * dx / 2],
        widths=[dx / 2],
        showfliers=False,
      )
      for patch in bp["medians"]:
        patch.set_color("black")

  plt.sca(axes[0, 1])
  plt.gca().set_xticks([0, 1])
  plt.gca().set_yticks([])
  plt.title("(a)", loc="left")
  plt.xlabel(r"\(x\)")
  plt.title("Random target functions")
  plt.gca().xaxis.set_label_coords(0.5, -0.05)

  plt.sca(axes[0, 0])
  plt.gca().set_xticks(
    [
      0,
      1,
      2,
    ],
    enames,
  )

  def change_alpha(handle, original):
    """Change the alpha and marker style of the legend handles"""
    handle.update_from(original)
    handle.set_alpha(1)

  leg = plt.legend(
    ncol=5,
    loc="lower center",
    borderaxespad=0.2,
    columnspacing=0.5,
    handletextpad=0.1,
    handler_map={plt.Line2D: HandlerLine2D(update_func=change_alpha)},
  )
  leg.get_frame().set_linewidth(0.0)

  plt.yscale("log")
  plt.ylim([8 * 10**-8, 8 * 10**1])
  plt.ylabel("MSE")
  plt.title("(b)", loc="left")
  plt.title("Performance on random tasks")


def main():
  os.chdir(Path(__file__).parent.parent / "./experiments/function_approx")

  sys.argv.extend(["-d", "./definition.py"])

  _, cfg = qrc.ui.setup()

  plt.rcParams.update(util.get_default_params())

  plt.rcParams.update({
    "figure.titlesize": util.FONT_NORMAL,  # fontsize of the figure title
    "font.size": util.FONT_NORMAL,  # controls default text sizes
    "axes.titlesize": util.FONT_MEDIUM,  # fontsize of the axes title
    "axes.labelsize": util.FONT_MEDIUM,  # fontsize of the x and y labels
    "xtick.labelsize": util.FONT_MEDIUM,  # fontsize of the tick labels
    "ytick.labelsize": util.FONT_MEDIUM,  # fontsize of the tick labels
    "legend.fontsize": util.FONT_MEDIUM,
  })

  ip = 0.3

  plt.rcParams["figure.figsize"][0] = 170 * util.INpMM
  plt.rcParams["figure.figsize"][1] = 6
  fig = plt.figure()

  axes, _ = util.default_grid_axes(
    fig,
    [1, 2],
    outpads=[0.3, 0.1, 0, 0.1],
    inpads=[ip + 0.13, ip, ip, ip],
    axkwargs={},
    get_div=True,
  )

  draw_figure(axes, cfg)

  fig.savefig(
    (Path(__file__).parent / "rand_interp").with_suffix(".pdf"),
    transparent=True,
    pad_inches=0,
  )


if __name__ == "__main__":
  main()
