import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figutil import util
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerLine2D

import qrc.ui

HALFCOL = True


def change_alpha(handle, original):
  """Change the alpha and marker style of the legend handles"""
  handle.update_from(original)
  handle.set_alpha(1)


def draw_figure(axes, cfg):
  loaded = np.load(
    cfg["indir"] / "task_interp_data.npz",
    allow_pickle=True,
  )
  data = loaded["data"].ravel()[0]

  msess = np.zeros([10, 10])
  iundss = np.zeros([10, 10])

  def filled_marker_style(col):
    return {
      "marker": "o",
      "linestyle": "None",
      "markersize": cfg["marker_size"],
      "color": col,
      "markerfacecolor": col,
      "markerfacecoloralt": col,
      "markeredgecolor": "none",
    }

  for i, (_, state) in enumerate(list(data.items())[::-1]):
    for _, (ename, enc) in enumerate(state.items()):
      if ename != "spiral":
        continue
      for k, (tname, tasks) in enumerate(enc.items()):
        if tname not in ("sinc", "rect"):
          continue
        k = ("sinc", "rect").index(tname)

        mses = []
        Rcs = []
        for p, (_, task) in enumerate(tasks.items()):
          svs = task["svs2"]
          svs = svs / svs.sum()
          Rc = int((svs > 3 * 10**-7).sum())
          mses.append(task["mse"])
          Rcs.append(Rc)

        mses = np.array(mses)
        Rcs = np.array(Rcs)
        msess[i + 5 * k, :] = mses
        iundss[i + 5 * k, :] = np.argsort(mses)

        if i == 0:
          ax = axes[k, -1]
          ax.plot(task["gtx"], task["gty"], "#1f77b4", label=r"\(f(x)\)")

        ax: Axes = axes[k, i]
        for p, (_, task) in enumerate(tasks.items()):
          if p == 0:
            l1 = r"\(f(x\){" + cfg["legend_subfont"] + "\\textsubscript{TEST}}\\()\\)"
            l1 = r"\(f(x)\)"
          else:
            l1 = "_nolegend_"
          ax.plot(
            task["testx"],
            task["testy"],
            **filled_marker_style("k"),
            label=l1,
            zorder=1000,
            alpha=0.1,
          )

        ax.plot(
          task["gtx"],
          task["gty"],
          "#1f77b4",
          label="_nolegend_",
          linewidth=0.5,
          zorder=-2,
          alpha=0.5,
        )

        ind = np.argsort(mses)[0]
        task = list(tasks.values())[ind]

        l3 = r"\(f\){" + cfg["legend_subfont"] + "\\textsubscript{BEST}}\\((x)\\)"
        ax.plot(
          task["testx"],
          task["testy"],
          **filled_marker_style("red"),
          label=l3,
          zorder=1001,
        )

        ax.set_title(f"{Rcs.mean():.1f}, {mses.mean():.3f}", y=cfg["title_height"])

        leg = ax.legend(
          loc="upper left",
          handlelength=0.8,
          bbox_to_anchor=(0, 1),
          borderaxespad=0.2,
          ncol=2,
          frameon=True,
          columnspacing=0.5,
          handletextpad=0.1,
          borderpad=0.1,
          handler_map={plt.Line2D: HandlerLine2D(update_func=change_alpha)},
        )
        leg.set_zorder(2000)
        leg.get_frame().set_linewidth(0.0)

  for ax in axes.ravel():
    ax.set_xticks([0, 1])
    ax.tick_params(**cfg["tick_params"])
    ax.set_yticks([0, 1])
    ax.set_ylim([-0.4, 1.4])


def main():
  os.chdir(Path(__file__).parent.parent / "./experiments/function_approx")

  sys.argv.extend(["-d", "./definition.py"])

  _, cfg = qrc.ui.setup()

  plt.rcParams.update(util.get_default_params())

  plt.rcParams.update({
    "xtick.labelsize": util.FONT_MEDIUM,  # fontsize of the tick labels
    "ytick.labelsize": util.FONT_MEDIUM,  # fontsize of the tick labels
    "legend.fontsize": util.FONT_MEDIUM,
  })
  cfg["legend_subfont"] = "\\fontsize{8pt}{8pt}"
  cfg["title_height"] = 1
  cfg["marker_size"] = 2
  cfg["tick_params"] = {"length": 2}

  p = 0.5
  ip = 0.2
  pv = p
  plt.rcParams["figure.figsize"][0] = 170 * util.INpMM
  plt.rcParams["figure.figsize"][1] = 6 * (0.475 + 2 * ip) + 2 * pv

  if HALFCOL:
    plt.rcParams.update({
      "figure.titlesize": util.FONT_TINY + 1,  # fontsize of the figure title
      "font.size": util.FONT_TINY + 1,  # controls default text sizes
      "axes.titlesize": util.FONT_TINY + 1,  # fontsize of the axes title
      "axes.labelsize": util.FONT_TINY + 1,  # fontsize of the x and y labels
      "legend.fontsize": 4,
    })
    cfg["legend_subfont"] = "\\fontsize{2pt}{2pt}"
    cfg["title_height"] = 0.8
    cfg["marker_size"] = 1
    cfg["tick_params"] = {"length": 2, "pad": 1}

    p = 0.5 / 2
    ip = 0.2 / 2
    pv = p
    plt.rcParams["figure.figsize"][0] = ((170 - 8) / 2) * util.INpMM
    plt.rcParams["figure.figsize"][1] = 6 * (0.5 / 2 + 2 * ip) + 2 * pv

  fig = plt.figure()

  axes, _ = util.default_grid_axes(
    fig,
    [2, 6],
    outpads=[0.3, 0.1, 0, 0.1],
    inpads=[ip + 0.13, ip, ip, ip],
    axkwargs={},
    get_div=True,
  )

  lbls = [
    "Coherent\nIntensity",
    "Dist.",
    "Coherent\nPNR",
    "Hybrid",
    "Fock",
    "Target\n\\(f^*(x)\\)",
  ]

  for i in range(6):
    axes[0, i].set_ylabel(lbls[i])

  for i in range(2):
    axes[i, -1].set_title(
      "\\(\\langle R_c\\rangle\\), \\(\\langle \\mathrm{MSE}\\rangle\\)"
    )

  axes[0, 0].set_xlabel(r"\(x\)")
  axes[0, 0].xaxis.set_label_coords(0.5, -0.2)
  axes[1, 0].set_xlabel(r"\(x\)")
  axes[1, 0].xaxis.set_label_coords(0.5, -0.2)

  draw_figure(axes, cfg)

  if HALFCOL:
    fig.savefig(
      (Path(__file__).parent / "interp_hw").with_suffix(".pdf"),
      transparent=True,
      pad_inches=0,
    )
  else:
    fig.savefig(
      (Path(__file__).parent / "interp_fw").with_suffix(".pdf"),
      transparent=True,
      pad_inches=0,
    )


if __name__ == "__main__":
  main()
