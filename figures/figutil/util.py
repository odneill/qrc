import os
import warnings

import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size

HAS_LATEX = os.system("pdflatex --version 2>/dev/null 1>&2") == 0
if not HAS_LATEX:
  warnings.warn("No latex found, using non-latex mode")

FONT_TINY = 6
FONT_SMALL = 8
FONT_MEDIUM = 10
FONT_NORMAL = 12


# Unit conversions
MMpIN = 25.4
INpMM = 1 / MMpIN
PTpIN = 72
INpPT = 1 / PTpIN
PTpMM = PTpIN * INpMM
MMpPT = 1 / PTpMM


def get_default_params():
  # A4 21cm - 5.5cm total margins 15.5cm to inches
  # 3.6cm top bottom total on 297mm page gives max height 261mm
  fig_size = [6.102362, 3.1]
  fig_size = [155 / 25.4, 3.1]

  preamble = r"""
    \usepackage{amsmath}
    \usepackage{bm}
    \usepackage{anyfontsize}
    %\usepackage{scalefnt}
    \usepackage{ifxetex}
    \ifxetex
        % XeLaTeX
        % \usepackage{polyglossia}
        % \setmainlanguage[spelling=new,babelshorthands=true]{german}
        \usepackage{fontspec}
        % \usepackage[]{unicode-math}
        \setmainfont{CMU Serif}
        % \setmathfont{Latin Modern Math}
        % \setmathfont{latinmodern-math.otf}
    \else
        % default: pdfLaTeX
        % \usepackage[T1]{fontenc}
        % \usepackage{lmodern}
        % \usepackage[utf8]{inputenc}
    \fi
    """

  params = {
    "figure.dpi": 100,  # DPI of the figure
    "figure.figsize": fig_size,  # inches on screen
    "axes.linewidth": 0.8,  # default 0.8
    "grid.linewidth": 0.8,  # default 0.8
    "lines.linewidth": 1,  # default 1.5
    "patch.linewidth": 1,  # default 1.0
    "figure.titlesize": FONT_NORMAL,  # fontsize of the figure title
    "font.size": FONT_NORMAL,  # controls default text sizes
    "axes.titlesize": FONT_NORMAL,  # fontsize of the axes title
    "axes.labelsize": FONT_NORMAL,  # fontsize of the x and y labels
    "xtick.labelsize": FONT_TINY,  # fontsize of the tick labels
    "ytick.labelsize": FONT_TINY,  # fontsize of the tick labels
    "legend.fontsize": FONT_TINY,  # legend fontsize
    "text.usetex": HAS_LATEX,  # Use LaTeX
    "text.latex.preamble": preamble,
    "pgf.texsystem": "xelatex",
    "pgf.preamble": preamble,
    "pdf.fonttype": 42,  # 3 or 42, 3 can cause issues with nature
    "font.family": "serif",  # Font family
    "svg.fonttype": "none",  # or path, whether text is preserved.
  }

  return params


def default_grid_axes(
  fig,
  dims,
  *,
  outpads=(0, 0),
  inpads=(0, 0),
  axkwargs=None,
  get_div=False,
  postfunc=None,
):
  """
  left, top, right, bottom

  """

  if len(outpads) == 2:
    op = list(outpads) * 2
  else:
    assert len(outpads) == 4
    op = list(outpads)
  if len(inpads) == 2:
    ip = list(inpads) * 2
  else:
    assert len(inpads) == 4
    ip = list(inpads)

  h = [
    Size.Fixed(op[0]),
    *(
      [
        Size.Fixed(ip[0]),
        # Size.Fixed(hw),
        Size.Scaled(1),
        Size.Fixed(ip[2]),
      ]
      * dims[0]
    ),
    Size.Fixed(op[2]),
  ]
  v = [
    Size.Fixed(op[3]),
    *(
      [
        Size.Fixed(ip[3]),
        # Size.Fixed(vw),
        Size.Scaled(1),
        Size.Fixed(ip[1]),
      ]
      * dims[1]
    ),
    Size.Fixed(op[1]),
  ]

  if postfunc is not None:
    h, v = postfunc(h, v)

  # Divides up figure space in h and v
  # Each split point (h,v) splits region
  divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

  axkwargs = {} if axkwargs is None else axkwargs

  axes = np.zeros(dims, dtype=object)
  for i in range(dims[0]):
    for j in range(dims[1]):
      ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=3 * i + 2, ny=3 * j + 2),
        **axkwargs,
      )
      axes[i, j] = ax

  if get_div:
    return axes, divider
  return axes
