import warnings
from pathlib import Path


def parse_css_line(css):
  if "--" not in css:
    return None
  if "#" not in css:
    return None
  if "/*" == css.strip(" ")[:2]:
    return None
  css = css.strip("\n").strip(" ").split(" ")
  css = [c for c in css if c != ""]
  chunk = css[0].strip("-").strip(":").split("-")
  classname = chunk[0]
  name = chunk[1]
  color = css[1].strip(";")

  return classname, name, color


def _gen_colours():
  with open(Path(__file__).parent / "colours.css") as f:
    css = f.readlines()

  colours = {}
  for line in css:
    out = parse_css_line(line)
    if out is not None:
      classname, name, color = out
      if classname not in colours:
        colours[classname] = {}
      if name in colours[classname]:
        warnings.warn(f"Duplicate color, skipping: {name} in {classname}")
      else:
        colours[classname][name] = color

  return colours


# ----------------------------- Set colour lookup ---------------------------- #

colours = _gen_colours()
