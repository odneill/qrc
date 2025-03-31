import shutil
import sys
from glob import glob

import pytest
from absl.testing import absltest

import qrc.tools


class TestIntegration1(absltest.TestCase):
  def test_1(self):
    sys.argv = [
      "dummy.py",
      "--definition",
      "./_int_definition.py",
      "--datadir",
      "./_int_test/",
      "--outdatadir",
      "./_int_test/",
      "--logfile",
      "./_int_test/out.log",
    ]
    qrc.tools.do_optics_simulation()
    sys.argv = [
      "dummy.py",
      "--definition",
      "./_int_definition.py",
      "--datadir",
      "./_int_test/",
      "--outdatadir",
      "./_int_test/",
      "--logfile",
      "./_int_test/out.log",
    ]
    qrc.tools.do_postprocess()
    sys.argv = [
      "dummy.py",
      "--definition",
      "./_int_definition.py",
      "--datadir",
      "./_int_test/",
      "--outdatadir",
      "./_int_test/",
      "--logfile",
      "./_int_test/out.log",
    ]
    qrc.tools.do_task_eval()

    files = glob("./_int_test/*.tasks.npz")
    assert len(files) == 15

    shutil.rmtree("./_int_test/")


if __name__ == "__main__":
  pytest.main([__file__])
