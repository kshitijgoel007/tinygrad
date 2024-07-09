#!/usr/bin/env python
import unittest
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, BufferOps
from tinygrad.nn import Conv2d

class TestConvShapetracker(unittest.TestCase):
  def test_conv_3x3_one_view(self):
    conv = Conv2d(16, 32, (3, 3))

    # first run to init the weights, they are scheduled
    conv(Tensor.empty(1, 16, 10, 10)).schedule()
    # run it again to get the kernels
    sched = [si for si in conv(Tensor.empty(1, 16, 10, 10)).schedule() if si.ast[0].op not in LoadOps]
    assert len(sched) == 1, f"conv should only have one kernel, getting {len(sched)}"
    for st in [x.arg.st for x in sched[0].ast[0].lazyops if x.op is BufferOps.LOAD]:
      assert len(st.views) == 1

if __name__ == '__main__':
  unittest.main()
