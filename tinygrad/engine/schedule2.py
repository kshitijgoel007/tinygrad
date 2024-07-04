from collections import defaultdict
import functools
from typing import DefaultDict, Dict, List, Tuple, cast

from tinygrad.engine.graph import print_tree
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.helpers import DEBUG, prod
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import BufferOps, ConstBuffer, LazyOp, LoadOps, MemBuffer, Op, ReduceOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

def lower_lazybuffer(out:LazyBuffer, global_stores:Dict[LazyBuffer, None]) -> Tuple[LazyOp, List[LazyBuffer]]:
  if out.op in {LoadOps.CUSTOM, LoadOps.COPY, LoadOps.EMPTY, LoadOps.VIEW}: return LazyOp(out.op, (), out.arg), list(out.srcs)
  inputs: List[LazyBuffer] = []
  output_st = out.st
  @functools.lru_cache(None)
  def _dfs(x:LazyBuffer, st:ShapeTracker):
    nonlocal output_st
    if x != x.base:
      st = x.st + st
      x = x.base
    if x.op is LoadOps.CONST: return LazyOp(BufferOps.CONST, (), ConstBuffer(x.arg, x.dtype, st))
    if x is not out and (x in global_stores or x.realized is not None):
      assert x not in inputs
      inputs.append(x)
      return LazyOp(BufferOps.LOAD, (), MemBuffer(len(inputs), x.dtype, st))
    if x.op in ReduceOps:
      assert st.contiguous, "ReduceOps late fusion must be contiguous"
      output_st = st
      st = ShapeTracker.from_shape(x.srcs[0].shape)
    lop = _dfs(x.srcs[0], st) if x.op in {LoadOps.CONTIGUOUS, LoadOps.ASSIGN} else LazyOp(cast(Op, x.op), tuple(_dfs(s, st) for s in x.srcs), x.arg)
    if x is out: lop = LazyOp(BufferOps.STORE, (lop, ), MemBuffer(0, x.dtype, output_st))
    return lop
  return _dfs(out, out.st), inputs

def create_schedule(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  global_stores = {x.base:None for x in outs if x.base.op is not LoadOps.CONST and x.base.realized is None}
  @functools.lru_cache(None)
  def _dfs(x:LazyBuffer):
    if x.base.realized is not None: return
    if x is not x.base:
      if prod(x.base.shape) < prod(x.shape) and x.base.op is not LoadOps.CONST: return global_stores.setdefault(x.base, None)
      return _dfs(x.base)
    for s in x.srcs: _dfs(s)
    if (x.op in LoadOps or x.forced_realize) and x.op is not LoadOps.CONST: global_stores.setdefault(x, None)
    if x.op in ReduceOps: global_stores.setdefault(x, None)
  for x in outs: _dfs(x)

  rev_children = {x:lower_lazybuffer(x, global_stores) for x in global_stores}
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  in_degree: DefaultDict[LazyBuffer, int] = defaultdict(int)
  for buf, (_, inputs) in rev_children.items():
    in_degree[buf] = 0
    for x in inputs:
      if x.realized is not None: continue
      children[x][buf] = None
      in_degree[buf] += 1

  queue = [x for x in global_stores if in_degree[x] == 0]
  schedule: List[ScheduleItem] = []
  while queue:
    n = queue.pop(0)
    del n.srcs
    lop, inputs = rev_children[n]
    if DEBUG >= 3:
      print_tree(lop)
      print("---")
    schedule.append(ScheduleItem((lop, ), (n.buffer, )+tuple(x.buffer for x in inputs)))
    for x in children[n]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  assert len(schedule) == len(global_stores), f"{len(schedule)} != {len(global_stores)}"
  return schedule, {}
