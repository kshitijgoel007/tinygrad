from collections import defaultdict
import functools
from typing import DefaultDict, Dict, List, Tuple, cast

from tinygrad.engine.graph import print_tree
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.helpers import DEBUG, colored, prod
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import BufferOps, ConstBuffer, LazyOp, LoadOps, MemBuffer, Op, ReduceOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

def lower_lazybuffer(out:LazyBuffer, global_stores:Dict[LazyBuffer, None]) -> Tuple[LazyOp, List[LazyBuffer]]:
  if out.op in {LoadOps.CUSTOM, LoadOps.COPY, LoadOps.EMPTY, LoadOps.VIEW}: return LazyOp(out.op, (), out.arg), list(out.srcs)
  inputs: Dict[LazyBuffer, int] = {}
  @functools.lru_cache(None)
  def _dfs(x:LazyBuffer, st:ShapeTracker):
    nonlocal output_st
    if x != x.base:
      st = x.st + st
      x = x.base
    if x.op is LoadOps.CONST: return LazyOp(BufferOps.CONST, (), ConstBuffer(x.arg, x.dtype, st))
    if out.op is LoadOps.ASSIGN and x is out.srcs[1]: return LazyOp(BufferOps.LOAD, (), MemBuffer(0, x.dtype, st))
    if x is not out and (x in global_stores or x.realized is not None):
      return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.setdefault(x, len(inputs)+1), x.dtype, st))
    if x.op in ReduceOps:
      assert x is out, f"{x} != {out}"
      output_st = x.st
      st = ShapeTracker.from_shape(x.srcs[0].shape)
    if x.op is LoadOps.ASSIGN:
      assert x is out
      assert x.srcs[1].base is x.srcs[1], "assign must be to base"
      assert x.srcs[1].realized is not None, f"assign must be already realized to schedule {x.srcs[1]}"
      if len(out.arg) != 0: output_st = cast(ShapeTracker, out.arg[0])
    lop = _dfs(x.srcs[0], st) if x.op in {LoadOps.CONTIGUOUS, LoadOps.ASSIGN} else LazyOp(cast(Op, x.op), tuple(_dfs(s, st) for s in x.srcs), x.arg)
    if x is out: lop = LazyOp(BufferOps.STORE, (lop, ), MemBuffer(0, x.dtype, output_st))
    return lop
  return _dfs(out, output_st:=out.st), list(inputs)

def create_schedule(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  if DEBUG >= 2: print(colored(f"scheduling {outs}", "green"))
  global_stores = {x.base:None for x in outs if x.base.op is not LoadOps.CONST and x.base.realized is None}
  assign_targets: Dict[LazyBuffer, LazyBuffer] = {}
  @functools.lru_cache(None)
  def _dfs(x:LazyBuffer):
    if x.base.realized is not None or x.base.op is LoadOps.CONST: return
    if x is not x.base:
      if prod(x.base.shape) < prod(x.shape): global_stores[x.base] = None
      return _dfs(x.base)
    for s in x.srcs: _dfs(s)
    if x.op in LoadOps or x.forced_realize: global_stores[x] = None
    if x.op in ReduceOps: global_stores[x] = None
    if x.op is LoadOps.ASSIGN: assign_targets[x.srcs[1]] = x
  for x in outs: _dfs(x)

  rev_children = {x:lower_lazybuffer(x, global_stores) for x in global_stores}
  # *** TODO: graph rewrite asts in rev_children
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  in_degree: DefaultDict[LazyBuffer, int] = defaultdict(int)
  for buf, (_, inputs) in rev_children.items():
    in_degree[buf] = 0
    for x in inputs:
      if x in assign_targets and assign_targets[x] is not buf:
        children[buf][assign_targets[x]] = None
        in_degree[assign_targets[x]] += 1
      if x.realized is None:
        children[x][buf] = None
        in_degree[buf] += 1

  queue = [x for x in global_stores if in_degree[x] == 0]
  schedule: List[ScheduleItem] = []
  while queue:
    n = queue.pop(0)
    del n.srcs
    lop, inputs = rev_children[n]
    if DEBUG >= 4:
      print(colored(n, "green"))
      print_tree(lop)
      print("--")
    schedule.append(ScheduleItem((lop, ), (n.buffer, )+tuple(x.buffer for x in inputs)))
    for x in children[n]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  if len(schedule) != len(global_stores): raise RuntimeError(f"cycle detected in graph {len(schedule)} != {len(global_stores)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, {}
