from __future__ import annotations
import functools, pickle, atexit
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, cast

from tinygrad.engine.schedule import ScheduleItem
from tinygrad.helpers import DEBUG, SAVE_SCHEDULE, colored, getenv, prod
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import BufferOps, ConstBuffer, LazyOp, LoadOps, MemBuffer, Op, ReduceOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

def lower_lazybuffer(out:LazyBuffer, global_stores:Dict[LazyBuffer, None]) -> Tuple[LazyOp, List[LazyBuffer]]:
  if out.op in {LoadOps.CUSTOM, LoadOps.COPY, LoadOps.EMPTY, LoadOps.VIEW}: return LazyOp(out.op, (), out.arg), list(out.srcs)
  inputs: Dict[LazyBuffer, int] = {}
  @functools.lru_cache(None)
  def _dfs(x:LazyBuffer, st:ShapeTracker, output_st:ShapeTracker):
    if x.op is LoadOps.CONST: return LazyOp(BufferOps.CONST, (), ConstBuffer(x.arg, x.dtype, st))
    if out.op is LoadOps.ASSIGN and x is out.srcs[1]: return LazyOp(BufferOps.LOAD, (), MemBuffer(0, x.dtype, st))
    if x is not out and (x in global_stores or x.realized is not None):
      return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.setdefault(x, len(inputs)+1), x.dtype, st))
    if x.op in ReduceOps: output_st, st = x.st, ShapeTracker.from_shape(x.srcs[0].shape)
    if x.op is LoadOps.ASSIGN and len(out.arg) != 0: output_st = out.arg[0]
    src = tuple(_dfs(s.base, st if s.base is s else s.st+st, output_st) for s in x.srcs)
    lop = src[0] if x.op in {LoadOps.CONTIGUOUS, LoadOps.ASSIGN} else LazyOp(cast(Op,x.op), src, x.arg)
    return LazyOp(BufferOps.STORE, (lop, ), MemBuffer(0, x.dtype, output_st)) if x is out else lop
  return _dfs(out, out.st, out.st), list(inputs)

SCHEDULES: List = []
def create_schedule(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  global_stores = {x.base:None for x in outs if x.base.op is not LoadOps.CONST and x.base.realized is None}
  if DEBUG >= 2: print(colored(f"scheduling {list(outs)}", "magenta"))
  assign_targets: Dict[LazyBuffer, LazyBuffer] = {}
  @functools.lru_cache(None)
  def _dfs_store(x:LazyBuffer):
    if x.base.realized is not None or x.base.op is LoadOps.CONST: return
    if x is not x.base:
      if prod(x.base.shape) < prod(x.shape): global_stores[x.base] = None
      return _dfs_store(x.base)
    for s in x.srcs: _dfs_store(s)
    if x.op in LoadOps or x.op in ReduceOps or x.forced_realize: global_stores[x] = None
    if x.op is LoadOps.VIEW and x.srcs[0].base.realized is None: global_stores[x.srcs[0].base] = None
    if x.op is LoadOps.ASSIGN: assign_targets[x.srcs[1]] = x

  for x in outs: _dfs_store(x)

  rev_children = {x:lower_lazybuffer(x, global_stores) for x in global_stores}
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  in_degree: DefaultDict[LazyBuffer, int] = defaultdict(int)
  for buf, (_, inputs) in rev_children.items():
    for x in inputs:
      if x in assign_targets and assign_targets[x] is not buf:
        children[buf][assign_targets[x]] = None
        in_degree[assign_targets[x]] += 1
      elif x.realized is None:
        children[x][buf] = None
        in_degree[buf] += 1

  def graph_rewrite(n:LazyBuffer) -> Tuple[Tuple[LazyOp, ...], List[LazyBuffer]]:
    lop, inputs = rev_children[n]
    return (lop, ), inputs

  queue = [x for x in global_stores if in_degree[x] == 0]
  schedule: List[ScheduleItem] = []
  while queue:
    n = queue.pop(0)
    del n.srcs
    ast, inputs = graph_rewrite(n)
    schedule.append(ScheduleItem(ast, (n.buffer,)+tuple(x.buffer for x in inputs)))
    if getenv("DEBUG_TOPOSORT"): print(colored(n, "green"))
    for x in children[n]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  if SAVE_SCHEDULE:
    def _save():
      print(f"saving {len(SCHEDULES)} schedule graphs to", fp:=getenv("SAVE_SCHEDULE_PATH", "schedule.pkl"))
      with open(fp, "wb") as f: pickle.dump(SCHEDULES, f)
    if len(SCHEDULES) == 0: atexit.register(_save)
    SCHEDULES.extend([(children, rev_children)])

  if len(schedule) != len(global_stores) or any(d != 0 for d in in_degree.values()):
    raise RuntimeError(f"cycle detected in graph {len(schedule)} != {len(global_stores)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, {}
