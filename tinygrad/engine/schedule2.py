from __future__ import annotations
import functools, pickle, atexit
from collections import defaultdict
from dataclasses import replace
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
      del buf.srcs

  def graph_rewrite(n:LazyBuffer) -> Tuple[List[LazyOp], List[LazyBuffer], List[LazyBuffer]]:
    (lop, inputs) = rev_children[n]
    tr_next: List[LazyBuffer] = []
    for tr in children[n]:
      in_degree[tr] -= 1
      if in_degree[tr] == 0: tr_next.append(tr)
    if n.op in LoadOps: return [lop], [n]+inputs, tr_next
    outputs = [n]
    for tr in tr_next.copy():
      if tr.op in LoadOps: continue
      # TODO: real matcher!
      outputs.append(tr)
      tr_next.remove(tr)
    if len(outputs) == 1: return [lop], [n]+inputs, tr_next
    ast: List[LazyOp] = []
    allbufs = {x:i for i,x in enumerate(outputs)}
    if reduceops:=[x for x in outputs if x.op in ReduceOps]:
      output_st = ShapeTracker.from_shape(reduceops[0].st.shape)
    else: output_st = ShapeTracker.from_shape(outputs[0].shape)
    for i,out in enumerate(outputs):
      print(f"-> {out.op}")
      lop, inputs = rev_children[out]
      @functools.lru_cache(None)
      def _recursive_rewrite(lop:LazyOp, st:ShapeTracker) -> LazyOp:
        src, arg = tuple([_recursive_rewrite(x, st) for x in lop.src]), lop.arg
        if lop.op in BufferOps:
          if lop.op is BufferOps.LOAD:
            if (buf:=inputs[lop.arg.idx-1]) in outputs: return _recursive_rewrite(rev_children[buf][0], st)
            arg = replace(lop.arg, idx=allbufs.setdefault(buf, len(allbufs)), st=st)
          if lop.op is BufferOps.STORE: arg = replace(lop.arg, idx=i, st=st)
          else: arg = replace(lop.arg, st=st)
        return replace(lop, src=src, arg=arg)
      lop = _recursive_rewrite(lop, output_st)
      ast.append(lop)
    return ast, list(allbufs), tr_next

  queue = [x for x in global_stores if in_degree[x] == 0]
  schedule: List[ScheduleItem] = []
  while queue:
    n = queue.pop(0)
    ast, allbufs, tr_next = graph_rewrite(n)
    schedule.append(ScheduleItem(tuple(ast), tuple([x.buffer for x in allbufs])))
    queue.extend(tr_next)

  if SAVE_SCHEDULE:
    def _save():
      print(f"saving {len(SCHEDULES)} schedule graphs to", fp:=getenv("SAVE_SCHEDULE_PATH", "schedule.pkl"))
      with open(fp, "wb") as f: pickle.dump(SCHEDULES, f)
    if len(SCHEDULES) == 0: atexit.register(_save)
    SCHEDULES.extend([(children, rev_children)])

  if left_out:=dict(filter(lambda x:x[1]!=0, in_degree.items())): raise RuntimeError(f"some realizes never realized: {left_out}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, {}
