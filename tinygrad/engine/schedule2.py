from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from enum import EnumType
import functools
from typing import Callable, DefaultDict, Dict, List, Optional, Set, Tuple, cast

from tinygrad.engine.graph import print_tree
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.helpers import DEBUG, colored, getenv, prod
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import BinaryOps, BufferOps, ConstBuffer, LazyOp, LoadOps, MemBuffer, Op, ReduceOps, TernaryOps, UnaryOps
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
    if x is not out and (x in global_stores or x.realized is not None): return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.setdefault(x, len(inputs)+1), x.dtype, st))
    if x.op in ReduceOps: output_st, st = x.st, ShapeTracker.from_shape(x.srcs[0].shape)
    if x.op is LoadOps.ASSIGN and len(out.arg) != 0: output_st = cast(ShapeTracker, out.arg[0])
    lop = _dfs(x.srcs[0], st) if x.op in {LoadOps.CONTIGUOUS, LoadOps.ASSIGN} else LazyOp(cast(Op, x.op), tuple(_dfs(s, st) for s in x.srcs), x.arg)
    return LazyOp(BufferOps.STORE, (lop, ), MemBuffer(0, x.dtype, output_st)) if x is out else lop
  return _dfs(out, output_st:=out.st), list(inputs)

@dataclass(frozen=True)
class OpPat:
  op: Optional[EnumType|Set[EnumType]] = None
  name: Optional[str] = None
  children: Optional[OpPat] = None
  #parents: Optional[OpPat] = None

class PatternMatcher:
  def __init__(self, patterns:List[Tuple[OpPat, Callable]], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], rev_children:Dict[LazyBuffer, Tuple[LazyOp, List[LazyBuffer]]]):
    self.patterns, self.children, self.rev_children = patterns, children, rev_children

def _fuse_elementwise_child(r:LazyBuffer, e:LazyBuffer, ctx:PatternMatcher):
  output_st = r.st
  r_ast, r_inputs = ctx.rev_children[r]
  e_ast, e_inputs = ctx.rev_children[e]
  print_tree(r_ast)
  print_tree(e_ast)
  replace_idxs = {}
  raise Exception()

elementwise_ops = {UnaryOps, BinaryOps, TernaryOps}
kernel_fusion = [
    (OpPat(ReduceOps, "r", children=OpPat(elementwise_ops, "e"), ), lambda r,e,ctx: _fuse_elementwise_child(r, e, ctx)),
]

def create_schedule(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  if DEBUG >= 3: print(colored(f"scheduling {outs}", "yellow"))
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
      elif x.realized is None:
        children[x][buf] = None
        in_degree[buf] += 1

  """
  def _match(op:LazyBuffer, children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], pat:OpPat, store:Dict[str, LazyBuffer]) -> bool:
    if pat.name is not None and store.setdefault(pat.name, op) is not op: return False
    if pat.op is not None and (not any(op.op in opset for opset in pat.op) if isinstance(pat.op, set) else op.op not in pat.op): return False
    if pat.children is None: return True
    #for x in children[op]: print(_match(x, children, pat.children, store))
    new_store = store.copy()
    if all(_match(x, children, pat.children, store) for x in children[op]):
      store = new_store
      return True
    return False
  ctx = PatternMatcher(kernel_fusion, children, rev_children)
  for out, chld in children.items():
    pattern = kernel_fusion[0]
    print(pattern, out)
    x = _match(out, children, pattern[0], store:={})
    if x: ret = pattern[1](**store, ctx=ctx)
  """

  queue = [x for x in global_stores if in_degree[x] == 0]
  schedule: List[ScheduleItem] = []
  while queue:
    n = queue.pop(0)
    if getenv("DEBUG_TOPOSORT"): print(colored(n, "green"))
    del n.srcs
    lop, inputs = rev_children[n]
    if DEBUG >= 4:
      print_tree(lop)
      print("--")
    schedule.append(ScheduleItem((lop, ), (n.buffer, )+tuple(x.buffer for x in inputs)))
    for x in children[n]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  if len(schedule) != len(global_stores) or any(d != 0 for d in in_degree.values()): raise RuntimeError(f"cycle detected in graph {len(schedule)} != {len(global_stores)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, {}
