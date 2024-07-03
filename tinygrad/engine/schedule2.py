from typing import Dict, List, Tuple

from tinygrad.engine.schedule import ScheduleItem
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.symbolic import Variable

def create_schedule(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  schedule: List[ScheduleItem] = []
  return schedule, {}
