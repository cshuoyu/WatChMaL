# prof_utils.py
import time
import contextlib
from collections import defaultdict
import torch

ENABLE_PROFILING = True
_TIMES = defaultdict(float)
_COUNTS = defaultdict(int)

def _now():
    return time.perf_counter()

@contextlib.contextmanager
def section(name, cuda=True):
    if not ENABLE_PROFILING:
        yield
        return
    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = _now()
    try:
        yield
    finally:
        if cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = _now() - t0
        _TIMES[name] += dt
        _COUNTS[name] += 1

def reset():
    _TIMES.clear()
    _COUNTS.clear()

def report(topk=None):
    items = [(k, _TIMES[k], _COUNTS[k], _TIMES[k] / max(_COUNTS[k],1)) for k in _TIMES]
    items.sort(key=lambda x: x[1], reverse=True)
    lines = []
    total = sum(t for _, t, _, _ in items) or 1.0
    lines.append(f"{'Stage':35s}  {'Total(s)':>9}  {'Calls':>5}  {'Avg(ms)':>9}  {'%':>6}")
    for i,(k,t,c,avg) in enumerate(items):
        if topk is not None and i >= topk: break
        lines.append(f"{k:35s}  {t:9.4f}  {c:5d}  {avg*1000:9.3f}  {t/total*100:6.2f}")
    return "\n".join(lines)
