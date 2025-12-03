'''
Author: Shuoyu Chen shuoyuchen.physics@gmail.com
Date: 2025-09-19 11:39:34
LastEditors: Shuoyu Chen shuoyuchen.physics@gmail.com
LastEditTime: 2025-09-19 11:39:34
FilePath: /schen/workspace/WatChMaL/prof_utils.py
Description: 
'''
# prof_utils.py
import time
import contextlib
from collections import defaultdict
import torch

ENABLE_PROFILING = True  # 想关就置 False

_TIMES = defaultdict(float)
_COUNTS = defaultdict(int)

def _sync(cuda=True):
    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

def _now():
    return time.perf_counter()

@contextlib.contextmanager
def section(name, cuda=True):
    if not ENABLE_PROFILING:
        yield
        return
    _sync(cuda)
    t0 = _now()
    try:
        yield
    finally:
        _sync(cuda)
        dt = _now() - t0
        _TIMES[name] += dt
        _COUNTS[name] += 1

def reset():
    _TIMES.clear()
    _COUNTS.clear()

def report(topk=None):
    items = [(k, _TIMES[k], _COUNTS[k], _TIMES[k] / max(_COUNTS[k],1)) for k in _TIMES]
    items.sort(key=lambda x: x[1], reverse=True)
    total = sum(t for _, t, _, _ in items) or 1.0
    lines = []
    lines.append(f"{'Stage':40s} {'Total(s)':>10} {'Calls':>6} {'Avg(ms)':>10} {'%':>6}")
    for i,(k,t,c,avg) in enumerate(items):
        if topk is not None and i >= topk: break
        lines.append(f"{k:40s} {t:10.4f} {c:6d} {avg*1000:10.3f} {t/total*100:6.2f}")
    return "\n".join(lines)
