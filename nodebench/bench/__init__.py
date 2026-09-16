"""Workload modules.

Contract: a bench module runs something and returns a plain dict. It never
prints a report, never writes HTML, never decides what a number means. The
report layer formats; the analyze layer interprets.
"""

from .base import BenchResult, cuda_timer  # noqa: F401
