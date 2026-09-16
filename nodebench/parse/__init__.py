"""Parsers. Every function here is pure: str in, dict out, no IO.

That is not stylistic. It means each one can be unit-tested against a saved
snippet of real tool output, so a change in an upstream tool's format is caught
by a test instead of by a wrong number in a report.
"""

from .mmengine import parse_mmengine_log, summarize_training  # noqa: F401
from .native import (  # noqa: F401
    parse_babelstream,
    parse_cutlass_profiler,
    parse_nccl_tests,
    parse_nvbandwidth,
)
