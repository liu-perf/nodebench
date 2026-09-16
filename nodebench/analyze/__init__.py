"""Interpretation layer. Takes measurements and says what they mean.

Kept separate from `bench/` on purpose: a measurement and its interpretation
have different lifetimes. Reconciliation logic changes as you understand the
hardware better; the raw numbers must not.
"""

from .baseline import assess_idle, signal_to_noise  # noqa: F401
from .crosscheck import cross_check  # noqa: F401
from .ring_theory import (  # noqa: F401
    busbw_factor,
    gradient_traffic_per_step,
    reconcile,
    ring_allreduce_bytes,
)
from .scaling import efficiency, scaling_table  # noqa: F401
