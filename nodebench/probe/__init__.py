"""Read-only probes. Nothing in here runs a workload or allocates real memory."""

from .compat import check_compat, probe_environment  # noqa: F401
from .p2p import classify_p2p, probe_p2p  # noqa: F401
from .topology import probe_topology, suggest_gpu_sets  # noqa: F401
