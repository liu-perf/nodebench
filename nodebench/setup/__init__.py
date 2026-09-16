"""Optional native toolchain builder.

Nothing in the default path needs this. It exists so `--backend native` and
`--backend both` have something to run, and it is a separate command
(`nodebench setup --native`) precisely so that a first-time user is never made
to wait for CUTLASS before seeing a report.
"""

from .build import TOOLS, build_all  # noqa: F401
