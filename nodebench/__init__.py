"""nodebench -- benchmark a whole multi-GPU training node.

Design rules that the whole package obeys:

1. Every bench module returns a plain dict. It never prints a report and never
   writes HTML. The report layer is the only thing that formats.
2. Every parser is a pure function: str in, dict out, no IO. That makes it
   testable against a saved fixture of real tool output.
3. No path, GPU id, batch size or dataset size is ever hardcoded. It all comes
   from the config object.
4. Anything the run did NOT cover is recorded, not silently omitted.
"""

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
