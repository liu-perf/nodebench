"""Reporting: manifest, auto-generated caveats, self-contained HTML.

A report here is data plus a template, never handwritten prose about a specific
machine. Run the same code on a different node and every conclusion changes
because the measurements changed -- not because someone edited the text.
"""

from .caveats import build_caveats  # noqa: F401
from .html import render_report, write_report  # noqa: F401
from .manifest import build_manifest, record  # noqa: F401
