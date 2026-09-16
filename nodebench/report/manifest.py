"""Provenance. Every number in a report must trace back to a file on disk.

A benchmark report is a claim about hardware. The difference between a claim
and evidence is whether a reader can go back to the raw output and check it --
and whether they can tell that the raw output has not been edited since.

So every artifact gets recorded with:

    path        where it is, relative to the run directory
    bytes       size
    lines       line count, for text
    md5         content hash
    produced_by which command wrote it

md5 is used because this is tamper-evidence for honest work, not security. It
answers "is this the same file the report was generated from" cheaply, which is
the only question being asked.

The habit this enforces is more valuable than the hashes themselves: if a
number in the report has no artifact behind it, writing the manifest is where
you notice.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

TEXT_EXT = {".txt", ".log", ".csv", ".json", ".yaml", ".yml", ".md", ".out", ".err"}


def _md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _lines(path: str) -> Optional[int]:
    if os.path.splitext(path)[1].lower() not in TEXT_EXT:
        return None
    n = 0
    try:
        with open(path, "rb") as f:
            for _ in f:
                n += 1
    except OSError:
        return None
    return n


def record(path: str, root: str, produced_by: str = "") -> Dict[str, Any]:
    """Describe one artifact. Missing files are recorded as missing, not skipped."""
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    if not os.path.isfile(path):
        return {"path": rel, "exists": False, "note": "declared but not produced"}
    st = os.stat(path)
    return {
        "path": rel,
        "exists": True,
        "bytes": st.st_size,
        "lines": _lines(path),
        "md5": _md5(path),
        "produced_by": produced_by,
        "empty": st.st_size == 0,
    }


def build_manifest(root: str, produced_by: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Walk a run directory and record every artifact in it."""
    produced_by = produced_by or {}
    entries: List[Dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn == "manifest.json":
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            entries.append(record(full, root, produced_by.get(rel, "")))

    empties = [e["path"] for e in entries if e.get("empty")]
    missing = [e["path"] for e in entries if not e.get("exists")]
    notes: List[str] = []
    if empties:
        notes.append(
            "Empty artifact(s): " + ", ".join(empties) + ". An empty file means the tool "
            "ran and produced nothing -- that is a result, and the report must say so "
            "rather than silently omitting the section."
        )
    if missing:
        notes.append("Declared but absent: " + ", ".join(missing) + ".")

    return {
        "root": root,
        "n_files": len(entries),
        "total_bytes": sum(e.get("bytes", 0) for e in entries),
        "files": entries,
        "notes": notes,
        "hash_algorithm": "md5",
        "purpose": (
            "Tamper-evidence and traceability for honest work. Each md5 lets a reader "
            "confirm the raw output is the same file this report was generated from."
        ),
    }
