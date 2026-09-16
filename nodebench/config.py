"""Configuration: the single place any machine-specific value is allowed to live.

If you find yourself typing a path, a hostname, a GPU id or a batch size
anywhere else in this package, it belongs here instead.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge. `override` wins; `None` in override means 'inherit'."""
    out = dict(base)
    for k, v in (override or {}).items():
        if v is None and k in out:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


DEFAULTS: Dict[str, Any] = {
    "node": {"label": "unnamed-node", "vendor": "", "notes": ""},
    "paths": {"workdir": "./nodebench_out", "third_party": "./third_party"},
    "topology": {"numa_groups": None},
    "bench": {
        "modules": ["flops", "membw", "pcie", "nccl"],
        "backend": "torch",
        "flops": {
            "size": 8192,
            "dtypes": ["fp32", "tf32", "fp16", "bf16", "fp8"],
            "iters": 30,
            "warmup": 10,
        },
        "membw": {"elements": 268_435_456, "iters": 30, "warmup": 10},
        "pcie": {"bytes": 1_073_741_824, "iters": 20, "warmup": 5},
        "nccl": {
            "sizes": [16_777_216, 67_108_864, 268_435_456, 1_073_741_824],
            "collectives": ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"],
            "iters": 20,
            "warmup": 5,
            "gpu_sets": None,
        },
    },
    "monitor": {
        "enabled": True,
        "interval_s": 1.0,
        "idle_baseline_s": 12,
        "busy_sm_threshold": 90,
    },
    "report": {"title": "Node benchmark report", "manifest": True, "caveats": True},
    "recipes": {
        "llm_lora": {
            "enabled": False,
            "model": "Qwen/Qwen3-8B",
            "batch_per_gpu": 4,
            "seq_len": 1024,
            "lora_r": 16,
            "lora_alpha": 32,
            "steps": 30,
            "warmup_steps": 8,
            "synthetic": True,
        }
    },
}


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))
    source: Optional[str] = None

    # -- dotted access ------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)

    # -- derived paths ------------------------------------------------------
    @property
    def workdir(self) -> Path:
        p = Path(os.path.expandvars(str(self.get("paths.workdir")))).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def third_party(self) -> Path:
        p = Path(os.path.expandvars(str(self.get("paths.third_party")))).expanduser()
        return p

    @property
    def raw_dir(self) -> Path:
        p = self.workdir / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> Dict[str, Any]:
        return {"config": self.raw, "source": self.source}


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    """Load YAML config on top of DEFAULTS. Both arguments are optional."""
    data = dict(DEFAULTS)
    src = None
    if path:
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise SystemExit("PyYAML is required to read a config file: pip install PyYAML") from e
        text = Path(path).read_text(encoding="utf-8")
        data = _deep_merge(data, yaml.safe_load(text) or {})
        src = str(Path(path).resolve())
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(raw=data, source=src)


def dump_default_yaml() -> str:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        import json

        return json.dumps(DEFAULTS, indent=2)
    return yaml.safe_dump(asdict(Config()) ["raw"], sort_keys=False, allow_unicode=True)
