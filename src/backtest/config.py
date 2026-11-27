"""User inputs of one run, read from a TOML file (configs/goal.toml) or built in code.

The [inputs] fields are what a user enters. [overrides] sets any Settings field after the presets (advanced, taken as
given, not scaled by the risk tolerance). [catch_up] and n_cand are loop inputs that are not Settings fields.
"""
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from src.optimize.model import Settings
from src.optimize.presets import make_settings

ASSET_TYPES = ("corporate", "treasury", "mbs")


@dataclass(frozen=True)
class RunConfig:
    asset_types: tuple = ("corporate", "treasury")
    g_annual: float = 0.0050        # annualized active return goal over the benchmark
    start: str = "2021-01"          # start month, snaps to its month-end
    T: float = 4                    # horizon in years, N = 12 T monthly rebalances
    risk_tol: str = "medium"        # low | medium | high
    k_max: int = 150
    W: float = 1_000_000            # wealth in dollars, sets the minimum lot weights
    mode: str = "tracking"          # tracking | return
    overrides: dict = field(default_factory=dict)
    catch_up: dict = field(default_factory=lambda: dict(on=True, cap=2.0, floor=0.0))
    n_cand: int = 750               # candidate set size: n_cand largest eligible benchmark weights plus holdings
    tag: str = ""                   # appended to the run folder name

    def __post_init__(self):
        bad = set(self.asset_types) - set(ASSET_TYPES)
        if bad or not self.asset_types:
            raise ValueError(f"asset_types must be a non-empty subset of {ASSET_TYPES}, got {self.asset_types}")
        object.__setattr__(self, "asset_types", tuple(sorted(self.asset_types)))
        unknown = set(self.overrides) - {f.name for f in fields(Settings)}
        if unknown:
            raise ValueError(f"unknown Settings fields in overrides: {sorted(unknown)}")

    def settings(self) -> Settings:
        """Presets for (g, risk_tol, k_max, mode), then overrides. Raises ValueError for a goal above the IR limit."""
        return make_settings(self.g_annual, self.risk_tol, self.k_max, self.mode, **self.overrides)

    def label(self) -> str:
        """Short name for folders and tables, e.g. tracking-medium-k150-g50."""
        base = f"{self.mode}-{self.risk_tol}-k{self.k_max}-g{1e4 * self.g_annual:g}"
        return f"{base}-{self.tag}" if self.tag else base

    def with_(self, **kw) -> "RunConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["asset_types"] = list(self.asset_types)
        return d

    @classmethod
    def from_toml(cls, path) -> "RunConfig":
        raw = tomllib.loads(Path(path).read_text())
        kw = dict(raw.get("inputs", {}))
        for k in ("overrides", "catch_up"):
            if k in raw:
                kw[k] = dict(raw[k])
        kw.update({k: v for k, v in raw.items() if k not in ("inputs", "overrides", "catch_up")})
        if "asset_types" in kw:
            kw["asset_types"] = tuple(kw["asset_types"])
        return cls(**kw)
