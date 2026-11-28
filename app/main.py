"""GoalPath-Bond front end: uv run streamlit run app/main.py

Inputs map to RunConfig (configs/goal.toml supplies the defaults and overrides), runs go through src.backtest.run and
are saved under results/runs, plots are the src.analytics set drawn with plotly. A saved run with the same inputs is
loaded instead of re-run, and params come from results/cache/params when present.
"""
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.figures import FIGURES  # noqa: E402
from src.analytics import daily as D  # noqa: E402
from src.analytics.metrics import build, headline  # noqa: E402
from src.analytics.plots import PLOTS  # noqa: E402
from src.backtest import Context, RunConfig, load_run, run  # noqa: E402
from src.backtest.cache import _fingerprint  # noqa: E402
from src.backtest.results import RESULTS  # noqa: E402
from src.optimize.model import Settings  # noqa: E402
from src.optimize.presets import IR_MAX, RISK_TOL, WARMUP, make_settings  # noqa: E402

st.set_page_config(page_title="GoalPath-Bond", layout="wide")

BASE = RunConfig.from_toml(ROOT / "configs" / "goal.toml")
RUNS = RESULTS / "runs"
EST_DEFAULTS = dict(sigma_method="ledoit_wolf", p=3, window=None, halflife=12, min_obs=6, credit=True, psi_n0=12)
GROUPS = {
    "Performance": ["growth", "relative", "drawdown", "monthly_active"],
    "Risk": ["te", "te_ex_ante_source", "te_realised_source"],
    "Positioning": ["sector_weights", "active_sector", "active_duration", "active_krd"],
    "Holdings and trading": ["holdings", "concentration", "turnover", "cost"],
}
PLOT = {p.name: p for p in PLOTS}
TIPS = {
    "growth": "Net wealth, marked daily.",
    "relative": "Above the dashed line is ahead of goal.",
    "drawdown": "Worst fall behind the benchmark from its peak.",
    "monthly_active": "Net active return vs the month's catch-up goal.",
    "te": "Monthly realised TE is the one comparable to the forecast.",
    "te_ex_ante_source": "Forecast TE by curve, credit and issuer risk.",
    "te_realised_source": "Realised TE by sector. Negative bars hedge.",
    "sector_weights": "Post-trade weights.",
    "active_sector": "Positive is overweight.",
    "active_duration": "Bounded by ±delta_D.",
    "active_krd": "Blue long, red short vs benchmark.",
    "holdings": "Effective N = 1 / sum of squared weights.",
    "concentration": "Coverage is the benchmark weight of bonds held.",
    "turnover": "Share bought each month, first month left out.",
    "cost": "Rebalance trades plus forced sales of index exits.",
}

# keep advanced widget values alive while the basic view hides them
for _k in [k for k in st.session_state if str(k).startswith(("adv_", "b_"))]:
    st.session_state[_k] = st.session_state[_k]


# ---------- cached backend ----------

@st.cache_resource(show_spinner="Loading processed data (once per server)")
def month_ends() -> pd.DatetimeIndex:
    from src.params import load_data
    return load_data().panel.index.get_level_values("month_end").unique().sort_values()


@st.cache_resource(show_spinner="Building benchmark, cash flows and residual panel", max_entries=3)
def context(asset_types: tuple, start: str, T: float) -> Context:
    return Context(BASE.with_(asset_types=asset_types, start=start, T=T))


@st.cache_resource(show_spinner="Marking holdings to market daily", max_entries=6)
def analytics(path: str, mtime: float):
    r = load_run(path)
    try:
        daily = D.daily_paths(r, Path(path) / "analytics")
    except SystemExit as e:     # params cache of the run is gone
        raise FileNotFoundError(str(e)) from None
    mx = build(r, daily)
    return r, mx, headline(mx), D.check(daily, r.monthly)


def _norm(cfg_dict: dict) -> str:
    return json.dumps(RunConfig(**cfg_dict).to_dict(), sort_keys=True, default=str)


def saved_runs() -> list[dict]:
    """Saved src.backtest runs (single runs and sweep members), newest first."""
    out = []
    for f in [*RUNS.glob("*/run.json"), *(RESULTS / "sweeps").glob("*/*/run.json")]:
        m = json.loads(f.read_text())
        if "summary" not in m:          # legacy src.optimize.backtest runs
            continue
        try:
            out.append(dict(path=f.parent, meta=m, norm=_norm(m["inputs"])))
        except (TypeError, ValueError):
            continue
    return sorted(out, key=lambda r: r["meta"]["run"]["started"], reverse=True)


def run_label(m: dict) -> str:
    i, h = m["inputs"], m["horizon"]
    tag = f" · {i['tag']}" if i.get("tag") else ""
    when = m["run"]["started"][5:16].replace("T", " ")
    return (f"{h['first'][:7]} to {h['last'][:7]} · k{i['k_max']} · g{1e4 * i['g_annual']:g} · "
            f"{i['risk_tol']} {i['mode']}{tag} · run {when}")


def find_saved(cfg: RunConfig, runs: list[dict]) -> tuple[Path | None, bool]:
    """Newest saved run with the same inputs, and whether its params came from the current src/params."""
    want, src = _norm(cfg.to_dict()), _fingerprint()
    same = [r for r in runs if r["norm"] == want]
    for r in same:
        if r["meta"]["params_cache"].get("source") == src:
            return r["path"], True
    return (same[0]["path"], False) if same else (None, False)


@contextmanager
def progress_hook(cache, n: int):
    """Wrap cache.get (called once at the start of every month by loop.run) to drive a progress bar."""
    slot = st.empty()
    bar = slot.progress(0.0, text=f"0 / {n} months")
    get, state = cache.get, dict(k=0, hit=0, t=time.time())

    def tracked(d, data_fn):
        d = pd.Timestamp(d)
        hit = d in cache.mem or (cache.dir is not None and (cache.dir / f"{d.date()}.pkl").exists())
        k = state["k"]
        src = "params cached, solving" if hit else "estimating params, then solving"
        bar.progress(k / n, text=f"{k} / {n} months · {d:%Y-%m} · {src}")
        state["k"] += 1
        state["hit"] += hit
        return get(d, data_fn)

    cache.get = tracked
    try:
        yield state
    finally:
        del cache.get
        built = state["k"] - state["hit"]
        slot.empty()
        st.toast(f"{state['k']} / {n} months · {state['hit']} from cache, {built} estimated · "
                 f"{time.time() - state['t']:.0f} s")


def execute(cfg: RunConfig) -> Path:
    ctx = context(cfg.asset_types, cfg.start, cfg.T)
    s = cfg.settings()
    with st.spinner("Preparing params cache"):
        cache = ctx.cache(s.H, cfg.W, cfg.estimation)
    with progress_hook(cache, len(ctx.dates)):
        r = run(cfg, ctx)
    return r.path


# ---------- inputs ----------

BASIC = dict(b_assets=list(BASE.asset_types), b_W=int(BASE.W), b_kmax=int(BASE.k_max), b_mode=BASE.mode,
             b_risk=BASE.risk_tol, b_g=1e4 * BASE.g_annual, b_T=float(BASE.T), b_start=BASE.start)


def basic_inputs(ends) -> dict:
    sb = st.sidebar
    sb.subheader("Portfolio")
    assets = sb.multiselect("Asset types", ["corporate", "treasury", "mbs"], key="b_assets",
                            help="AGG holdings of these types form the benchmark.")
    W = sb.number_input("Wealth ($)", 10_000, 1_000_000_000, step=100_000, format="%d", key="b_W",
                        help="Sets minimum lot sizes.")
    k_max = sb.number_input("Max bonds held", 5, 1000, step=10, key="b_kmax",
                            help="k_max. Fewer names, higher tracking error.")
    sb.subheader("Goal")
    mode = sb.radio("Mode", ["tracking", "return"], horizontal=True, key="b_mode",
                    help="Tracking: hug the benchmark, aim for the goal. Return: maximise active return within the risk limits.")
    risk = sb.select_slider("Risk tolerance", list(RISK_TOL), key="b_risk",
                            help="TE cap: " + ", ".join(f"{k} {1e4 * v['te_cap']:.0f}" for k, v in RISK_TOL.items())
                                 + " bp/yr. Also widens exposure bands.")
    cap_bp = 1e4 * IR_MAX * RISK_TOL[risk]["te_cap"]
    g_bp = sb.number_input("Return goal (bp / year)", -100.0, 500.0, step=5.0, format="%.1f", key="b_g",
                           help=f"Active return over the benchmark. Max {cap_bp:.0f} at {risk} risk.")
    sb.subheader("Horizon")
    T = sb.number_input("Length (years)", 0.25, 7.5, step=0.25, format="%.2f", key="b_T",
                        help="Rebalanced monthly.")
    N = round(12 * T)
    months = [f"{d:%Y-%m}" for d in ends[WARMUP: len(ends) - N]]
    if not months:
        sb.error("Horizon is longer than the data allows.")
        st.stop()
    if st.session_state.b_start not in months:
        st.session_state.b_start = months[-1] if st.session_state.b_start > months[-1] else months[0]
    start = sb.selectbox("Start month", months, key="b_start",
                         help=f"First rebalance, at month-end. Range {months[0]} to {months[-1]}.")
    return dict(asset_types=tuple(assets), W=W, k_max=int(k_max), mode=mode, risk_tol=risk, g_annual=g_bp / 1e4,
                T=T, start=start)


ADV = dict(
    adv_method=BASE.overrides.get("method", "heuristic"), adv_backend=BASE.overrides.get("backend", "gurobipy"),
    adv_H=int(BASE.overrides.get("H", 2)), adv_time=float(BASE.overrides.get("time_limit", 5)),
    adv_gap=100 * BASE.overrides.get("mip_gap", 0.01), adv_mipcand=int(BASE.overrides.get("mip_cand", 150)),
    adv_tolmode=BASE.overrides.get("tol_mode", "relative"),
    adv_tecap=None, adv_eps2=None, adv_dD=None, adv_tolrel=None, adv_maw=None, adv_gamma=None, adv_gpen=None,
    adv_cw=None, adv_ncand=BASE.n_cand, adv_cu=BASE.catch_up["on"], adv_cucap=float(BASE.catch_up["cap"]),
    adv_cufloor=1e4 * BASE.catch_up["floor"], adv_sig="ledoit_wolf", adv_hl=12.0, adv_p=3, adv_win=None,
    adv_minobs=6, adv_credit=True, adv_psi=12.0, adv_reuse=True,
)
for _k, _v in {**BASIC, **ADV}.items():
    st.session_state.setdefault(_k, _v)

# RunConfig field -> (widget key, display scale)
OV_KEYS = dict(method=("adv_method", 1), backend=("adv_backend", 1), H=("adv_H", 1), time_limit=("adv_time", 1),
               mip_gap=("adv_gap", 100), mip_cand=("adv_mipcand", 1), tol_mode=("adv_tolmode", 1),
               te_cap=("adv_tecap", 1e4), eps2=("adv_eps2", 1), delta_D=("adv_dD", 1), tol_rel=("adv_tolrel", 100),
               max_active_w=("adv_maw", 100), gamma=("adv_gamma", 1), goal_penalty=("adv_gpen", 1),
               cost_weight=("adv_cw", 1))
EST_KEYS = dict(sigma_method="adv_sig", halflife="adv_hl", p="adv_p", window="adv_win", min_obs="adv_minobs",
                credit="adv_credit", psi_n0="adv_psi")


def fill_from_run():
    """on_change of the previous-runs picker: show the run and set every input to its values."""
    path = PREV.get(st.session_state.prev_run)
    if not path:
        return
    ss, i = st.session_state, RunConfig(**json.loads((Path(path) / "run.json").read_text())["inputs"])
    ss.update(b_assets=list(i.asset_types), b_W=int(i.W), b_kmax=int(i.k_max), b_mode=i.mode, b_risk=i.risk_tol,
              b_g=1e4 * i.g_annual, b_T=float(i.T), b_start=f"{pd.Timestamp(i.start):%Y-%m}")
    ss.update({k: v for k, v in ADV.items() if k != "adv_reuse"})
    for f, v in i.overrides.items():
        if f in OV_KEYS:
            key, scale = OV_KEYS[f]
            ss[key] = None if v is None else v * scale
    for f, v in i.estimation.items():
        if f in EST_KEYS:
            ss[EST_KEYS[f]] = 0.0 if (f == "psi_n0" and v is None) else v
    ss.update(adv_ncand=i.n_cand, adv_cu=i.catch_up["on"], adv_cucap=float(i.catch_up["cap"]),
              adv_cufloor=1e4 * i.catch_up["floor"])
    if (i.overrides, i.estimation, i.catch_up, i.n_cand) != (BASE.overrides, {}, BASE.catch_up, BASE.n_cand):
        ss.level = "Advanced"       # the basic view would drop this run's advanced settings
    ss.run_path = path


def _num(label, key, preset, help, scale=1.0, fmt="%.2f", step=None):
    """Optional override: empty keeps the preset (shown as placeholder). Returns the value in model units or None."""
    ph = "none" if preset is None else (fmt % (preset * scale))
    v = st.number_input(label, key=key, placeholder=f"preset {ph}", format=fmt, step=step,
                        help=help + " Empty = preset.")
    return None if v is None else v / scale


def advanced_inputs(base: dict) -> tuple[dict, dict, dict, int, bool]:
    sb = st.sidebar
    try:
        pre = make_settings(base["g_annual"], base["risk_tol"], base["k_max"], base["mode"], **BASE.overrides)
    except ValueError:
        pre = Settings()
    ov = dict(BASE.overrides)

    with sb.expander("Solver"):
        ov["method"] = st.selectbox("Method", ["heuristic", "misocp"], key="adv_method",
                                    help="Heuristic: relax and truncate. MISOCP: exact, slower.")
        ov["backend"] = st.selectbox("Backend", ["gurobipy", "cvxpy"], key="adv_backend",
                                     help="gurobipy is primary.")
        ov["H"] = st.number_input("Planning periods H", 1, 6, key="adv_H",
                                  help="Months planned per solve, first one traded.")
        ov["time_limit"] = st.number_input("Time limit (s)", 1.0, 600.0, key="adv_time",
                                           help="Per MIP solve.")
        if ov["method"] == "misocp":
            ov["mip_gap"] = st.number_input("MIP gap (%)", 0.01, 10.0, key="adv_gap",
                                            help="Gurobi stops at this gap.") / 100
            ov["mip_cand"] = st.number_input("MIP candidates", 10, 2000, key="adv_mipcand",
                                             help="Bonds given a binary.")

    with sb.expander("Risk and tolerances"):
        ov["tol_mode"] = st.selectbox("Tolerance mode", ["relative", "flat"], key="adv_tolmode",
                                      help="Relative: bands scale with benchmark exposure. Flat: fixed.")
        extra = {
            "te_cap": _num("TE cap (bp / year)", "adv_tecap", pre.te_cap, "Ex-ante, annual.",
                           1e4, "%.0f", 5.0),
            "eps2": _num("Estimation error weight ε₂", "adv_eps2", pre.eps2,
                         "Higher is more cautious."),
            "delta_D": _num("Active duration band (years)", "adv_dD", pre.delta_D,
                            "Max |active duration|.", step=0.05),
            "tol_rel": _num("Relative band (% of benchmark exposure)", "adv_tolrel", pre.tol_rel,
                            "Key-rate, DTS, convexity.", 100, "%.1f"),
            "max_active_w": _num("Max overweight per bond (%)", "adv_maw", pre.max_active_w,
                                 "Above benchmark weight.", 100, "%.1f"),
        }
    with sb.expander("Objective"):
        extra |= {
            "gamma": _num("Tracking variance weight γ", "adv_gamma", pre.gamma,
                          "Versus trading cost.", fmt="%.0f"),
            "goal_penalty": _num("Goal shortfall penalty", "adv_gpen", pre.goal_penalty,
                                 "Per unit of monthly shortfall.",
                                 fmt="%.0f"),
            "cost_weight": _num("Cost weight", "adv_cw", pre.cost_weight,
                                "1/12 spreads cost over a year.", fmt="%.3f"),
        }
    ov |= {k: v for k, v in extra.items() if v is not None}

    with sb.expander("Rolling loop"):
        n_cand = st.number_input("Candidate set size", 50, 5000, step=50, key="adv_ncand",
                                 help="Largest benchmark bonds considered, plus holdings.")
        cu = dict(
            on=st.toggle("Goal catch-up", key="adv_cu",
                         help="Re-aim each month to still reach the goal."),
            cap=st.number_input("Catch-up cap (× monthly goal)", 1.0, 10.0, step=0.5, key="adv_cucap",
                                help="Max ask when behind."),
            floor=st.number_input("Catch-up floor (bp / month)", -50.0, 50.0, step=0.5, key="adv_cufloor",
                                  help="Min ask when ahead.") / 1e4,
        )

    with sb.expander("Estimation"):
        e = {
            "sigma_method": st.selectbox("Key-rate covariance", ["ledoit_wolf", "sample", "ewma"], key="adv_sig",
                                         help="Estimator for Sigma."),
            "halflife": st.number_input("EWMA half-life (months)", 1.0, 120.0, key="adv_hl",
                                        help="ewma only."),
            "p": st.number_input("Curve PCs", 1, 8, key="adv_p", help="Key-rate principal components."),
            "window": st.number_input("History window (months)", 6, 120, key="adv_win", placeholder="all",
                                      help="Empty = all."),
            "min_obs": st.number_input("Min observations", 2, 60, key="adv_minobs",
                                       help="Per bond, for its own psi."),
            "credit": st.toggle("Credit factors", key="adv_credit", help="One per corporate sector."),
            "psi_n0": st.number_input("Residual prior weight (months)", 0.0, 120.0, key="adv_psi",
                                      help="Sector prior in psi. 0 = none."),
        }
        e["psi_n0"] = e["psi_n0"] or None
        est = {k: v for k, v in e.items() if v != EST_DEFAULTS[k]}
        st.caption("Changing these builds a new params cache.")

    reuse = sb.toggle("Reuse saved run when inputs match", key="adv_reuse")
    return ov, est, cu, int(n_cand), reuse


# ---------- results ----------

def tiles(h: dict, goal_bp: float, adv: bool):
    act = 1e4 * h["active_ann"]
    items = [
        (f"Active return :{'green' if act >= goal_bp else 'red'}-badge[{'↑' if act >= goal_bp else '↓'} "
         f"{act - goal_bp:+.1f} bp vs goal]", f"{act:+.1f} bp/yr", None, "Net, annualised."),
        ("Portfolio / benchmark", f"{h['cum_port']:.2%} / {h['cum_bench']:.2%}", None, "Cumulative return."),
        ("Tracking error", f"{1e4 * h['te_ann']:.0f} bp/yr", None, "Realised, monthly."),
        ("Hit rate", f"{h['hit_rate']:.0%}", None, "Months beating the benchmark."),
    ]
    if adv:
        items += [
            ("Information ratio", f"{h['info_ratio']:.2f}", None, "Active return / TE."),
            ("Max active drawdown", f"{1e4 * h['max_active_drawdown_daily']:.0f} bp", None, "Daily marks."),
            ("Trading cost", f"{h['cost_bp_total']:.1f} bp", None, "Total."),
            ("Holdings", f"{h['holdings_avg']:.0f} / {h['eff_n_avg']:.0f}", None, "Names / effective N."),
        ]
    for row in range(0, len(items), 4):
        for col, (label, val, delta, hlp) in zip(st.columns(4), items[row: row + 4]):
            col.metric(label, val, delta, help=hlp, border=True)


def show(path: Path, adv: bool):
    try:
        r, mx, h, gap = analytics(str(path), (path / "run.json").stat().st_mtime)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    i = r.meta["inputs"]
    st.subheader(r.meta["label"])
    st.caption(f"{r.meta['horizon']['first']} to {r.meta['horizon']['returns_to']} · {h['months']} rebalances · "
               f"{', '.join(i['asset_types'])} · goal {1e4 * i['g_annual']:g} bp/yr · {i['risk_tol']} risk · "
               f"k_max {i['k_max']} · {i['mode']} · {path.name}")
    tiles(h, 1e4 * i["g_annual"], adv)
    tabs = st.tabs(list(GROUPS) + (["Data"] if adv else []))
    for tab, names in zip(tabs, GROUPS.values()):
        with tab:
            cols = st.columns(2)
            for j, name in enumerate(names):
                p = PLOT[name]
                with cols[j % 2]:
                    st.markdown(f"**{p.title}**", help=TIPS[name])
                    st.plotly_chart(FIGURES[name](mx), width="stretch", config={"displaylogo": False},
                                    key=f"fig_{name}")
                    if adv and p.formula:
                        st.latex(p.formula)
    if adv:
        with tabs[-1]:
            st.caption(f"Daily paths reproduce the monthly returns to within {gap:.3f} bp.")
            st.dataframe(r.monthly, width="stretch")
            c1, c2 = st.columns(2)
            c1.json(r.summary, expanded=False)
            c2.json(r.meta["settings"], expanded=False)


# ---------- page ----------

st.session_state.setdefault("level", "Basic")
level = st.sidebar.segmented_control("View", ["Basic", "Advanced"], key="level",
                                     help="Advanced adds overrides, formulas and data.")
adv = (level or "Basic") == "Advanced"

runs = saved_runs()
PREV = {run_label(r["meta"]): str(r["path"]) for r in runs}
st.sidebar.selectbox("Previous runs", list(PREV), index=None, key="prev_run", placeholder=f"{len(PREV)} saved",
                     on_change=fill_from_run,
                     help="Shows the run and fills its inputs.")

ends = month_ends()
base = basic_inputs(ends)
if adv:
    ov, est, cu, n_cand, reuse = advanced_inputs(base)
else:
    ov, est, cu, n_cand, reuse = dict(BASE.overrides), {}, dict(BASE.catch_up), BASE.n_cand, True

error = None
try:
    cfg = BASE.with_(**base, overrides=ov, estimation=est, catch_up=cu, n_cand=n_cand)
    cfg.settings()
except ValueError as e:
    error, cfg = str(e), None

if error:
    st.sidebar.error(error)
hit, current = find_saved(cfg, runs) if cfg is not None else (None, False)
use_saved = hit is not None and current and reuse
clicked = st.sidebar.button("Load saved run" if use_saved else "Run backtest", type="primary", width="stretch",
                            disabled=cfg is None)
if hit is not None and not current:
    st.sidebar.caption("Matches a saved run estimated with older src/params, so it re-runs.")
elif use_saved:
    st.sidebar.caption("Matches a saved run.")

if clicked:
    if use_saved:
        st.toast(f"Loaded saved run {hit.name}")
        st.session_state.run_path = str(hit)
    else:
        st.session_state.run_path = str(execute(cfg))

if "run_path" in st.session_state:
    show(Path(st.session_state.run_path), adv)
else:
    st.title("GoalPath-Bond")
