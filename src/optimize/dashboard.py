"""Static dashboard of the chosen backtest config: uv run python -m src.optimize.dashboard -> results/dashboard.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.params import build_cashflows
from src.params.krd import cash_between
from src.params.prices import load_processed

from .backtest import RESULTS, SPLITS, pick_config, summarize
from .model import Settings

K_MAX = Settings().k_max   # k_max is fixed across the grid
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SECTOR = {"treasury": "#2a78d6", "industrial": "#eb6834", "financial": "#1baf7a", "utility": "#eda100",
          "mbs_passthrough": "#e87ba4"}
RES_COLOR = {**SECTOR, "treasury": "#8d8c88"}   # treasury residual grey so it does not read as the level factor


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    for k, sp in ax.spines.items():
        sp.set_visible(k == "bottom")
        sp.set_color(GRID)


def shade(ax, df):
    """Vertical lines at the start of validation and test."""
    for k in ("val", "test"):
        ax.axvline(pd.Timestamp(SPLITS[k][0]), color="black", alpha=0.1, lw=1.2, zorder=3)


def daily_paths(run_dir, best, df) -> pd.DataFrame:
    """Daily wealth of the portfolio and the benchmark (cached in run_dir/daily.parquet).

    Between rebalances the weights are held (no trading), so each bond's path is (dirty price + cash paid since the
    rebalance) / price at the rebalance, with the last observed price carried forward. Bonds the backtest priced at mu_hat
    (stale endpoint) move linearly over the month to that return. The path therefore ends each month exactly on the
    backtest's monthly return; the trading cost is taken on the rebalance day.
    """
    f = run_dir / "daily.parquet"
    if f.exists():
        return pd.read_parquet(f)
    if not (run_dir / "holdings.parquet").exists():
        raise SystemExit("this run has no holdings.parquet; rerun `python -m src.optimize`")
    hold = pd.read_parquet(run_dir / "holdings.parquet")
    hold = hold[(hold.gamma == best["gamma"]) & (hold.H == best["H"])]
    bm = pd.read_parquet(run_dir / "bond_month.parquet")
    universe, prices, curve = load_processed()
    cal = pd.DatetimeIndex(curve["date"])
    table = build_cashflows(universe[universe.maturity_date.notna() & universe.coupon_rate.notna()].reset_index(drop=True),
                            pd.Timestamp(df.date.min()) - pd.Timedelta(days=400))
    tpos = pd.Series(np.arange(len(table.cusips)), index=table.cusips)
    px = prices.drop_duplicates(["cusip", "date"], keep="last").pivot(index="date", columns="cusip", values="dirty_price")
    dates = list(pd.to_datetime(df.date))
    wp, wb, out = 1.0, 1.0, []
    for d0, cost in zip(dates, df.cost):
        d1 = _next_month_end(cal, d0)
        days = cal[(cal > d0) & (cal <= d1)]
        m = bm[bm.date == d0].set_index("cusip")
        x = hold[hold.date == d0].set_index("cusip").x.reindex(m.index).fillna(0.0).to_numpy()
        b, r, bad, p0 = (m[c].to_numpy() for c in ("b", "r", "bad", "p0"))
        p = px.reindex(index=days, columns=m.index)
        p = pd.concat([pd.DataFrame([p0], index=[d0], columns=m.index), p]).ffill().iloc[1:].to_numpy()
        cash = np.stack([cash_between(table, d0, t) for t in days])[:, tpos.reindex(m.index).fillna(0).astype(int).to_numpy()]
        cash[:, tpos.reindex(m.index).isna().to_numpy()] = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            R = (p + cash) / p0 - 1
        frac = (np.arange(1, len(days) + 1) / len(days))[:, None]
        R = np.where(bad[None, :] | ~np.isfinite(R), r[None, :] * frac, R)
        gp, gb = (1 + R) @ x, (1 + R) @ b
        gp = gp * (1 - cost)
        out.append(pd.DataFrame({"date": days, "wealth_port": wp * gp, "wealth_bench": wb * gb, "month": d0}))
        wp, wb = wp * gp[-1], wb * gb[-1]
    daily = pd.concat(out, ignore_index=True)
    daily.to_parquet(f)
    return daily


def _next_month_end(cal, d0):
    ym = cal.to_period("M")
    nxt = ym[cal > d0][0]
    return cal[ym == nxt].max()




def te_stats(active: pd.Series) -> float:
    """Annualised tracking error in bp from daily active returns."""
    return 1e4 * active.std(ddof=1) * np.sqrt(252)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run folder name under results/runs (default: latest)")
    args = ap.parse_args()
    run_dir = RESULTS / "runs" / (args.run or sorted(p.name for p in (RESULTS / "runs").iterdir())[-1])
    all_rows = pd.read_parquet(run_dir / "backtest.parquet")
    summary = pd.read_csv(run_dir / "summary.csv")
    best = pick_config(summary)
    df = all_rows[(all_rows.gamma == best["gamma"]) & (all_rows.H == best["H"])].sort_values("date").reset_index(drop=True)
    d = pd.to_datetime(df.date)
    sm = summarize(df)

    # daily series
    daily = daily_paths(run_dir, best, df)
    dd = pd.to_datetime(daily.date)
    daily = daily.assign(ret_port=daily.wealth_port.pct_change().fillna(daily.wealth_port.iloc[0] - 1),
                         ret_bench=daily.wealth_bench.pct_change().fillna(daily.wealth_bench.iloc[0] - 1))
    daily["active"] = daily.ret_port - daily.ret_bench
    daily["split"] = [next((k for k, (a, b) in SPLITS.items() if pd.Timestamp(a) <= m <= pd.Timestamp(b)), None)
                      for m in pd.to_datetime(daily.month)]
    te_d = {k: te_stats(g.active) for k, g in daily.groupby("split")}
    cum = {k: ((1 + g.ret_port).prod() - 1, (1 + g.ret_bench).prod() - 1) for k, g in daily.groupby("split")}

    # concentration from holdings
    hold = pd.read_parquet(run_dir / "holdings.parquet")
    hold = hold[(hold.gamma == best["gamma"]) & (hold.H == best["H"])]
    bm = pd.read_parquet(run_dir / "bond_month.parquet")
    conc = []
    for dt, g in hold.groupby("date"):
        w = np.sort(g.x.to_numpy() / g.x.sum())[::-1]
        b = bm[bm.date == dt].set_index("cusip").b
        conc.append(dict(date=dt, eff_n=1 / np.sum(w**2), top1=w[0], top10=w[:10].sum(),
                         coverage=b.reindex(g.cusip).fillna(0).sum() / b.sum()))
    conc = pd.DataFrame(conc).sort_values("date")
    cd = pd.to_datetime(conc.date)

    # ex-ante vs realised tracking error (annualised bp)
    ex_ante = 1e4 * np.sqrt(df.track_var_fcst * 12)
    roll = 1e4 * daily.set_index(dd).active.rolling(252, min_periods=126).std() * np.sqrt(252)

    # realised TE by source: monthly active return attributed to sector (sum of active weight * realised return),
    # plus trading cost. Rolling window share = cov(source, total) / var(total), so shares sum to the realised TE.
    hh = hold.copy()
    hh["x"] = hh.x / hh.groupby("date").x.transform("sum")
    mm = bm.merge(hh[["date", "cusip", "x"]], on=["date", "cusip"], how="left").fillna({"x": 0})
    mm["contrib"] = (mm.x - mm.b) * mm.r
    src = mm.groupby(["date", "sector"]).contrib.sum().unstack().fillna(0)
    src.index = pd.to_datetime(src.index)
    src = src.reindex(d)
    src["trading cost"] = df.active.to_numpy() - src.sum(axis=1).to_numpy()
    tot = df.active.set_axis(d)
    WIN = 12
    te_roll = 1e4 * tot.rolling(WIN, min_periods=6).std() * np.sqrt(12)
    share = pd.DataFrame({c: src[c].rolling(WIN, min_periods=6).cov(tot) / tot.rolling(WIN, min_periods=6).var()
                          for c in src})
    real_src = share.mul(te_roll, axis=0)

    fig = plt.figure(figsize=(15, 19.5), facecolor=SURFACE)
    gs = fig.add_gridspec(7, 3, height_ratios=[0.5, 1.4, 1.3, 1.3, 1.2, 1.0, 1.3], hspace=0.6, wspace=0.22,
                          left=0.05, right=0.97, top=0.945, bottom=0.04)
    fig.text(0.05, 0.968, "Preliminary backtest vs AGG weights", fontsize=13, color=INK)
    fig.text(0.05, 0.952, f"gamma={best['gamma']:g}, H={int(best['H'])}, k_max={K_MAX}. Net of costs, monthly rebalance, "
             "daily marks. Vertical lines: start of validation, start of test.", fontsize=9, color=MUTED)

    # stat tiles (test split, the number to quote)
    t = sm.loc["test"]
    tiles = [("Test cum. return", f"{cum['test'][0]:.2%}", f"benchmark {cum['test'][1]:.2%}"),
             ("Test tracking error", f"{te_d['test']:.0f} bp", f"annualised, daily (monthly: {t.te_ann_bp:.0f} bp)"),
             ("Mean active return", f"{t.mean_active_bp:+.1f} bp", "per month, net"),
             ("Avg turnover", f"{t.turnover_pct:.1f}%", f"cost {t.cost_bp:.2f} bp / month"),
             ("Eff. holdings", f"{conc.eff_n.mean():.0f}", f"{t.avg_holdings:.0f} names, {int(df.n_bonds.iloc[-1]):,} in index")]
    gt = gs[0, :].subgridspec(1, 5, wspace=0.08)
    for i, (a, b, c) in enumerate(tiles):
        ax = fig.add_subplot(gt[i])
        ax.axis("off")
        ax.text(0, 0.78, a, fontsize=9, color=MUTED, transform=ax.transAxes)
        ax.text(0, 0.30, b, fontsize=19, color=INK, transform=ax.transAxes)
        ax.text(0, 0.0, c, fontsize=8.5, color=MUTED, transform=ax.transAxes)

    # row 1: daily growth of 1, cumulative relative return
    ax = fig.add_subplot(gs[1, :2])
    shade(ax, df)
    ax.plot(dd, daily.wealth_port, color=BLUE, lw=1.6, label="Portfolio")
    ax.plot(dd, daily.wealth_bench, color=ORANGE, lw=1.6, label="Benchmark (AGG weights)")
    style(ax, "Growth of 1, daily, from first rebalance")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = fig.add_subplot(gs[1, 2])
    shade(ax, df)
    ax.plot(dd, 1e4 * (daily.wealth_port / daily.wealth_bench - 1), color=BLUE, lw=1.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, "Cumulative relative return (bp)")

    # row 2: ex-ante vs realised TE, TE decomposition
    ax = fig.add_subplot(gs[2, :2])
    shade(ax, df)
    ax.step(d, ex_ante, where="mid", color=ORANGE, lw=2, label="Ex-ante (optimizer, monthly forecast)")
    ax.plot(dd, roll, color=BLUE, lw=2, label="Realised, rolling 12 months of daily active")
    style(ax, "Tracking error, annualised (bp)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = fig.add_subplot(gs[2, 2])
    shade(ax, df)
    pcs = [c for c in df.columns if c.startswith("var_pc")]
    ress = [c for c in df.columns if c.startswith("var_res_")]
    total = df[pcs + ress].sum(axis=1)
    parts = [("Level (PC1)", "var_pc0", "#2a78d6"), ("Slope (PC2)", "var_pc1", "#6aa6ec"),
             ("Curvature (PC3)", "var_pc2", "#b5d1f5")]
    parts += [(f"Residual, {c[8:].replace('_', ' ')}", c, RES_COLOR.get(c[8:], MUTED)) for c in ress]
    bottom = np.zeros(len(df))
    for name, c, col in parts:
        h = ex_ante.to_numpy() * (df[c] / total).to_numpy()
        ax.bar(d, h, width=20, bottom=bottom, color=col, label=name)
        bottom += h
    style(ax, "Ex-ante TE by source (bp, split by variance share)")
    ax.set_ylim(0, 1.5 * bottom.max())
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right", handlelength=1.0, columnspacing=1.0)

    # row 3: sector stack, active sector weight, active return distribution
    ax = fig.add_subplot(gs[3, 0])
    shade(ax, df)
    secs = [k for k in SECTOR if f"w_port_{k}" in df]
    ax.stackplot(d, *[df[f"w_port_{k}"] for k in secs], colors=[SECTOR[k] for k in secs], step="post",
                 labels=[k.replace("_", " ") for k in secs])
    ax.set_ylim(0, 1)
    style(ax, "Portfolio weight by sector")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), columnspacing=1.0,
              handlelength=1.2)
    ax = fig.add_subplot(gs[3, 1])
    shade(ax, df)
    for k in secs:
        ax.plot(d, 100 * (df[f"w_port_{k}"] - df[f"w_bench_{k}"]), color=SECTOR[k], lw=1.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, "Active sector weight (pct points)")
    ax = fig.add_subplot(gs[3, 2])
    shade(ax, df)
    ax.plot(d, df.act_D, color=BLUE, lw=2, marker="o", ms=4, mec=SURFACE)
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, "Active duration (years)")

    # row 4: concentration, active duration
    ax = fig.add_subplot(gs[4, 0])
    shade(ax, df)
    ax.plot(cd, conc.eff_n, color=BLUE, lw=2, marker="o", ms=4, mec=SURFACE, label="Effective holdings (1 / HHI)")
    ax.plot(d, df.n_hold, color=MUTED, lw=1.2, ls="--", label="Names held")
    style(ax, "Effective number of holdings")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = fig.add_subplot(gs[4, 1])
    shade(ax, df)
    ax.plot(cd, 100 * conc.top1, color=ORANGE, lw=2, label="Largest weight")
    ax.plot(cd, 100 * conc.top10, color=BLUE, lw=2, label="Top 10 weights")
    ax.plot(cd, 100 * conc.coverage, color=SECTOR["financial"], lw=2, label="Index value in held names")
    style(ax, "Concentration (% of portfolio / index)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = fig.add_subplot(gs[4, 2])
    shade(ax, df)
    ax.bar(d, 100 * df.turnover, width=20, color=BLUE)
    style(ax, "Turnover (% of portfolio bought)")

    # row 5: turnover, cost, split table
    ax = fig.add_subplot(gs[5, 0])
    shade(ax, df)
    ax.bar(d, 1e4 * df.cost, width=20, color=ORANGE)
    style(ax, "Trading cost (bp of wealth)")
    ax = fig.add_subplot(gs[5, 1:])
    ax.axis("off")
    names = ["port", "bench", "TE d", "TE m", "turn %", "cost bp"]
    cell = [[f"{cum[k][0]:.1%}", f"{cum[k][1]:.1%}", f"{te_d[k]:.0f}", f"{sm.loc[k, 'te_ann_bp']:.0f}",
             f"{sm.loc[k, 'turnover_pct']:.1f}", f"{sm.loc[k, 'cost_bp']:.2f}"] for k in sm.index]
    tb = ax.table(cellText=cell, rowLabels=list(sm.index), colLabels=names, loc="center", cellLoc="right")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 1.5)

    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID)
        cl.get_text().set_color(INK if r else MUTED)
    ax.set_title("By split (TE bp: d = daily, m = monthly)", loc="left", fontsize=10, color=INK)

    # row 6: realised TE by source
    ax = fig.add_subplot(gs[6, :2])
    shade(ax, df)
    pos = np.zeros(len(df))
    neg = np.zeros(len(df))
    for c in real_src:
        v = real_src[c].fillna(0).to_numpy()
        base = np.where(v >= 0, pos, neg)
        ax.bar(d, v, width=20, bottom=base, color=SECTOR.get(c, MUTED), label=c.replace("_", " "))
        pos += np.where(v >= 0, v, 0)
        neg += np.where(v < 0, v, 0)
    ax.plot(d, te_roll, color=INK, lw=1.2, ls="--", label="Realised TE (rolling 12m, monthly)")
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, "Realised TE by source (bp, rolling 12m)")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), handlelength=1.0, columnspacing=1.0)

    out = run_dir / "dashboard.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("saved", out)
    print("daily TE by split (bp):", {k: round(v, 1) for k, v in te_d.items()},
          "| monthly:", sm.te_ann_bp.round(1).to_dict())
    chk = daily.groupby("month")["ret_port"].apply(lambda g: (1 + g).prod() - 1)
    print("max |daily-compounded vs backtest monthly port ret| (bp):",
          round(1e4 * float(np.max(np.abs(chk.to_numpy() - df.ret_port.to_numpy()))), 3))


if __name__ == "__main__":
    main()
