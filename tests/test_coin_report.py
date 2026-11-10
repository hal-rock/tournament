"""coin_report.py on synthetic rows: a report that reads " up" gives kappa and the share; a patch that only shrinks the
report gives lambda < 0 and kappa ~ 0, though its swap slope on the coin is negative too."""

import numpy as np

from coin_report import load, stats
from matskit import io
from report_patch import ARMS


def write_rows(path, kappa=0.0, shrink=0.0, n=600, seed=0):
    """Order 0 / 1 of n pairs. Coin c_o = 0.3 + s c; report r_o = 0.5 + s (0.8 c + x) + noise, s = +1 / -1. Each arm adds
    kappa (c_src - c_o) and shrinks the report by `shrink` (as a disruption would)."""
    rng = np.random.default_rng(seed)
    c, x = rng.normal(0, 1, n), rng.normal(0, 1, n)
    for p in range(n):
        for order, s in ((0, 1), (1, -1)):
            c_o, c_other = 0.3 + s * c[p], 0.3 - s * c[p]
            c_ok = 0.3 + s * 0.8 * c[p]
            r = 0.5 + s * (0.8 * c[p] + x[p]) + rng.normal(0, 0.05)
            report, mass = {"none": r}, {"none": 1.0}
            for arm, (source, _, _) in ARMS.items():
                c_src = {"own": c_o, "other": c_other, "ok": c_ok}[source]
                report[arm] = r * (1 - shrink * (source != "own")) + kappa * (c_src - c_o) + rng.normal(0, 0.01)
                mass[arm] = 1.0
            io.append_jsonl(path, dict(pair=p, order=order, a=f"t{p}", b=f"u{p}", coin=c_o, coin_mass=1.0, coin_ok=c_ok,
                                       report=report, mass=mass))


def test_reading_up_gives_kappa_and_share(tmp_path):
    write_rows(tmp_path / "rows.jsonl", kappa=0.3)
    D = load(tmp_path / "rows.jsonl")
    s = stats(D, np.arange(len(D["c"])), {})
    assert abs(s["kappa (swap+ctrl)"] - 0.3) < 0.02 and abs(s["lambda (swap+ctrl)"]) < 0.02
    # share = kappa / slope(report on coin) = 0.3 / 0.8
    assert abs(s["swap: share"] - 0.3 / 0.8) < 0.08
    assert abs(s["self: mean |d|"]) < 0.02


def test_shrinking_is_not_kappa(tmp_path):
    write_rows(tmp_path / "rows.jsonl", shrink=0.4)
    D = load(tmp_path / "rows.jsonl")
    s = stats(D, np.arange(len(D["c"])), {})
    assert s["swap: slope(d on coin)"] < -0.2  # the naive slope can't tell shrinkage from reading " up"
    assert abs(s["kappa (swap+ctrl)"]) < 0.02 and abs(s["lambda (swap+ctrl)"] + 0.4) < 0.02
