"""The replication's analysis on the earlier Llama 70B and Gemma 4 31B runs: assemble them into the replication layout, then
check that replication_analysis.py reproduces the README's numbers.

    uv run python scripts/replication_legacy.py            # assemble, analyze, check (all three)
    uv run python scripts/replication_legacy.py assemble   # outputs/replication/legacy-{llama-70b,gemma4-31b}/results/
    uv run python scripts/replication_legacy.py check      # after replication_analysis.py on both keys

assemble: each replication condition the earlier sections scored is symlinked from outputs/<run_id>/results/ (downloaded from
W&B first where it isn't local) into outputs/replication/<key>/results/, with run_config.yaml and sources.yaml (the run each
condition came from). Where several runs scored a condition, the layout takes one (SOURCES, with the reason) and the others
become variants (VARIANTS): the same analysis with the run a section actually used swapped in.

check: every README number listed in CHECKS against the recomputed one, from the section's own inputs (the variant, if any)
and from the assembled layout. A number matches when the recomputed value rounds to the README's, at the README's precision.
Writes replication/legacy_check.md.
"""

import argparse
import re
from decimal import ROUND_HALF_UP, Decimal
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replication_analysis as ra  # noqa: E402
from persona_placement import placement_call  # noqa: E402
from persona_readouts import readout_call  # noqa: E402
from personas import pull_call  # noqa: E402
from predict_user import delta_call  # noqa: E402
from roleplay import compare_from  # noqa: E402
from user_choice import call_from  # noqa: E402

PROJECT = "halrockwell-mats/tournament"
OUTPUTS = Path("outputs")
CHECK_MD = Path("replication/legacy_check.md")

# Runs by section (README tables). The sweep: "Model families"; user_choice: "Does post-training make the user's choice
# more like the Assistant's?"; predict_user: "What does the Assistant think the user will choose?"; personas: "Personas in
# the system prompt" (pod a evil, pod b slacker); placement: "Persona placement" (pod a +user, pod b +self); readouts: "The
# persona as the user" (one pod per persona); roleplay: "Personas as roleplay" (one pod per persona).
RUNS = {
    "legacy-llama-70b": dict(
        label="Llama 3.3 70B (legacy)", base="unsloth/Llama-3.1-70B", final="unsloth/Llama-3.3-70B-Instruct",
        sweep="tvlqj8a0", user_choice=("7v41hj1v", "bhv8okeh"), predict_user=("jn9wpn3n", "w8bq8aa5"),
        personas={"evil": "0ex0e8lc", "slacker": "5d5gcran"}, placement={"user": "chyhcvk4", "self": "rtg1enyk"},
        readouts={"evil": "5f98la6b", "slacker": "68ixsqgh"}, roleplay={"evil": "eb20zbnb", "slacker": "n0p20nof"}),
    "legacy-gemma4-31b": dict(
        label="Gemma 4 31B (legacy)", base="google/gemma-4-31B", final="google/gemma-4-31B-it",
        sweep="jrl62npf", user_choice=("oejp2bs9", "rxow1qso"), predict_user=("zcxqd9fp", "wgtt5s6v"),
        personas={"evil": "z9v16lk3", "slacker": "mgldkcxm"}, placement={"user": "3k3rfilv", "self": "1k8fj865"},
        readouts={"evil": "2id90q0w", "slacker": "6rd9rfmr"}, roleplay={"evil": "evehj27v", "slacker": "b10cpytg"}),
}
PERSONAS = ("evil", "slacker")  # the only replication personas the earlier runs have


def sources(key):
    """{"stage/condition": (run id, section, note)} of the assembled layout."""
    R = RUNS[key]
    out = {}
    for s in ("base", "instruct"):
        for c in ("assistant", "coinflip"):
            out[f"{s}/{c}"] = (R["sweep"], "Model families (sweep)",
                               "also re-scored by later pods; every section's A_0 and C_0 are the sweep's")
    for s, run in zip(("base", "instruct"), R["user_choice"]):
        out[f"{s}/user_choice"] = (run, "user's choice", "the user-choice section's U; the predict-user pods re-scored it "
                                   "(variant predict_user)")
    for s, run in zip(("base", "instruct"), R["predict_user"]):
        out[f"{s}/predict_user"] = (run, "predict user", "")
    for p in PERSONAS:
        for c in ("assistant", "coinflip"):
            out[f"instruct/{c}@{p}"] = (R["personas"][p], "personas, system prompt", "")
            out[f"instruct/{c}@{p}+roleplay"] = (R["roleplay"][p], "roleplay", "")
        for c in ("assistant", "coinflip", "user_choice", "predict_user"):
            note = ("the readouts pod scored all four self readouts together, as the replication does; the placement "
                    "section's own A and C of the self-description are variant placement_self") if p == "evil" and \
                c in ("assistant", "coinflip") else ""
            out[f"instruct/{c}@{p}+self"] = (R["readouts"][p], "persona readouts", note)
    for c in ("assistant", "coinflip"):
        out[f"instruct/{c}@evil+user"] = (R["placement"]["user"], "persona placement", "evil only (no slacker +user run)")
    return out


def variants(key):
    """{variant: {"stage/condition": run id}}: the conditions a section took from another run than the layout's."""
    R = RUNS[key]
    pu = dict(zip(("base", "instruct"), R["predict_user"]))
    return {
        # "What does the Assistant think the user will choose?" re-scored A and U on its own pods.
        "predict_user": {f"{s}/{c}": pu[s] for s in pu for c in ("assistant", "user_choice")},
        # "Persona placement" scored the self-description's A and C on its pod b.
        "placement_self": {f"instruct/{c}@evil+self": R["placement"]["self"] for c in ("assistant", "coinflip")},
        # "The persona as the user" took U_0 (no persona) from the predict-user run.
        "readouts_u0": {"instruct/user_choice": pu["instruct"]},
    }


def results_file(run_id, cond_key):
    """outputs/<run>/results/<stage>__<condition>.csv.gz, downloaded from W&B if it isn't local."""
    stage, cond = cond_key.split("/", 1)
    name = f"results/{stage}__{cond}.csv.gz"
    path = OUTPUTS / run_id / name
    if not path.exists():
        import wandb
        print(f"  downloading {run_id}/{name}")
        wandb.Api().run(f"{PROJECT}/{run_id}").file(name).download(root=str(OUTPUTS / run_id), replace=True)
    return path


def assemble(key):
    R = RUNS[key]
    out = ra.OUT / key
    res = out / "results"
    res.mkdir(parents=True, exist_ok=True)
    src = sources(key)
    for cond_key, (run, _, _) in src.items():
        target = results_file(run, cond_key).resolve()
        link = res / target.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    for runs in variants(key).values():
        for cond_key, run in runs.items():
            results_file(run, cond_key)
    cfg = dict(run_name=key, label=R["label"], legacy=True, tasks_file=ra.DEFAULT_TASKS, max_pairs=15000, seed=0,
               models=[dict(name=R["base"], stage="base"), dict(name=R["final"], stage="instruct")])
    (out / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "sources.yaml").write_text(yaml.safe_dump(
        dict(conditions={k: dict(run=r, section=s, note=n) for k, (r, s, n) in src.items()}, variants=variants(key)),
        sort_keys=False, allow_unicode=True))
    print(f"{key}: {len(src)} conditions linked into {res}")


def variant_stats(model, name, runs, n_boot):
    """The model's statistics with the variant's runs swapped in."""
    U = dict(model.U)
    for cond_key, run in runs.items():
        info = ra.cell_info(results_file(run, cond_key), model.ids, model.dir / "analysis" / "variants" / run)
        U[cond_key] = info["u"]
    print(f"  {model.key}: variant {name}")
    return ra.bootstrap(U, model.origins, model.nonharm, ra.resamples_for(model.origins, n_boot), model.personas())


# README values, per model: (section, quantity, value or call function, Llama, Gemma, variant). A value is a stat name
# from stats.csv; a call is a function of the statistics. "" skips a model.
CALLS = {
    "user choice call": lambda S, p: call_from(*S["f2:d"], S["shift:U~A"][0], S["shift:U~A"][1]),
    "predict user call": lambda S, p: delta_call(*S["f3:delta"]),
    "persona call": lambda S, p: pull_call(S[f"fit:C_sys@{p}:b_def"][1], *S[f"fit:C_sys@{p}:b_per"][1:])
    if abs(S[f"f4:{p}:sep_sys"][0]) <= ra.SEPARABLE else "not separable",
    "placement call, user": lambda S, p: placement_call(*S[f"fit:C_user@{p}:b_per"][1:], *S[f"f4:{p}:rho_user"][1:]),
    "placement call, self": lambda S, p: placement_call(*S[f"fit:C_self@{p}:b_per"][1:], *S[f"f5:{p}:rho_self"][1:]),
    "roleplay call": lambda S, p: pull_call(S[f"f6:{p}:b_def"][1], *S[f"f6:{p}:b_per"][1:])
    if abs(S[f"f6:{p}:sep_r"][0]) <= ra.SEPARABLE else "not separable",
    "roleplay vs system prompt": lambda S, p: compare_from(CALLS["roleplay call"](S, p), *S[f"f6:{p}:dw"][1:]),
    "Δ_U call": lambda S, p: readout_call(S[f"f5:{p}:sep_self"][0], *S[f"f5:{p}:dU"]),
    "Δ_C call": lambda S, p: readout_call(S[f"f5:{p}:sep_self"][0], *S[f"f5:{p}:dC"]),
}

FAM = "Model families"
UC = "Does post-training make the user's choice more like the Assistant's?"
PU = "What does the Assistant think the user will choose?"
PS = "Personas in the system prompt"
PL = "Persona placement"
RP = "Personas as roleplay"
RO = "The persona as the user"
CHECKS = [
    (FAM, "r(A_f, C_f)", "r:C_f~A_f", "0.92 [0.90, 0.93]", "0.91 [0.88, 0.93]", None),
    (FAM, "r(A_b, C_b)", "r:C_b~A_b", "0.15 [0.04, 0.26]", "0.50 [0.43, 0.56]", None),
    (FAM, "r(A_f, C_f), non-harmful", "r:C_f~A_f|non_harmful", "0.88", "0.87", None),
    (FAM, "r(A_b, C_b), non-harmful", "r:C_b~A_b|non_harmful", "0.16", "0.42", None),
    (FAM, "r shift, coin", "shift:C~A", "0.92", "0.90", None),
    (FAM, "coin SD, final", "sd:C_f", "1.15", "3.38", None),
    (FAM, "coin SD, base", "sd:C_b", "0.05", "0.09", None),
    (FAM, "coin benign-harmful, final", "gap:C_f", "2.13", "5.74", None),
    (FAM, "coin benign-harmful, base", "gap:C_b", "-0.01", "0.07", None),
    (UC, "r(A_f, C_f) within origin (pre-registration table)", "r:C_f~A_f|within", "0.78", "0.75", None),
    (UC, "r(A_b, U_b)", "r:U_b~A_b", "0.82", "0.80", None),
    (UC, "r(A_f, U_b)", "r:U_b~A_f", "0.60", "0.68", None),
    (UC, "r(A_f, U_f)", "r:U_f~A_f", "0.95", "0.90", None),
    (UC, "d = r(A_f, U_f) - r(A_f, U_b)", "f2:d", "+0.36 [+0.28, +0.43]", "+0.22 [+0.16, +0.30]", None),
    (UC, "r shift", "shift:U~A", "0.96", "0.89", None),
    (UC, "call", "user choice call", "more Assistant-like", "more Assistant-like", None),
    (UC, "final user r, non-harmful", "r:U_f~A_f|non_harmful", "0.94", "0.91", None),
    (UC, "final user r, within origin", "r:U_f~A_f|within", "0.88", "0.78", None),
    (UC, "base user r, within origin", "r:U_b~A_b|within", "0.74", "0.65", None),
    (UC, "final user b (per Assistant SD)", "slope:U_f~A_f", "1.62", "2.85", None),
    (UC, "final user gap", "gap:U_f", "3.20", "4.07", None),
    (UC, "final user SD", "sd:U_f", "1.69", "3.16", None),
    (UC, "base user SD", "sd:U_b", "0.11", "0.19", None),
    (PU, "call", "predict user call", "similar", "own choice", "predict_user"),
    (PU, "Δ = r(U_f, P_f) - r(U_f, A_f)", "f3:delta", "+0.01 [-0.00, +0.02]", "-0.14 [-0.16, -0.12]", "predict_user"),
    (PU, "r(U_f, P_f)", "r:U_f~P_f", "0.96", "0.76", "predict_user"),
    (PU, "r(U_f, A_f)", "r:U_f~A_f", "0.95", "0.90", "predict_user"),
    (PU, "r(U_f, A_f | P_f)", "f3:partial_ua_p", "0.51 [0.41, 0.60]", "0.75 [0.71, 0.79]", "predict_user"),
    (PU, "r(U_f, P_f | A_f)", "f3:partial_up_a", "0.59 [0.51, 0.67]", "-0.08 [-0.22, 0.06]", "predict_user"),
    (PU, "Δ, non-harmful", "f3:delta|non_harmful", "-0.00 [-0.02, +0.01]", "-0.11 [-0.15, -0.08]", "predict_user"),
    (PU, "Δ, within origin", "f3:delta|within", "+0.02 [-0.01, +0.05]", "-0.20 [-0.30, -0.13]", "predict_user"),
    (PU, "r(A_f, P_f)", "r:P_f~A_f", "0.95", "0.86", "predict_user"),
    (PU, "r(A_f, P_f), non-harmful", "r:P_f~A_f|non_harmful", "0.89", "0.77", "predict_user"),
    (PU, "r(A_b, P_b)", "r:P_b~A_b", "0.73", "0.72", "predict_user"),
    (PU, "r(A_f, P_b)", "r:P_b~A_f", "0.48", "0.51", "predict_user"),
    (PU, "d_P", "f3:d_P", "+0.46 [+0.37, +0.57]", "+0.35 [+0.28, +0.44]", "predict_user"),
    (PU, "r shift, A and P", "shift:P~A", "0.95", "0.86", "predict_user"),
    (PU, "P SD, final", "sd:P_f", "6.16", "6.66", "predict_user"),
    (PU, "slope of P_f on A_f", "slope:P_f~A_f", "5.82", "5.75", "predict_user"),
    (PU, "gap P_f", "gap:P_f", "12.24", "14.16", "predict_user"),
    (PS, "b_0 (coin slope on A_0)", "slope:C_f~A_f", "1.06", "3.07", None),
]
PERSONA_CHECKS = [  # (section, quantity, stat with {p}, {persona: (Llama, Gemma)}, variant)
    (PS, "r(A_sys, A_0)", "f4:{p}:sep_sys", {"evil": ("-0.12", "-0.33"), "slacker": ("0.29", "0.39")}, None),
    (PS, "b_def", "fit:C_sys@{p}:b_def", {"evil": ("0.30 [0.26, 0.34]", "-0.55 [-0.74, -0.36]"),
                                         "slacker": ("0.41 [0.36, 0.47]", "0.24 [0.08, 0.39]")}, None),
    (PS, "b_per", "fit:C_sys@{p}:b_per", {"evil": ("0.49 [0.44, 0.53]", "2.55 [2.28, 2.81]"),
                                         "slacker": ("0.51 [0.45, 0.55]", "4.13 [3.87, 4.38]")}, None),
    (PS, "w", "fit:C_sys@{p}:w", {"evil": ("0.62 [0.58, 0.66]", "1.27 [1.16, 1.42]"),
                                 "slacker": ("0.55 [0.51, 0.59]", "0.95 [0.91, 0.98]")}, None),
    (PS, "call", "persona call", {"evil": ("mix", "persona"), "slacker": ("mix", "mix")}, None),
    (PS, "total pull b_def + b_per", "fit:C_sys@{p}:total", {"evil": ("0.78", "2.00"), "slacker": ("0.92", "4.37")}, None),
    (PS, "coin gap C_sys", "gap:C_sys@{p}", {"evil": ("-0.02", "-6.22"), "slacker": ("1.26", "4.90")}, None),
    (PL, "user: b_def", "fit:C_user@{p}:b_def", {"evil": ("0.17", "1.22")}, None),
    (PL, "user: b_per", "fit:C_user@{p}:b_per", {"evil": ("0.36 [0.32, 0.40]", "1.66 [1.44, 1.87]")}, None),
    (PL, "user: ρ_user", "f4:{p}:rho_user", {"evil": ("0.74 [0.70, 0.78]", "0.65 [0.60, 0.71]")}, None),
    (PL, "user: call", "placement call, user", {"evil": ("weaker", "weaker")}, None),
    (PL, "user: coin gap", "gap:C_user@{p}", {"evil": ("-0.16", "-0.56")}, None),
    (PL, "user: r(A_user, A_e)", "f4:{p}:r_auser_ap", {"evil": ("0.98", "0.22")}, None),
    (PL, "user: r(A_user, A_0)", "f4:{p}:sep_user", {"evil": ("", "0.74")}, None),
    (PL, "user: A_user gap", "gap:A_user@{p}", {"evil": ("-7.0", "+7.6")}, None),
    (PL, "self: b_def", "fit:C_self@{p}:b_def", {"evil": ("0.73", "2.15")}, "placement_self"),
    (PL, "self: b_per", "fit:C_self@{p}:b_per", {"evil": ("0.10 [0.06, 0.14]", "1.37 [1.18, 1.57]")}, "placement_self"),
    (PL, "self: ρ_self", "f5:{p}:rho_self", {"evil": ("0.20 [0.12, 0.28]", "0.54 [0.47, 0.62]")}, "placement_self"),
    (PL, "self: call", "placement call, self", {"evil": ("weaker", "weaker")}, "placement_self"),
    (PL, "ρ_self - ρ_user", "f5:{p}:drho", {"evil": ("-0.54 [-0.63, -0.46]", "-0.11 [-0.16, -0.06]")}, "placement_self"),
    (PL, "self: coin gap", "gap:C_self@{p}", {"evil": ("+1.52", "+1.54")}, "placement_self"),
    (PL, "self: r(A_self, A_0)", "r0:A_self@{p}", {"evil": ("0.86", "0.94")}, "placement_self"),
    (PL, "self: A_self gap", "gap:A_self@{p}", {"evil": ("+13.5", "+18.3")}, "placement_self"),
    (RP, "r(A_r, A_s)", "f6:{p}:r_ar_ap", {"evil": ("0.98", "0.81"), "slacker": ("0.98", "1.00")}, None),
    (RP, "b_def", "f6:{p}:b_def", {"evil": ("0.15 [0.11, 0.19]", "-0.02 [-0.24, 0.18]"),
                                  "slacker": ("0.22 [0.18, 0.27]", "0.12 [-0.02, 0.25]")}, None),
    (RP, "b_per", "f6:{p}:b_per", {"evil": ("0.41 [0.35, 0.45]", "1.75 [1.49, 1.99]"),
                                  "slacker": ("0.62 [0.57, 0.68]", "4.06 [3.82, 4.28]")}, None),
    (RP, "w_r", "f6:{p}:w", {"evil": ("0.73 [0.68, 0.80]", "1.01 [0.90, 1.15]"),
                            "slacker": ("0.74 [0.70, 0.78]", "0.97 [0.94, 1.01]")}, None),
    (RP, "call", "roleplay call", {"evil": ("mix", "persona"), "slacker": ("mix", "persona")}, None),
    (RP, "Δw = w_r - w_s", "f6:{p}:dw", {"evil": ("+0.11 [+0.08, +0.15]", "-0.26 [-0.37, -0.17]"),
                                        "slacker": ("+0.19 [+0.16, +0.21]", "+0.03 [+0.01, +0.04]")}, None),
    (RP, "vs the system prompt", "roleplay vs system prompt",
     {"evil": ("more than the system prompt", "less than the system prompt"),
      "slacker": ("more than the system prompt", "more than the system prompt")}, None),
    (RP, "coin gap C_r", "gap:C_roleplay@{p}", {"evil": ("-0.24", "-1.36"), "slacker": ("0.83", "4.75")}, None),
    (RP, "A_r gap", "gap:A_roleplay@{p}", {"evil": ("-5.9", "-3.9"), "slacker": ("6.9", "10.9")}, None),
    (RO, "r(A_self, P_self)", "f5:{p}:sep_self", {"evil": ("0.69", "-0.25"), "slacker": ("0.93", "0.97")}, None),
    (RO, "Δ_U", "f5:{p}:dU", {"evil": ("+0.04 [-0.01, +0.11]", "-0.85 [-0.94, -0.76]")}, None),
    (RO, "Δ_U call", "Δ_U call", {"evil": ("similar", "Assistant"), "slacker": ("not separable", "not separable")}, None),
    (RO, "Δ_C", "f5:{p}:dC", {"evil": ("-0.28 [-0.33, -0.23]", "-0.27 [-0.36, -0.17]")}, None),
    (RO, "Δ_C call", "Δ_C call", {"evil": ("Assistant", "Assistant"), "slacker": ("not separable", "not separable")}, None),
    (RO, "r(U, A | P)", "f5:{p}:partial_Ua_p", {"evil": ("0.66 [0.59, 0.73]", "0.92 [0.90, 0.93]")}, None),
    (RO, "r(U, P | A)", "f5:{p}:partial_Up_a", {"evil": ("0.75 [0.67, 0.82]", "0.54 [0.45, 0.63]")}, None),
    (RO, "r(C, A | P)", "f5:{p}:partial_Ca_p", {"evil": ("0.80 [0.77, 0.84]", "0.82 [0.79, 0.85]")}, None),
    (RO, "r(C, P | A)", "f5:{p}:partial_Cp_a", {"evil": ("-0.01 [-0.17, 0.14]", "0.72 [0.67, 0.77]")}, None),
] + [
    (RO, f"w, {READ} {place}", f"fit:{x}_{s}@{{p}}:w", vals, variant)
    for READ, x, s, vals, variant in (
        ("A", "A", "self", {"evil": ("0.18", "-0.03"), "slacker": ("0.79", "0.81")}, None),
        ("C", "C", "self", {"evil": ("0.12", "0.39"), "slacker": ("0.38", "0.69")}, None),
        ("U", "U", "self", {"evil": ("0.38", "0.22"), "slacker": ("0.82", "0.63")}, None),
        ("P", "P", "self", {"evil": ("0.57", "0.94"), "slacker": ("1.06", "1.01")}, None),
        ("A", "A", "none", {"evil": ("0.00", "0.00"), "slacker": ("0.00", "0.00")}, None),
        ("C", "C", "none", {"evil": ("0.13", "0.05"), "slacker": ("-0.11", "-0.23")}, None),
        ("U", "U", "none", {"evil": ("0.12", "0.21"), "slacker": ("-0.03", "-0.20")}, "readouts_u0"),
        ("P", "P", "none", {"evil": ("0.06", "-0.20"), "slacker": ("-0.07", "-0.25")}, None),
        ("C", "C", "sys", {"evil": ("0.62", "1.27"), "slacker": ("0.55", "0.95")}, None))
    for place in ("self-description" if s == "self" else "no persona" if s == "none" else "system prompt",)
] + [
    (RO, f"gap, {x} self-description", f"gap:{x}_self@{{p}}", {"evil": vals}, None)
    for x, vals in (("A", ("13.5", "18.3")), ("C", ("1.52", "1.54")), ("U", ("0.99", "5.22")), ("P", ("1.5", "-12.4")))
]


def parse(text):
    """README value -> (numbers, decimals): "0.92 [0.90, 0.93]" -> ([0.92, 0.90, 0.93], 2)."""
    nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", text)]
    dec = len(re.search(r"\d+\.(\d+)", text).group(1))
    return nums, dec


def rounded(v, dec):
    return Decimal(repr(v)).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)


def compare(readme, S, stat, persona):
    """(recomputed text, status) of one README value against statistics S. Status: "yes" if the recomputed number rounds
    to the README's at its precision (CI ends included); "rounded twice" if it does only when first rounded to 3 decimals,
    as the sections' reports print (0.5048 -> 0.505 -> 0.51); else "no"."""
    if stat in CALLS:
        try:
            got = CALLS[stat](S, persona)
        except KeyError:
            return "missing", "no"
        return got, "yes" if got == readme else "no"
    name = stat.format(p=persona)
    if name not in S:
        return "missing", "no"
    nums, dec = parse(readme)
    vals = S[name][:1] if len(nums) == 1 else S[name]
    text = f"{vals[0]:+.4f}" + (f" [{vals[1]:+.4f}, {vals[2]:+.4f}]" if len(vals) == 3 else "")
    target = [Decimal(repr(n)).quantize(Decimal(1).scaleb(-dec)) for n in nums]
    if [rounded(v, dec) for v in vals] == target:
        return text, "yes"
    if [rounded(float(rounded(v, 3)), dec) for v in vals] == target:
        return text, "rounded twice"
    return text, "no"


def check(n_boot):
    results = []
    for col, key in enumerate(RUNS):
        model = ra.Model(key)
        S0 = ra.read_stats(key)  # replication_analysis.py's own output for the assembled layout
        SV = {name: variant_stats(model, name, runs, n_boot) for name, runs in variants(key).items()}
        rows = [(sec, q, stat, (l, g)[col], None, v) for sec, q, stat, l, g, v in CHECKS]
        rows += [(sec, f"{p}: {q}", stat, vals[col], p, v) for sec, q, stat, by_p, v in PERSONA_CHECKS
                 for p, vals in by_p.items()]
        for sec, q, stat, readme, p, v in rows:
            if not readme:
                continue
            got, status = compare(readme, SV[v] if v else S0, stat, p)
            assembled, status_a = compare(readme, S0, stat, p) if v else (got, status)
            results.append(dict(model=model.label.replace(" (legacy)", ""), section=sec, quantity=q, readme=readme,
                                section_inputs=got, assembled=assembled, variant=v or "", status=status,
                                status_assembled=status_a))
    count = lambda field, s: sum(r[field] == s for r in results)
    print(f"{len(results)} README values: from the sections' inputs, {count('status', 'yes')} match, "
          f"{count('status', 'rounded twice')} match once the README's double rounding is undone, "
          f"{count('status', 'no')} don't; from the assembled layout {count('status_assembled', 'yes')} / "
          f"{count('status_assembled', 'rounded twice')} / {count('status_assembled', 'no')}")
    for r in results:
        if r["status"] != "yes" or r["status_assembled"] != "yes":
            print(f"  {r['status']:13} {r['status_assembled']:13} {r['model']}, {r['section']}: {r['quantity']}: README "
                  f"{r['readme']}, recomputed {r['section_inputs']} (assembled {r['assembled']})")
    write_md(results)


def write_md(results):
    count = lambda field, s: sum(r[field] == s for r in results)
    n = len(results)
    L = ["# Legacy check: the replication analysis on the earlier Llama 70B and Gemma 4 31B runs", "",
         "Written by `scripts/replication_legacy.py check`. The earlier sections' result files are assembled into the "
         "replication layout (`outputs/replication/legacy-llama-70b/`, `legacy-gemma4-31b/`; `sources.yaml` there lists "
         "the run behind each condition), `replication_analysis.py` analyzes them, and each README number below is "
         "compared with its recomputed value. CIs use the sections' own resamples (2,000 task bootstraps within origin, "
         "seed 0), so they should match exactly, not just to bootstrap noise.", "",
         "Two recomputed values per README number, shown to 4 decimals:",
         "- **section's inputs:** the conditions the README section used. Where the section took a condition from another "
         "run than the layout does, that run is swapped in (the variant named in brackets):",
         "  - `predict_user`: \"What does the Assistant think the user will choose?\" re-scored A and U, base and final, "
         "on its own pods;",
         "  - `placement_self`: \"Persona placement\" scored the evil self-description's A and C on its pod b; the layout "
         "takes the \"persona as the user\" pod's, which scored all four self readouts together, as the replication will;",
         "  - `readouts_u0`: \"The persona as the user\" took the no-persona U from the predict-user run.",
         "- **assembled layout:** what `replication_analysis.py legacy-...` reports, with one scoring per condition. It "
         "differs from the section's inputs only by re-scoring noise: each section's gate matched the re-scored cells at "
         "r = 1.0000.", "",
         "**Match:** \"yes\" if the recomputed number rounds to the README's at the README's precision, CI ends included. "
         "\"README rounded twice\" if it does only after first rounding to 3 decimals: the README took these numbers from "
         "the sections' printed reports (3 decimals) and rounded again, so a value like 0.5048 printed as 0.505 became "
         "0.51 where 0.50 is right. Each of those was also checked against the section's own saved table at full "
         "precision (`outputs/<section>/*.csv`): the recomputed values are identical to at least 5 decimals.", "",
         f"**Result: of {n} README values, {count('status', 'yes')} reproduce exactly from the sections' inputs, "
         f"{count('status', 'rounded twice')} are README double-rounding (the recomputed value equals the section's own "
         f"saved value), and {count('status', 'no')} differ.** From the assembled layout: "
         f"{count('status_assembled', 'yes')} / {count('status_assembled', 'rounded twice')} / "
         f"{count('status_assembled', 'no')}.", ""]
    odd = [r for r in results if r["status"] != "yes" or r["status_assembled"] != "yes"]
    if odd:
        L += ["Every value that isn't a plain match:", "", "| model | section | quantity | README | section's inputs | "
              "match | assembled layout | match |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['model']} | {r['section']} | {r['quantity']} | {r['readme']} | {r['section_inputs']} | {r['status']} | "
              f"{r['assembled']} | {r['status_assembled']} |" for r in odd]
        L.append("")
    L += ["## The replication's calls on the earlier runs", "",
          "`replication_analysis.py`'s pre-registered calls on the assembled layouts (from "
          "`outputs/replication/<key>/analysis/findings.csv`). The earlier runs have only evil and slacker, and slacker "
          "only in the system prompt, the self-description and roleplay, so the other cells are \"not run\".", "",
          "| finding | " + " | ".join(RUNS[k]["label"].replace(" (legacy)", "") for k in RUNS) + " |",
          "|---|" + "---|" * len(RUNS)]
    calls = {k: ra.read_findings(k) for k in RUNS}
    for f, p in ra.matrix_rows(PERSONAS):
        cells = []
        for k in RUNS:
            r = calls[k].get((f, p))
            cells.append(f"{r['call']}: {r['key_number']}" + (f" ({r['section_call']})" if r["section_call"] else "")
                         if r and r["key_number"] else (r["call"] if r else "–"))
        L.append(f"| {f}. {ra.FINDINGS[f]}{', ' + p if p else ''} | " + " | ".join(cells) + " |")
    L += ["", "## All values", ""]
    for sec in dict.fromkeys(r["section"] for r in results):
        L += [f"### {sec}", "", "| model | quantity | README | recomputed, section's inputs | recomputed, assembled layout "
              "| match |", "|---|---|---|---|---|---|"]
        for r in results:
            if r["section"] != sec:
                continue
            mark = {"yes": "yes", "rounded twice": "README rounded twice", "no": "**no**"}[r["status"]]
            if r["status_assembled"] != r["status"]:
                mark += f" (assembled: {r['status_assembled']})"
            v = f" ({r['variant']})" if r["variant"] else ""
            L.append(f"| {r['model']} | {r['quantity']} | {r['readme']} | {r['section_inputs']}{v} | {r['assembled']} "
                     f"| {mark} |")
        L.append("")
    CHECK_MD.parent.mkdir(parents=True, exist_ok=True)
    CHECK_MD.write_text("\n".join(L))
    print(f"wrote {CHECK_MD}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", nargs="?", default="all", choices=["all", "assemble", "check"])
    ap.add_argument("--n-boot", type=int, default=ra.N_BOOTSTRAP)
    a = ap.parse_args()
    if a.step in ("all", "assemble"):
        for key in RUNS:
            assemble(key)
    if a.step == "all":
        ra.main([*RUNS, "--n-boot", str(a.n_boot)])
    if a.step in ("all", "check"):
        check(a.n_boot)


if __name__ == "__main__":
    main()
