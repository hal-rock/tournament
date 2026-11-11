# Legacy check: the replication analysis on the earlier Llama 70B and Gemma 4 31B runs

Written by `scripts/replication_legacy.py check`. The earlier sections' result files are assembled into the replication layout (`outputs/replication/legacy-llama-70b/`, `legacy-gemma4-31b/`; `sources.yaml` there lists the run behind each condition), `replication_analysis.py` analyzes them, and each README number below is compared with its recomputed value. CIs use the sections' own resamples (2,000 task bootstraps within origin, seed 0), so they should match exactly, not just to bootstrap noise.

Two recomputed values per README number, shown to 4 decimals:
- **section's inputs:** the conditions the README section used. Where the section took a condition from another run than the layout does, that run is swapped in (the variant named in brackets):
  - `predict_user`: "What does the Assistant think the user will choose?" re-scored A and U, base and final, on its own pods;
  - `placement_self`: "Persona placement" scored the evil self-description's A and C on its pod b; the layout takes the "persona as the user" pod's, which scored all four self readouts together, as the replication will;
  - `readouts_u0`: "The persona as the user" took the no-persona U from the predict-user run.
- **assembled layout:** what `replication_analysis.py legacy-...` reports, with one scoring per condition. It differs from the section's inputs only by re-scoring noise: each section's gate matched the re-scored cells at r = 1.0000.

**Match:** "yes" if the recomputed number rounds to the README's at the README's precision, CI ends included. "README rounded twice" if it does only after first rounding to 3 decimals: the README took these numbers from the sections' printed reports (3 decimals) and rounded again, so a value like 0.5048 printed as 0.505 became 0.51 where 0.50 is right. Each of those was also checked against the section's own saved table at full precision (`outputs/<section>/*.csv`): the recomputed values are identical to at least 5 decimals.

**Result: of 245 README values, 235 reproduce exactly from the sections' inputs, 10 are README double-rounding (the recomputed value equals the section's own saved value), and 0 differ.** From the assembled layout: 235 / 10 / 0.

Every value that isn't a plain match:

| model | section | quantity | README | section's inputs | match | assembled layout | match |
|---|---|---|---|---|---|---|---|
| Llama 3.3 70B | Does post-training make the user's choice more like the Assistant's? | d = r(A_f, U_f) - r(A_f, U_b) | +0.36 [+0.28, +0.43] | +0.3547 [+0.2844, +0.4338] | rounded twice | +0.3547 [+0.2844, +0.4338] | rounded twice |
| Llama 3.3 70B | Personas in the system prompt | slacker: b_per | 0.51 [0.45, 0.55] | +0.5048 [+0.4527, +0.5539] | rounded twice | +0.5048 [+0.4527, +0.5539] | rounded twice |
| Llama 3.3 70B | Persona placement | evil: user: A_user gap | -7.0 | -6.9500 | rounded twice | -6.9500 | rounded twice |
| Llama 3.3 70B | Persona placement | evil: ρ_self - ρ_user | -0.54 [-0.63, -0.46] | -0.5430 [-0.6246, -0.4625] | rounded twice | -0.5429 [-0.6247, -0.4624] | rounded twice |
| Llama 3.3 70B | Personas as roleplay | evil: b_per | 0.41 [0.35, 0.45] | +0.4048 [+0.3535, +0.4533] | rounded twice | +0.4048 [+0.3535, +0.4533] | rounded twice |
| Llama 3.3 70B | The persona as the user | evil: w, P no persona | 0.06 | +0.0548 | rounded twice | +0.0548 | rounded twice |
| Gemma 4 31B | Personas as roleplay | evil: r(A_r, A_s) | 0.81 | +0.8048 | rounded twice | +0.8048 | rounded twice |
| Gemma 4 31B | Personas as roleplay | slacker: b_per | 4.06 [3.82, 4.28] | +4.0546 [+3.8183, +4.2789] | rounded twice | +4.0546 [+3.8183, +4.2789] | rounded twice |
| Gemma 4 31B | Personas as roleplay | slacker: w_r | 0.97 [0.94, 1.01] | +0.9717 [+0.9414, +1.0048] | rounded twice | +0.9717 [+0.9414, +1.0048] | rounded twice |
| Gemma 4 31B | The persona as the user | evil: w, U self-description | 0.22 | +0.2148 | rounded twice | +0.2148 | rounded twice |

## The replication's calls on the earlier runs

`replication_analysis.py`'s pre-registered calls on the assembled layouts (from `outputs/replication/<key>/analysis/findings.csv`). The earlier runs have only evil and slacker, and slacker only in the system prompt, the self-description and roleplay, so the other cells are "not run".

| finding | Llama 3.3 70B | Gemma 4 31B |
|---|---|---|
| 1. Coin tracks the Assistant after post-training | replicates: r 0.92 (base 0.15), shift 0.92 | replicates: r 0.91 (base 0.50), shift 0.90 |
| 2. User's choice becomes more Assistant-like | replicates: d +0.35, r 0.95 (user-choice call: more Assistant-like) | replicates: d +0.22, r 0.90 (user-choice call: more Assistant-like) |
| 3. Assistant's prediction of the user is Assistant-like | replicates: r 0.95, d_P +0.46 (Δ: similar) | replicates: r 0.86, d_P +0.35 (Δ: own choice) |
| 4. Coin follows a persona the user instructs, evil | replicates: b_per 0.36, ρ 0.74 (placement call: weaker) | replicates: b_per 1.66, ρ 0.65 (placement call: weaker) |
| 4. Coin follows a persona the user instructs, slacker | not run | not run |
| 5. Coin follows a self-description less, evil | replicates: Δρ -0.54 (Δ_C: Assistant) | replicates: Δρ -0.11 (Δ_C: Assistant) |
| 5. Coin follows a self-description less, slacker | not run | not run |
| 6. Coin follows a persona framed as roleplay, evil | replicates: w_r 0.73, Δw +0.11 (roleplay call: mix) | replicates: w_r 1.01, Δw -0.26 (roleplay call: persona) |
| 6. Coin follows a persona framed as roleplay, slacker | replicates: w_r 0.74, Δw +0.19 (roleplay call: mix) | replicates: w_r 0.97, Δw +0.03 (roleplay call: persona) |
| 7. Prediction follows a self-description more than A does, evil | replicates: Δw +0.39 (U: replicates) | replicates: Δw +0.97 (U: replicates) |
| 7. Prediction follows a self-description more than A does, slacker | replicates: Δw +0.27 (U: replicates) | replicates: Δw +0.20 (U: doesn't) |

## All values

### Model families

| model | quantity | README | recomputed, section's inputs | recomputed, assembled layout | match |
|---|---|---|---|---|---|
| Llama 3.3 70B | r(A_f, C_f) | 0.92 [0.90, 0.93] | +0.9203 [+0.9045, +0.9347] | +0.9203 [+0.9045, +0.9347] | yes |
| Llama 3.3 70B | r(A_b, C_b) | 0.15 [0.04, 0.26] | +0.1526 [+0.0400, +0.2646] | +0.1526 [+0.0400, +0.2646] | yes |
| Llama 3.3 70B | r(A_f, C_f), non-harmful | 0.88 | +0.8848 | +0.8848 | yes |
| Llama 3.3 70B | r(A_b, C_b), non-harmful | 0.16 | +0.1564 | +0.1564 | yes |
| Llama 3.3 70B | r shift, coin | 0.92 | +0.9190 | +0.9190 | yes |
| Llama 3.3 70B | coin SD, final | 1.15 | +1.1546 | +1.1546 | yes |
| Llama 3.3 70B | coin SD, base | 0.05 | +0.0489 | +0.0489 | yes |
| Llama 3.3 70B | coin benign-harmful, final | 2.13 | +2.1340 | +2.1340 | yes |
| Llama 3.3 70B | coin benign-harmful, base | -0.01 | -0.0067 | -0.0067 | yes |
| Gemma 4 31B | r(A_f, C_f) | 0.91 [0.88, 0.93] | +0.9081 [+0.8830, +0.9296] | +0.9081 [+0.8830, +0.9296] | yes |
| Gemma 4 31B | r(A_b, C_b) | 0.50 [0.43, 0.56] | +0.5004 [+0.4308, +0.5643] | +0.5004 [+0.4308, +0.5643] | yes |
| Gemma 4 31B | r(A_f, C_f), non-harmful | 0.87 | +0.8707 | +0.8707 | yes |
| Gemma 4 31B | r(A_b, C_b), non-harmful | 0.42 | +0.4166 | +0.4166 | yes |
| Gemma 4 31B | r shift, coin | 0.90 | +0.9033 | +0.9033 | yes |
| Gemma 4 31B | coin SD, final | 3.38 | +3.3765 | +3.3765 | yes |
| Gemma 4 31B | coin SD, base | 0.09 | +0.0854 | +0.0854 | yes |
| Gemma 4 31B | coin benign-harmful, final | 5.74 | +5.7408 | +5.7408 | yes |
| Gemma 4 31B | coin benign-harmful, base | 0.07 | +0.0698 | +0.0698 | yes |

### Does post-training make the user's choice more like the Assistant's?

| model | quantity | README | recomputed, section's inputs | recomputed, assembled layout | match |
|---|---|---|---|---|---|
| Llama 3.3 70B | r(A_f, C_f) within origin (pre-registration table) | 0.78 | +0.7817 | +0.7817 | yes |
| Llama 3.3 70B | r(A_b, U_b) | 0.82 | +0.8180 | +0.8180 | yes |
| Llama 3.3 70B | r(A_f, U_b) | 0.60 | +0.6000 | +0.6000 | yes |
| Llama 3.3 70B | r(A_f, U_f) | 0.95 | +0.9547 | +0.9547 | yes |
| Llama 3.3 70B | d = r(A_f, U_f) - r(A_f, U_b) | +0.36 [+0.28, +0.43] | +0.3547 [+0.2844, +0.4338] | +0.3547 [+0.2844, +0.4338] | README rounded twice |
| Llama 3.3 70B | r shift | 0.96 | +0.9554 | +0.9554 | yes |
| Llama 3.3 70B | call | more Assistant-like | more Assistant-like | more Assistant-like | yes |
| Llama 3.3 70B | final user r, non-harmful | 0.94 | +0.9352 | +0.9352 | yes |
| Llama 3.3 70B | final user r, within origin | 0.88 | +0.8768 | +0.8768 | yes |
| Llama 3.3 70B | base user r, within origin | 0.74 | +0.7436 | +0.7436 | yes |
| Llama 3.3 70B | final user b (per Assistant SD) | 1.62 | +1.6151 | +1.6151 | yes |
| Llama 3.3 70B | final user gap | 3.20 | +3.2002 | +3.2002 | yes |
| Llama 3.3 70B | final user SD | 1.69 | +1.6917 | +1.6917 | yes |
| Llama 3.3 70B | base user SD | 0.11 | +0.1120 | +0.1120 | yes |
| Gemma 4 31B | r(A_f, C_f) within origin (pre-registration table) | 0.75 | +0.7452 | +0.7452 | yes |
| Gemma 4 31B | r(A_b, U_b) | 0.80 | +0.8031 | +0.8031 | yes |
| Gemma 4 31B | r(A_f, U_b) | 0.68 | +0.6781 | +0.6781 | yes |
| Gemma 4 31B | r(A_f, U_f) | 0.90 | +0.9019 | +0.9019 | yes |
| Gemma 4 31B | d = r(A_f, U_f) - r(A_f, U_b) | +0.22 [+0.16, +0.30] | +0.2238 [+0.1580, +0.2963] | +0.2238 [+0.1580, +0.2963] | yes |
| Gemma 4 31B | r shift | 0.89 | +0.8946 | +0.8946 | yes |
| Gemma 4 31B | call | more Assistant-like | more Assistant-like | more Assistant-like | yes |
| Gemma 4 31B | final user r, non-harmful | 0.91 | +0.9091 | +0.9091 | yes |
| Gemma 4 31B | final user r, within origin | 0.78 | +0.7820 | +0.7820 | yes |
| Gemma 4 31B | base user r, within origin | 0.65 | +0.6520 | +0.6520 | yes |
| Gemma 4 31B | final user b (per Assistant SD) | 2.85 | +2.8480 | +2.8480 | yes |
| Gemma 4 31B | final user gap | 4.07 | +4.0675 | +4.0675 | yes |
| Gemma 4 31B | final user SD | 3.16 | +3.1577 | +3.1577 | yes |
| Gemma 4 31B | base user SD | 0.19 | +0.1888 | +0.1888 | yes |

### What does the Assistant think the user will choose?

| model | quantity | README | recomputed, section's inputs | recomputed, assembled layout | match |
|---|---|---|---|---|---|
| Llama 3.3 70B | call | similar | similar (predict_user) | similar | yes |
| Llama 3.3 70B | Δ = r(U_f, P_f) - r(U_f, A_f) | +0.01 [-0.00, +0.02] | +0.0056 [-0.0038, +0.0151] (predict_user) | +0.0056 [-0.0038, +0.0151] | yes |
| Llama 3.3 70B | r(U_f, P_f) | 0.96 | +0.9604 (predict_user) | +0.9604 | yes |
| Llama 3.3 70B | r(U_f, A_f) | 0.95 | +0.9547 (predict_user) | +0.9547 | yes |
| Llama 3.3 70B | r(U_f, A_f | P_f) | 0.51 [0.41, 0.60] | +0.5127 [+0.4143, +0.5994] (predict_user) | +0.5126 [+0.4143, +0.5994] | yes |
| Llama 3.3 70B | r(U_f, P_f | A_f) | 0.59 [0.51, 0.67] | +0.5939 [+0.5106, +0.6696] (predict_user) | +0.5939 [+0.5106, +0.6698] | yes |
| Llama 3.3 70B | Δ, non-harmful | -0.00 [-0.02, +0.01] | -0.0043 [-0.0244, +0.0132] (predict_user) | -0.0043 [-0.0244, +0.0132] | yes |
| Llama 3.3 70B | Δ, within origin | +0.02 [-0.01, +0.05] | +0.0175 [-0.0094, +0.0476] (predict_user) | +0.0175 [-0.0094, +0.0476] | yes |
| Llama 3.3 70B | r(A_f, P_f) | 0.95 | +0.9458 (predict_user) | +0.9458 | yes |
| Llama 3.3 70B | r(A_f, P_f), non-harmful | 0.89 | +0.8903 (predict_user) | +0.8903 | yes |
| Llama 3.3 70B | r(A_b, P_b) | 0.73 | +0.7333 (predict_user) | +0.7333 | yes |
| Llama 3.3 70B | r(A_f, P_b) | 0.48 | +0.4824 (predict_user) | +0.4824 | yes |
| Llama 3.3 70B | d_P | +0.46 [+0.37, +0.57] | +0.4634 [+0.3701, +0.5680] (predict_user) | +0.4634 [+0.3701, +0.5680] | yes |
| Llama 3.3 70B | r shift, A and P | 0.95 | +0.9454 (predict_user) | +0.9454 | yes |
| Llama 3.3 70B | P SD, final | 6.16 | +6.1555 (predict_user) | +6.1555 | yes |
| Llama 3.3 70B | slope of P_f on A_f | 5.82 | +5.8219 (predict_user) | +5.8219 | yes |
| Llama 3.3 70B | gap P_f | 12.24 | +12.2396 (predict_user) | +12.2396 | yes |
| Gemma 4 31B | call | own choice | own choice (predict_user) | own choice | yes |
| Gemma 4 31B | Δ = r(U_f, P_f) - r(U_f, A_f) | -0.14 [-0.16, -0.12] | -0.1396 [-0.1642, -0.1155] (predict_user) | -0.1396 [-0.1642, -0.1155] | yes |
| Gemma 4 31B | r(U_f, P_f) | 0.76 | +0.7623 (predict_user) | +0.7623 | yes |
| Gemma 4 31B | r(U_f, A_f) | 0.90 | +0.9019 (predict_user) | +0.9019 | yes |
| Gemma 4 31B | r(U_f, A_f | P_f) | 0.75 [0.71, 0.79] | +0.7465 [+0.7062, +0.7870] (predict_user) | +0.7465 [+0.7062, +0.7870] | yes |
| Gemma 4 31B | r(U_f, P_f | A_f) | -0.08 [-0.22, 0.06] | -0.0761 [-0.2222, +0.0617] (predict_user) | -0.0761 [-0.2222, +0.0617] | yes |
| Gemma 4 31B | Δ, non-harmful | -0.11 [-0.15, -0.08] | -0.1122 [-0.1459, -0.0786] (predict_user) | -0.1122 [-0.1459, -0.0786] | yes |
| Gemma 4 31B | Δ, within origin | -0.20 [-0.30, -0.13] | -0.2036 [-0.3039, -0.1260] (predict_user) | -0.2036 [-0.3039, -0.1260] | yes |
| Gemma 4 31B | r(A_f, P_f) | 0.86 | +0.8636 (predict_user) | +0.8636 | yes |
| Gemma 4 31B | r(A_f, P_f), non-harmful | 0.77 | +0.7651 (predict_user) | +0.7651 | yes |
| Gemma 4 31B | r(A_b, P_b) | 0.72 | +0.7214 (predict_user) | +0.7214 | yes |
| Gemma 4 31B | r(A_f, P_b) | 0.51 | +0.5108 (predict_user) | +0.5108 | yes |
| Gemma 4 31B | d_P | +0.35 [+0.28, +0.44] | +0.3528 [+0.2778, +0.4350] (predict_user) | +0.3528 [+0.2778, +0.4350] | yes |
| Gemma 4 31B | r shift, A and P | 0.86 | +0.8629 (predict_user) | +0.8629 | yes |
| Gemma 4 31B | P SD, final | 6.66 | +6.6584 (predict_user) | +6.6584 | yes |
| Gemma 4 31B | slope of P_f on A_f | 5.75 | +5.7501 (predict_user) | +5.7501 | yes |
| Gemma 4 31B | gap P_f | 14.16 | +14.1603 (predict_user) | +14.1603 | yes |

### Personas in the system prompt

| model | quantity | README | recomputed, section's inputs | recomputed, assembled layout | match |
|---|---|---|---|---|---|
| Llama 3.3 70B | b_0 (coin slope on A_0) | 1.06 | +1.0626 | +1.0626 | yes |
| Llama 3.3 70B | evil: r(A_sys, A_0) | -0.12 | -0.1206 | -0.1206 | yes |
| Llama 3.3 70B | slacker: r(A_sys, A_0) | 0.29 | +0.2927 | +0.2927 | yes |
| Llama 3.3 70B | evil: b_def | 0.30 [0.26, 0.34] | +0.2984 [+0.2580, +0.3368] | +0.2984 [+0.2580, +0.3368] | yes |
| Llama 3.3 70B | slacker: b_def | 0.41 [0.36, 0.47] | +0.4123 [+0.3581, +0.4703] | +0.4123 [+0.3581, +0.4703] | yes |
| Llama 3.3 70B | evil: b_per | 0.49 [0.44, 0.53] | +0.4854 [+0.4376, +0.5304] | +0.4854 [+0.4376, +0.5304] | yes |
| Llama 3.3 70B | slacker: b_per | 0.51 [0.45, 0.55] | +0.5048 [+0.4527, +0.5539] | +0.5048 [+0.4527, +0.5539] | README rounded twice |
| Llama 3.3 70B | evil: w | 0.62 [0.58, 0.66] | +0.6193 [+0.5828, +0.6556] | +0.6193 [+0.5828, +0.6556] | yes |
| Llama 3.3 70B | slacker: w | 0.55 [0.51, 0.59] | +0.5504 [+0.5136, +0.5877] | +0.5504 [+0.5136, +0.5877] | yes |
| Llama 3.3 70B | evil: call | mix | mix | mix | yes |
| Llama 3.3 70B | slacker: call | mix | mix | mix | yes |
| Llama 3.3 70B | evil: total pull b_def + b_per | 0.78 | +0.7839 | +0.7839 | yes |
| Llama 3.3 70B | slacker: total pull b_def + b_per | 0.92 | +0.9171 | +0.9171 | yes |
| Llama 3.3 70B | evil: coin gap C_sys | -0.02 | -0.0213 | -0.0213 | yes |
| Llama 3.3 70B | slacker: coin gap C_sys | 1.26 | +1.2634 | +1.2634 | yes |
| Gemma 4 31B | b_0 (coin slope on A_0) | 3.07 | +3.0662 | +3.0662 | yes |
| Gemma 4 31B | evil: r(A_sys, A_0) | -0.33 | -0.3302 | -0.3302 | yes |
| Gemma 4 31B | slacker: r(A_sys, A_0) | 0.39 | +0.3922 | +0.3922 | yes |
| Gemma 4 31B | evil: b_def | -0.55 [-0.74, -0.36] | -0.5471 [-0.7406, -0.3579] | -0.5471 [-0.7406, -0.3579] | yes |
| Gemma 4 31B | slacker: b_def | 0.24 [0.08, 0.39] | +0.2357 [+0.0760, +0.3920] | +0.2357 [+0.0760, +0.3920] | yes |
| Gemma 4 31B | evil: b_per | 2.55 [2.28, 2.81] | +2.5459 [+2.2763, +2.8094] | +2.5459 [+2.2763, +2.8094] | yes |
| Gemma 4 31B | slacker: b_per | 4.13 [3.87, 4.38] | +4.1334 [+3.8714, +4.3830] | +4.1334 [+3.8714, +4.3830] | yes |
| Gemma 4 31B | evil: w | 1.27 [1.16, 1.42] | +1.2737 [+1.1609, +1.4158] | +1.2737 [+1.1609, +1.4158] | yes |
| Gemma 4 31B | slacker: w | 0.95 [0.91, 0.98] | +0.9460 [+0.9115, +0.9819] | +0.9460 [+0.9115, +0.9819] | yes |
| Gemma 4 31B | evil: call | persona | persona | persona | yes |
| Gemma 4 31B | slacker: call | mix | mix | mix | yes |
| Gemma 4 31B | evil: total pull b_def + b_per | 2.00 | +1.9988 | +1.9988 | yes |
| Gemma 4 31B | slacker: total pull b_def + b_per | 4.37 | +4.3691 | +4.3691 | yes |
| Gemma 4 31B | evil: coin gap C_sys | -6.22 | -6.2229 | -6.2229 | yes |
| Gemma 4 31B | slacker: coin gap C_sys | 4.90 | +4.8951 | +4.8951 | yes |

### Persona placement

| model | quantity | README | recomputed, section's inputs | recomputed, assembled layout | match |
|---|---|---|---|---|---|
| Llama 3.3 70B | evil: user: b_def | 0.17 | +0.1728 | +0.1728 | yes |
| Llama 3.3 70B | evil: user: b_per | 0.36 [0.32, 0.40] | +0.3608 [+0.3206, +0.3989] | +0.3608 [+0.3206, +0.3989] | yes |
| Llama 3.3 70B | evil: user: ρ_user | 0.74 [0.70, 0.78] | +0.7433 [+0.7031, +0.7814] | +0.7433 [+0.7031, +0.7814] | yes |
| Llama 3.3 70B | evil: user: call | weaker | weaker | weaker | yes |
| Llama 3.3 70B | evil: user: coin gap | -0.16 | -0.1578 | -0.1578 | yes |
| Llama 3.3 70B | evil: user: r(A_user, A_e) | 0.98 | +0.9829 | +0.9829 | yes |
| Llama 3.3 70B | evil: user: A_user gap | -7.0 | -6.9500 | -6.9500 | README rounded twice |
| Llama 3.3 70B | evil: self: b_def | 0.73 | +0.7290 (placement_self) | +0.7289 | yes |
| Llama 3.3 70B | evil: self: b_per | 0.10 [0.06, 0.14] | +0.0973 [+0.0579, +0.1390] (placement_self) | +0.0973 [+0.0579, +0.1390] | yes |
| Llama 3.3 70B | evil: self: ρ_self | 0.20 [0.12, 0.28] | +0.2004 [+0.1245, +0.2803] (placement_self) | +0.2005 [+0.1246, +0.2802] | yes |
| Llama 3.3 70B | evil: self: call | weaker | weaker (placement_self) | weaker | yes |
| Llama 3.3 70B | evil: ρ_self - ρ_user | -0.54 [-0.63, -0.46] | -0.5430 [-0.6246, -0.4625] (placement_self) | -0.5429 [-0.6247, -0.4624] | README rounded twice |
| Llama 3.3 70B | evil: self: coin gap | +1.52 | +1.5228 (placement_self) | +1.5225 | yes |
| Llama 3.3 70B | evil: self: r(A_self, A_0) | 0.86 | +0.8620 (placement_self) | +0.8620 | yes |
| Llama 3.3 70B | evil: self: A_self gap | +13.5 | +13.5056 (placement_self) | +13.5056 | yes |
| Gemma 4 31B | evil: user: b_def | 1.22 | +1.2159 | +1.2159 | yes |
| Gemma 4 31B | evil: user: b_per | 1.66 [1.44, 1.87] | +1.6574 [+1.4378, +1.8688] | +1.6574 [+1.4378, +1.8688] | yes |
| Gemma 4 31B | evil: user: ρ_user | 0.65 [0.60, 0.71] | +0.6510 [+0.6006, +0.7053] | +0.6510 [+0.6006, +0.7053] | yes |
| Gemma 4 31B | evil: user: call | weaker | weaker | weaker | yes |
| Gemma 4 31B | evil: user: coin gap | -0.56 | -0.5570 | -0.5570 | yes |
| Gemma 4 31B | evil: user: r(A_user, A_e) | 0.22 | +0.2222 | +0.2222 | yes |
| Gemma 4 31B | evil: user: r(A_user, A_0) | 0.74 | +0.7361 | +0.7361 | yes |
| Gemma 4 31B | evil: user: A_user gap | +7.6 | +7.5983 | +7.5983 | yes |
| Gemma 4 31B | evil: self: b_def | 2.15 | +2.1457 (placement_self) | +2.1457 | yes |
| Gemma 4 31B | evil: self: b_per | 1.37 [1.18, 1.57] | +1.3688 [+1.1834, +1.5651] (placement_self) | +1.3688 [+1.1834, +1.5651] | yes |
| Gemma 4 31B | evil: self: ρ_self | 0.54 [0.47, 0.62] | +0.5377 [+0.4689, +0.6153] (placement_self) | +0.5377 [+0.4689, +0.6153] | yes |
| Gemma 4 31B | evil: self: call | weaker | weaker (placement_self) | weaker | yes |
| Gemma 4 31B | evil: ρ_self - ρ_user | -0.11 [-0.16, -0.06] | -0.1133 [-0.1596, -0.0615] (placement_self) | -0.1133 [-0.1596, -0.0615] | yes |
| Gemma 4 31B | evil: self: coin gap | +1.54 | +1.5421 (placement_self) | +1.5421 | yes |
| Gemma 4 31B | evil: self: r(A_self, A_0) | 0.94 | +0.9413 (placement_self) | +0.9413 | yes |
| Gemma 4 31B | evil: self: A_self gap | +18.3 | +18.3125 (placement_self) | +18.3125 | yes |

### Personas as roleplay

| model | quantity | README | recomputed, section's inputs | recomputed, assembled layout | match |
|---|---|---|---|---|---|
| Llama 3.3 70B | evil: r(A_r, A_s) | 0.98 | +0.9848 | +0.9848 | yes |
| Llama 3.3 70B | slacker: r(A_r, A_s) | 0.98 | +0.9824 | +0.9824 | yes |
| Llama 3.3 70B | evil: b_def | 0.15 [0.11, 0.19] | +0.1485 [+0.1060, +0.1885] | +0.1485 [+0.1060, +0.1885] | yes |
| Llama 3.3 70B | slacker: b_def | 0.22 [0.18, 0.27] | +0.2235 [+0.1818, +0.2672] | +0.2235 [+0.1818, +0.2672] | yes |
| Llama 3.3 70B | evil: b_per | 0.41 [0.35, 0.45] | +0.4048 [+0.3535, +0.4533] | +0.4048 [+0.3535, +0.4533] | README rounded twice |
| Llama 3.3 70B | slacker: b_per | 0.62 [0.57, 0.68] | +0.6243 [+0.5669, +0.6798] | +0.6243 [+0.5669, +0.6798] | yes |
| Llama 3.3 70B | evil: w_r | 0.73 [0.68, 0.80] | +0.7316 [+0.6752, +0.7955] | +0.7316 [+0.6752, +0.7955] | yes |
| Llama 3.3 70B | slacker: w_r | 0.74 [0.70, 0.78] | +0.7364 [+0.6960, +0.7754] | +0.7364 [+0.6960, +0.7754] | yes |
| Llama 3.3 70B | evil: call | mix | mix | mix | yes |
| Llama 3.3 70B | slacker: call | mix | mix | mix | yes |
| Llama 3.3 70B | evil: Δw = w_r - w_s | +0.11 [+0.08, +0.15] | +0.1123 [+0.0821, +0.1489] | +0.1123 [+0.0821, +0.1489] | yes |
| Llama 3.3 70B | slacker: Δw = w_r - w_s | +0.19 [+0.16, +0.21] | +0.1860 [+0.1644, +0.2081] | +0.1860 [+0.1644, +0.2081] | yes |
| Llama 3.3 70B | evil: vs the system prompt | more than the system prompt | more than the system prompt | more than the system prompt | yes |
| Llama 3.3 70B | slacker: vs the system prompt | more than the system prompt | more than the system prompt | more than the system prompt | yes |
| Llama 3.3 70B | evil: coin gap C_r | -0.24 | -0.2408 | -0.2408 | yes |
| Llama 3.3 70B | slacker: coin gap C_r | 0.83 | +0.8294 | +0.8294 | yes |
| Llama 3.3 70B | evil: A_r gap | -5.9 | -5.9034 | -5.9034 | yes |
| Llama 3.3 70B | slacker: A_r gap | 6.9 | +6.8938 | +6.8938 | yes |
| Gemma 4 31B | evil: r(A_r, A_s) | 0.81 | +0.8048 | +0.8048 | README rounded twice |
| Gemma 4 31B | slacker: r(A_r, A_s) | 1.00 | +0.9976 | +0.9976 | yes |
| Gemma 4 31B | evil: b_def | -0.02 [-0.24, 0.18] | -0.0240 [-0.2374, +0.1787] | -0.0240 [-0.2374, +0.1787] | yes |
| Gemma 4 31B | slacker: b_def | 0.12 [-0.02, 0.25] | +0.1179 [-0.0197, +0.2483] | +0.1179 [-0.0197, +0.2483] | yes |
| Gemma 4 31B | evil: b_per | 1.75 [1.49, 1.99] | +1.7452 [+1.4918, +1.9937] | +1.7452 [+1.4918, +1.9937] | yes |
| Gemma 4 31B | slacker: b_per | 4.06 [3.82, 4.28] | +4.0546 [+3.8183, +4.2789] | +4.0546 [+3.8183, +4.2789] | README rounded twice |
| Gemma 4 31B | evil: w_r | 1.01 [0.90, 1.15] | +1.0139 [+0.8988, +1.1480] | +1.0139 [+0.8988, +1.1480] | yes |
| Gemma 4 31B | slacker: w_r | 0.97 [0.94, 1.01] | +0.9717 [+0.9414, +1.0048] | +0.9717 [+0.9414, +1.0048] | README rounded twice |
| Gemma 4 31B | evil: call | persona | persona | persona | yes |
| Gemma 4 31B | slacker: call | persona | persona | persona | yes |
| Gemma 4 31B | evil: Δw = w_r - w_s | -0.26 [-0.37, -0.17] | -0.2598 [-0.3706, -0.1690] | -0.2598 [-0.3706, -0.1690] | yes |
| Gemma 4 31B | slacker: Δw = w_r - w_s | +0.03 [+0.01, +0.04] | +0.0257 [+0.0074, +0.0443] | +0.0257 [+0.0074, +0.0443] | yes |
| Gemma 4 31B | evil: vs the system prompt | less than the system prompt | less than the system prompt | less than the system prompt | yes |
| Gemma 4 31B | slacker: vs the system prompt | more than the system prompt | more than the system prompt | more than the system prompt | yes |
| Gemma 4 31B | evil: coin gap C_r | -1.36 | -1.3587 | -1.3587 | yes |
| Gemma 4 31B | slacker: coin gap C_r | 4.75 | +4.7501 | +4.7501 | yes |
| Gemma 4 31B | evil: A_r gap | -3.9 | -3.8845 | -3.8845 | yes |
| Gemma 4 31B | slacker: A_r gap | 10.9 | +10.9259 | +10.9259 | yes |

### The persona as the user

| model | quantity | README | recomputed, section's inputs | recomputed, assembled layout | match |
|---|---|---|---|---|---|
| Llama 3.3 70B | evil: r(A_self, P_self) | 0.69 | +0.6853 | +0.6853 | yes |
| Llama 3.3 70B | slacker: r(A_self, P_self) | 0.93 | +0.9308 | +0.9308 | yes |
| Llama 3.3 70B | evil: Δ_U | +0.04 [-0.01, +0.11] | +0.0432 [-0.0111, +0.1050] | +0.0432 [-0.0111, +0.1050] | yes |
| Llama 3.3 70B | evil: Δ_U call | similar | similar | similar | yes |
| Llama 3.3 70B | slacker: Δ_U call | not separable | not separable | not separable | yes |
| Llama 3.3 70B | evil: Δ_C | -0.28 [-0.33, -0.23] | -0.2806 [-0.3316, -0.2333] | -0.2806 [-0.3316, -0.2333] | yes |
| Llama 3.3 70B | evil: Δ_C call | Assistant | Assistant | Assistant | yes |
| Llama 3.3 70B | slacker: Δ_C call | not separable | not separable | not separable | yes |
| Llama 3.3 70B | evil: r(U, A | P) | 0.66 [0.59, 0.73] | +0.6566 [+0.5865, +0.7258] | +0.6566 [+0.5865, +0.7258] | yes |
| Llama 3.3 70B | evil: r(U, P | A) | 0.75 [0.67, 0.82] | +0.7529 [+0.6720, +0.8168] | +0.7529 [+0.6720, +0.8168] | yes |
| Llama 3.3 70B | evil: r(C, A | P) | 0.80 [0.77, 0.84] | +0.8042 [+0.7689, +0.8398] | +0.8042 [+0.7689, +0.8398] | yes |
| Llama 3.3 70B | evil: r(C, P | A) | -0.01 [-0.17, 0.14] | -0.0111 [-0.1715, +0.1397] | -0.0111 [-0.1715, +0.1397] | yes |
| Llama 3.3 70B | evil: w, A self-description | 0.18 | +0.1816 | +0.1816 | yes |
| Llama 3.3 70B | slacker: w, A self-description | 0.79 | +0.7874 | +0.7874 | yes |
| Llama 3.3 70B | evil: w, C self-description | 0.12 | +0.1178 | +0.1178 | yes |
| Llama 3.3 70B | slacker: w, C self-description | 0.38 | +0.3773 | +0.3773 | yes |
| Llama 3.3 70B | evil: w, U self-description | 0.38 | +0.3819 | +0.3819 | yes |
| Llama 3.3 70B | slacker: w, U self-description | 0.82 | +0.8207 | +0.8207 | yes |
| Llama 3.3 70B | evil: w, P self-description | 0.57 | +0.5707 | +0.5707 | yes |
| Llama 3.3 70B | slacker: w, P self-description | 1.06 | +1.0588 | +1.0588 | yes |
| Llama 3.3 70B | evil: w, A no persona | 0.00 | +0.0000 | +0.0000 | yes |
| Llama 3.3 70B | slacker: w, A no persona | 0.00 | +0.0000 | +0.0000 | yes |
| Llama 3.3 70B | evil: w, C no persona | 0.13 | +0.1267 | +0.1267 | yes |
| Llama 3.3 70B | slacker: w, C no persona | -0.11 | -0.1073 | -0.1073 | yes |
| Llama 3.3 70B | evil: w, U no persona | 0.12 | +0.1217 (readouts_u0) | +0.1218 | yes |
| Llama 3.3 70B | slacker: w, U no persona | -0.03 | -0.0289 (readouts_u0) | -0.0290 | yes |
| Llama 3.3 70B | evil: w, P no persona | 0.06 | +0.0548 | +0.0548 | README rounded twice |
| Llama 3.3 70B | slacker: w, P no persona | -0.07 | -0.0685 | -0.0685 | yes |
| Llama 3.3 70B | evil: w, C system prompt | 0.62 | +0.6193 | +0.6193 | yes |
| Llama 3.3 70B | slacker: w, C system prompt | 0.55 | +0.5504 | +0.5504 | yes |
| Llama 3.3 70B | evil: gap, A self-description | 13.5 | +13.5056 | +13.5056 | yes |
| Llama 3.3 70B | evil: gap, C self-description | 1.52 | +1.5225 | +1.5225 | yes |
| Llama 3.3 70B | evil: gap, U self-description | 0.99 | +0.9936 | +0.9936 | yes |
| Llama 3.3 70B | evil: gap, P self-description | 1.5 | +1.4896 | +1.4896 | yes |
| Gemma 4 31B | evil: r(A_self, P_self) | -0.25 | -0.2522 | -0.2522 | yes |
| Gemma 4 31B | slacker: r(A_self, P_self) | 0.97 | +0.9682 | +0.9682 | yes |
| Gemma 4 31B | evil: Δ_U | -0.85 [-0.94, -0.76] | -0.8517 [-0.9429, -0.7628] | -0.8517 [-0.9429, -0.7628] | yes |
| Gemma 4 31B | evil: Δ_U call | Assistant | Assistant | Assistant | yes |
| Gemma 4 31B | slacker: Δ_U call | not separable | not separable | not separable | yes |
| Gemma 4 31B | evil: Δ_C | -0.27 [-0.36, -0.17] | -0.2662 [-0.3613, -0.1724] | -0.2662 [-0.3613, -0.1724] | yes |
| Gemma 4 31B | evil: Δ_C call | Assistant | Assistant | Assistant | yes |
| Gemma 4 31B | slacker: Δ_C call | not separable | not separable | not separable | yes |
| Gemma 4 31B | evil: r(U, A | P) | 0.92 [0.90, 0.93] | +0.9157 [+0.8972, +0.9326] | +0.9157 [+0.8972, +0.9326] | yes |
| Gemma 4 31B | evil: r(U, P | A) | 0.54 [0.45, 0.63] | +0.5393 [+0.4503, +0.6274] | +0.5393 [+0.4503, +0.6274] | yes |
| Gemma 4 31B | evil: r(C, A | P) | 0.82 [0.79, 0.85] | +0.8188 [+0.7886, +0.8499] | +0.8188 [+0.7886, +0.8499] | yes |
| Gemma 4 31B | evil: r(C, P | A) | 0.72 [0.67, 0.77] | +0.7208 [+0.6657, +0.7706] | +0.7208 [+0.6657, +0.7706] | yes |
| Gemma 4 31B | evil: w, A self-description | -0.03 | -0.0313 | -0.0313 | yes |
| Gemma 4 31B | slacker: w, A self-description | 0.81 | +0.8132 | +0.8132 | yes |
| Gemma 4 31B | evil: w, C self-description | 0.39 | +0.3895 | +0.3895 | yes |
| Gemma 4 31B | slacker: w, C self-description | 0.69 | +0.6923 | +0.6923 | yes |
| Gemma 4 31B | evil: w, U self-description | 0.22 | +0.2148 | +0.2148 | README rounded twice |
| Gemma 4 31B | slacker: w, U self-description | 0.63 | +0.6265 | +0.6265 | yes |
| Gemma 4 31B | evil: w, P self-description | 0.94 | +0.9429 | +0.9429 | yes |
| Gemma 4 31B | slacker: w, P self-description | 1.01 | +1.0144 | +1.0144 | yes |
| Gemma 4 31B | evil: w, A no persona | 0.00 | +0.0000 | +0.0000 | yes |
| Gemma 4 31B | slacker: w, A no persona | 0.00 | +0.0000 | +0.0000 | yes |
| Gemma 4 31B | evil: w, C no persona | 0.05 | +0.0517 | +0.0517 | yes |
| Gemma 4 31B | slacker: w, C no persona | -0.23 | -0.2256 | -0.2256 | yes |
| Gemma 4 31B | evil: w, U no persona | 0.21 | +0.2114 (readouts_u0) | +0.2114 | yes |
| Gemma 4 31B | slacker: w, U no persona | -0.20 | -0.2001 (readouts_u0) | -0.2001 | yes |
| Gemma 4 31B | evil: w, P no persona | -0.20 | -0.2011 | -0.2011 | yes |
| Gemma 4 31B | slacker: w, P no persona | -0.25 | -0.2477 | -0.2477 | yes |
| Gemma 4 31B | evil: w, C system prompt | 1.27 | +1.2737 | +1.2737 | yes |
| Gemma 4 31B | slacker: w, C system prompt | 0.95 | +0.9460 | +0.9460 | yes |
| Gemma 4 31B | evil: gap, A self-description | 18.3 | +18.3125 | +18.3125 | yes |
| Gemma 4 31B | evil: gap, C self-description | 1.54 | +1.5421 | +1.5421 | yes |
| Gemma 4 31B | evil: gap, U self-description | 5.22 | +5.2154 | +5.2154 | yes |
| Gemma 4 31B | evil: gap, P self-description | -12.4 | -12.3530 | -12.3530 | yes |
