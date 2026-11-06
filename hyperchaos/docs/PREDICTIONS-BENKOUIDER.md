# Benkouider et al. 2024 reproduction: predictions and gates (written 2026-10-28, before any run)

Paper: Benkouider, Sambas, Bonny, Al Nassan, Moghrabi, Sulaiman, Hassan, Mamat, Sci. Rep. 14:12993 (2024). Read in full as text (14 pages; the circuit, encryption and offset-boosting parts only skimmed).
System (1): `x' = a x - y z + w`, `y' = x z - b y`, `z' = x y - c z`, `w' = -y + d`. Monomials: `ax`, `-yz`, `w`, `xz`, `-by`, `xy`, `-cz`, `-y`, `d` = 9 terms, 3 quadratic. Published (Wolf algorithm, MATLAB): a=8, b=40, c=15, d=-0.1, start (10,1,10,1): (3.036, 0.010, 0, -50.032), DKY 3.061. At c=14: (2.818, 0.012, 0, -48.810). At c=9.4: (0.847, 0, -0.073, -42.171). At c=7: (0, -0.078, -2.202, -36.719).
Divergence is a-b-c, so the published sum should be -47 at c=15: it is -46.986.

## Method
`spectrum_p` (float64 RK4 Benettin QR), T = 20000, 10 blocks, dt 0.01 and 0.005, starts (10,1,10,1) and (1,1,1,1), c in {15, 14, 9.4, 7}. Two starts also test the multistability claim.

## Gates
- **GB1.** At c=15 from (10,1,10,1): lambda1 within 0.05 of 3.036, lambda4 within 0.2 of -50.032, sum within 0.05 of -47.
- **GB2 (the real question).** lambda2 above 3 standard errors and above 0.005 at c=15 and at c=14. Passes only if the second positive exponent is confirmed.
- **GB3.** The two dt values agree on lambda1 to 0.02 (stiffness check, lambda4 near -50).

## Predictions
- P-B1: GB1 passes: 85%.
- P-B2: GB2 passes at c=15: 35%; at c=14: 35%. The paper's lambda2 is 0.010 to 0.012 from a Wolf estimate, the same weak-second-exponent pattern that did not reproduce for Kopp 2025.
- P-B3: GB3 passes: 85%.
- P-B4: the c=9.4 and c=7 rows match to within 0.05 on the three leading exponents: 60%.
- P-B5: the two starts give different attractors at some c in the list (multistability): 40%.
- P-B6: at least one prediction is missed: 70%.

## Not claimed
Whether the system is a 4D record: it has 9 terms, so it does not beat the 7-term systems in `LITERATURE.md`.

## Results (2026-10-28, written after the runs; no gate or prediction above was changed)

Files: `experiments/benkouider_check.py`, `experiments/benkouider_dt.py`, `results/benkouider_check.{log,json}`, `results/benkouider_dt.{log,json}`. float64 RK4 Benettin QR, T = 20000 (follow-up 40000), 10 blocks, CPU, Windows. Total about 5 minutes.

**Step size matters.** At dt = 0.01 (our default) the integration blows up to infinity at c = 15 and c = 14 from both starts (RK4 is unstable: the fourth exponent is about -50 and |x| reaches about 240). dt = 0.005 works. The paper used MATLAB ode45 (adaptive), so it did not meet this.

| c | start | dt | exponents (se of the second) | published |
|---|---|---|---|---|
| 15 | (10,1,10,1) | 0.005 | 3.0242, 0.0050 (0.0011), -0.0014, -50.0157 | 3.036, 0.010, 0, -50.032 |
| 15 | (10,1,10,1) | 0.0025 (T 40000) | 3.0339, 0.0034 (0.0009), -0.0006, -50.0361 | |
| 15 | (10,1,10,1) | 0.00125 (T 40000) | 3.0303, 0.0018 (0.0006), -0.0001, -50.0319 | |
| 14 | (10,1,10,1) | 0.005 / 0.0025 / 0.00125 | second exponent 0.0028 / 0.0027 / 0.0035 | 2.818, 0.012, 0, -48.810 (ours: 2.836, 2.838, 2.831 first; -48.83 fourth) |
| 9.4 | (10,1,10,1) | 0.01 | 0.8266, 0.0015, -0.0832, -42.084 | 0.847, 0, -0.073, -42.171 |
| 7 | (10,1,10,1) | 0.01 | 0.0001, -0.0767, -2.172, -36.719 | 0, -0.078, -2.202, -36.719 |
| 7 | (1,1,1,1) | 0.01 | **1.504, 0.002, -0.007, -40.43** | not given at this start |

**Gates.** GB1 passes (dt 0.005: first exponent 3.024, fourth -50.016, sum -47.008). **GB2 fails**: the second exponent is 0.005 or less at c = 15 and 0.003-0.004 at c = 14, never above the 0.005 line at the finer steps, and it keeps falling with smaller dt (0.0050, 0.0034, 0.0018 at c = 15). The "zero" exponent also sits at -0.001 to -0.003 at dt 0.005, the same size, so the second exponent is not separated from the numerical floor. **GB3 fails**: dt 0.01 diverges, so the two dt values cannot be compared at c = 14 and 15.

**Predictions.** P-B1 hit. P-B2 hit (predicted fail at both c, 65%). P-B3 **missed** (85%: stiffness was not expected to break dt = 0.01). P-B4 hit (c = 9.4 and 7 leading exponents within 0.03). P-B5 hit (40%): at c = 7 the start (1,1,1,1) is chaotic with lambda1 = 1.50 while (10,1,10,1) is periodic, which supports the paper's coexistence claim. P-B6 hit.

**What this says.** First exponent, fourth exponent and the divergence match the paper. The claimed second positive exponent (0.010 to 0.012) is not reproduced: we find at most about 0.005, shrinking with step size. So this is a chaotic system with a possibly very weak second exponent, not a confirmed hyperchaotic one. The block standard errors (10 blocks) are likely too small, so even the small positive values are not firmly above zero. Same pattern as Kopp 2025 (a published weak second exponent from a Wolf estimate that our evaluator does not give). Not shown: whether a much longer run or a smaller step turns it to zero.

Side note for the search code: the H2 and H3 pipelines integrate at dt 0.01 and 0.02. Stiff 4D candidates (exponent below -30) would diverge or give wrong spectra at those steps; any H3 search needs a step-size check on the fourth exponent.
