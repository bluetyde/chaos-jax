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
