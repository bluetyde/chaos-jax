# Literature (stage H0, read 2026-10-26)

How to read this file. "Read?" says how much of the source I (Claude) could actually read:
**full** = whole paper text read; **partial** = named sections read; **abstract** = abstract or catalogue page only; **not read** = only seen in a search result. Equations are copied from the source as printed. Where the source was a scanned page image I read it from a rendered image and say so. Term counts are mine, made with the definitions in `PLAN.md` (a term is one monomial with a nonzero coefficient over all four equations, constants included; a nonlinear term has degree 2 or more). The exponents are the authors' own numbers; nothing here was re-run.

## Summary in plain words

- **Smallest known 4D hyperchaotic flow, as far as I could find: 7 terms.** Six papers print a 7-term 4D system with two positive exponents. I found none with 6 or fewer terms.
- **Fewest nonlinear terms: 1, and that cannot be beaten, only tied** (a flow with no nonlinear term is linear and cannot have a bounded chaotic attractor; this is my reasoning, not a quoted result). The best system on both counts is Eq. (13) of Chlouverakis and Sprott 2006: 7 terms, 1 nonlinear term, published exponents (0.132, 0.035, 0, -1.25) at A = 3.6. The authors themselves call it "conjectured to be the simplest hyperchaotic flow". Its one nonlinear term is of degree 5 (`x'''` times `x^4`), so it is not quadratic.
- **Best system that is purely quadratic: Li and Sprott 2014**, 7 terms, 2 quadratic terms, exponents (0.0704, 0.0128, 0, -1.0832) at a = 2.6, b = 0.44. The second exponent is small.
- **Rössler 1979** is bigger: 9 terms, 1 nonlinear term. The equations I had "from memory" are right (checked against a later paper that prints them).
- **Confidence.** Fairly good that no published 4D hyperchaotic flow has fewer than 7 terms among the sources I could open. Not good enough to write the word "record" yet, for these reasons: I could not read the original Rössler paper; I could not read Sprott's book *Elegant Chaos* (section 6.7 is on hyperchaotic systems); two Chinese-language 2009/2010 papers were only partly read; the 2022 Sheet paper (4D, 7 terms, based on Sprott B) was not found; and no one has published a systematic, ordered search for hyperchaos by term count (my 4 queries found none). A 2025 paper (Kopp and Samuilik) claims 7 terms is "the minimum number of terms necessary" for 4D hyperchaos, but it gives no search or proof, so I treat that as an unsupported claim.
- **What could still beat it:** a 4D hyperchaotic flow with 6 or fewer terms that nobody searched for. The plan's search (H3) is the first systematic attempt I know of. Also note that a 7-term/1-nonlinear flow already exists, so a new system would have to win on terms (6 or fewer), not on nonlinear count.
- **Warning for PLAN H3:** the record system (Eq. 13) is a 4th-order hyperjerk with a degree-5 term. The plan's search (b), "general sparse quadratic 4D systems", can never find it. Search (a), the hyperjerk search, must allow terms up to degree 5 if it is to re-find it.

## A. Smallest known 4D hyperchaotic flows (H0.7 table)

Sorted by terms, then nonlinear terms. "Poly" = polynomial right-hand side (inside the `PLAN.md` definition of the state). Only systems whose source prints two positive exponents are listed as hyperchaotic.

| # | System (source) | Equations as printed (state form where I changed it) | Terms | Nonlinear | Poly | Exponents as published (parameters) | Read? |
|---|---|---|---|---|---|---|---|
| 1 | Chlouverakis and Sprott 2006, Eq. (13) | `x'''' + x''' x^4 + A x'' + x' + x = 0`. State form: `x1'=x2, x2'=x3, x3'=x4, x4' = -x1^4 x4 - A x3 - x2 - x1` | 7 | 1 (degree 5) | yes | (0.132, 0.035, 0, -1.25), A = 3.6, initial (0.1, 0.1, 0.1, 0.1); DL = 3.13 | full |
| 2 | Li and Sprott 2014, Eq. (1) | `x'=y-x, y'=-xz+u, z'=xy-a, u'=-by` | 7 | 2 (quadratic) | yes | (0.0704, 0.0128, 0, -1.0832), a = 2.6, b = 0.44; DKY = 3.0768 | full |
| 3 | Yu et al. 2023, Eq. (1) (Sprott C based) | `x'=yz, y'=x-y, z'=1-xx, u'=ax+bu` | 7 | 2 (quadratic) | yes | (0.099692, 0.078438, -0.004804, -1.174328), a = 1, b = -0.001, initial (1,1,1,1); third exponent is not zero | partial |
| 4 | Li, Sprott, Thio and Zhu, PHYSCON 2015, Eq. (3) | `x'=y-x, y'=-z sgn(x)+u, z'=x sgn(y)-a, u'=-by` | 7 | 2 (signum products) | no | (0.071, 0.022, 0, -1.089), a = 1, b = 0.1 | partial |
| 5 | Kopp 2025 (TWMS), Eq. (1) | `dx1/dt = a(-x1+x2)+x4, dx2/dt = -x3 tanh x1, dx3/dt = |x1| - 1, dx4/dt = -b x1` | 7 | 2 (`x3 tanh x1`, `|x1|`) | no | (0.1385, 0.0201, 0.0076, -0.7864), a = 0.62, b = 0.55, initial (1,1,1,1); (0.0715, 0.0327, -0.0046, -0.7497) at a = 0.65 | partial |
| 6 | Kopp and Samuilik 2025 (Mathematics 13, 2838), Eq. (3) | `dx1/dt = a x2 - (a+alpha) x1 - beta |x4| x1, dx2/dt = -x1 x3, dx3/dt = x1 x2 - b, dx4/dt = x1` | 7 | 3 (`|x4| x1`, `x1 x3`, `x1 x2`) | no | a = 3: (1.0087, 0.0328, -0.0049, -7.7900); a = 5: (0.8859, 0.0556, -0.0071, -10.8456); alpha = 1, beta = 0.1525, b = 90 | partial |
| 7 | Chlouverakis and Sprott 2006, Eq. (12) | `x'''' + 0.1 x''' - (3x - 4x'')(x'')^2 + (x')^3 = -2x^3` | 8 | 4 | yes | (0.141, 0.0168, 0, -0.257), DL = 3.61 | full |
| 8 | Rössler 1979 (as printed in Wilczak, Serrano and Barrio 2016, Eq. 2) | `x' = -(y+z), y' = x+ay+w, z' = b+xz, w' = -cz+dw` | 9 | 1 (`xz`) | yes | none printed in any text I read. Parameters a = 0.25, b = 3, c = 0.5, d = 0.05 (Wilczak et al. also use a = 0.27857, b = 3, c = 0.3, d = 0.05 for a proof) | original: not read; equations via secondary source (partial) |
| 9 | Zhou, Wei and Cheng 2009 (Acta Phys. Sin. 58, 5201), Eq. (1) | `x' = ax - 1.2y, y' = x - 0.1 y z^2, z' = -y - 1.2z - 5w, w' = z + 0.8w` | 9 | 1 (`y z^2`, cubic) | yes | (0.1191, 0.04961, 0, -0.85222), a = 0.66; DL = 3.198 | partial |

Count per system (every monomial listed, so a reader can check me):

1. Eq. (13): `x2`; `x3`; `x4`; `-x1^4 x4`; `-A x3`; `-x2`; `-x1` = 7. Nonlinear: `x1^4 x4`. The paper says "only seven terms" and "only one nonlinearity" (conclusions).
2. Li-Sprott: `y`, `-x` | `-xz`, `u` | `xy`, `-a` | `-by` = 7. Nonlinear: `xz`, `xy`. The paper says "seven terms, two quadratic nonlinearities".
3. Yu 2023: `yz` | `x`, `-y` | `1`, `-xx` | `ax`, `bu` = 7. Nonlinear: `yz`, `xx`. The paper says "seven terms, two nonlinear terms". The printed text is `1 - xx` (a squared x).
4. PHYSCON: `y`, `-x` | `-z sgn(x)`, `u` | `x sgn(y)`, `-a` | `-by` = 7. Nonlinear: the two signum products. The paper says "seven terms". Piecewise linear, so outside the polynomial class.
5. Kopp 2025: `-a x1`, `a x2`, `x4` | `-x3 tanh x1` | `|x1|`, `-1` | `-b x1` = 7. The paper says "four linear terms, one nonlinear term involving tanh, one absolute value term, and a constant". With the plan's definition the absolute value and the tanh product are both nonlinear, so 2. This reconciles the "sources disagree" note: the paper counts 1 and calls the absolute value a separate kind of term.
6. Kopp-Samuilik: `a x2`, `-(a+alpha) x1`, `-beta |x4| x1` | `-x1 x3` | `x1 x2`, `-b` | `x1` = 7. The paper says "seven terms ... three nonlinear terms".
7. Eq. (12): `x2`; `x3`; `x4`; `-0.1 x4`, `+3 x1 x3^2`, `-4 x3^3`, `-x2^3`, `-2 x1^3` = 8. The paper says "eight terms ... four nonlinearities". Matches.
8. Rössler: `-y`, `-z` | `x`, `ay`, `w` | `b`, `xz` | `-cz`, `dw` = 9. Li and Sprott 2014 independently say "nine terms and four parameters" with "a single quadratic nonlinearity", and Chlouverakis and Sprott say "nine terms ... only one" nonlinearity.
9. Zhou 2009: `ax`, `-1.2y` | `x`, `-0.1 y z^2` | `-y`, `-1.2z`, `-5w` | `z`, `0.8w` = 9. The title says "only one nonlinear term".

Remarks on the table.

- Systems 3, 5 and 6 have a third exponent that is close to zero but not zero (-0.0048, 0.0076, -0.0049 and -0.0071). The authors call these hyperchaotic. The plan's H1 rule (second exponent clearly above zero with a standard error) is stricter than what these papers show; row 2's second exponent (0.0128) is also small. Treat all "hyperchaotic" labels here as the authors' numerical claims.
- Li and Sprott say that near the chaos-hyperchaos boundary of their own system it is hard to decide numerically which of the two small exponents is the zero one.
- Chlouverakis and Sprott (Sec. 3) say their search did not reveal simpler n = 4 hyperchaotic flows (for example with fewer nonlinearities) but that this lay beyond the scope of the paper. So their search was not designed to prove minimality.
- n = 5 hyperchaotic case in that paper, Eq. (19): `x5 + 0.46 x4 G(x) + 8.5 x3 + 3.4 x2 + 8.9 x1 + 5.2 x - 1.256 G(x) = 0` with G a step function, exponents (0.0637, 0.0213, 0, -0.038, -0.454). That is 10 terms (4 chain terms, 6 right-hand terms), 2 non-polynomial nonlinear terms. It is 5D, so it is not in the 4D table.

### Sahle thesis (Tübingen 2005): not a table row

The thesis (German, "Hyperchaotische Flüsse in beliebig vielen Dimensionen") uses Rössler's idea in any dimension N: a linear chain plus one variable with one nonlinear term. General form printed as Eq. (4.2): `x1' = a x1 - x2`, `xi' = x(i-1) - x(i+1)` for the middle variables, `xN' = eps + b xN (x(N-1) - d)` (the Greek letter for the constant was lost in the text I could extract; I have called it eps). If I apply it with N = 4 I get 9 terms and 1 nonlinear term (a x1, -x2, x1, -x3, x2, -x4, eps, b x3 x4, -b d x4). **This count is my own reconstruction, not a number printed in the thesis**, and I did not establish that the thesis reports hyperchaos for N = 4 (the N values in the text were lost; the companion paper by Meyer et al. studies only odd N and says even N behaves differently). So it is not in the table.

## B. Sources and what I read

| Source | What it gave | Read? |
|---|---|---|
| Rössler, "An equation for hyperchaos", Phys. Lett. A 71, 155-157 (1979) | The original 4D system. Publisher page returned HTTP 403. | **not read** (listed in `NEEDED-FROM-OWNER.md`) |
| Wilczak, Serrano and Barrio, SIAM J. Appl. Dyn. Syst. 15(1), 356-390 (2016), "Coexistence and dynamical connections between hyperchaos and chaos in the 4D Rössler system: A computer-assisted proof" (open copy at zaguan.unizar.es) | Prints the system as Eq. (2); proves, for a = 0.27857, b = 3, c = 0.3, d = 0.05, a hyperchaotic invariant set (two expanding directions) coexisting with chaotic sets, and says an earlier paper (their ref. [55]) gave a computer-assisted proof of chaos with two expanding directions for the classical values a = 0.25, b = 3, c = 0.5, d = 0.05 | **partial**: abstract, introduction, Sec. 1 and the start of Sec. 4 read; the proofs, Sec. 2-3 and 5-7 not read |
| Chlouverakis and Sprott, Chaos Solitons Fractals 28, 739-746 (2006), "Chaotic hyperjerk systems" | All n = 4 and n = 5 cases (list below) | **full** (text layer plus page images for the equations) |
| Sprott, "Some simple chaotic flows", Phys. Rev. E 50, R647 (1994) | The 19 cases A to S (Section C). Search method: a fourth-order Runge-Kutta search of about 10^8 randomly chosen coefficient sets out of about 10^20, coefficients in -5 to 5 in steps of 0.1, 6 or fewer nonzero terms | **partial**: page 1 (abstract, method) and Table I read from a scanned image (200 dpi render); pages 3-4 not read |
| Sprott and Linz, Int. J. Chaos Theory Appl. 5(2), 3-22 (2000), "Algebraically simple chaotic flows" (open PDF) | Review. Abstract: there is reason to believe the algebraically simplest chaotic flows with quadratic and piecewise-linear nonlinearities have now been identified. It says that Rössler's 1976 example was not found by Sprott's 1994 search, "suggesting that even this extensive search was not exhaustive". Text search found no discussion of hyperchaos | **partial**: abstract, introduction and Sec. 3 "Computer Search" read; the rest only scanned by keyword |
| Sprott, "Simplest dissipative chaotic flow", Phys. Lett. A 228, 271-274 (1997) | The jerk flow `x''' + A x'' - x'^2 + x = 0` (five terms, one quadratic nonlinearity, chaos for 2.017 < A < 2.082; exponents (0.0550, 0, -2.0720) at A = 2.017). Sprott writes that it "is unlikely that any algebraically simpler form ... exists" and that the 1994 search missed it because chaos occurs only over a narrow parameter range | **full** (OCR text of a university-hosted copy; equation also checked on Sprott's own page) |
| Li and Sprott, Int. J. Bifurc. Chaos 24, 1450034 (2014), "Coexisting hidden attractors in a 4-D simplified Lorenz system" | System 2 of the table. States that the hyperchaotic Rössler system has "nine terms and four parameters", and that Chlouverakis and Sprott "proposed what may be the algebraically simplest hyperchaotic snap system" | **full** |
| Zhou, Wei and Cheng, Acta Phys. Sin. 58(8), 5201-5208 (2009), "只有一个非线性项的超混沌系统" | System 9 of the table | **partial**: pages 1-3 of 8 read from page images (abstract, introduction, the system, Lyapunov analysis); circuit and synchronisation sections not read |
| Chen, Chen, Ma and Long, J. Comput. Appl. 30(8), 2045-2048 (2010), "只含一个非线性项的超混沌系统及其控制比较" | Reported 4D hyperchaos with one nonlinear term plus linear-feedback and adaptive control. Full text link on the journal page does not work | **abstract** only (needed from owner) |
| Kopp, TWMS J. Appl. Eng. Math. 15(11), 2584-2598 (2025), tanh 4D system | System 5 | **partial**: introduction, Sec. 2.1-2.2, Table 1 |
| Kopp and Samuilik, Mathematics 13(17), 2838 (2025) | System 6; the minimal-count claim, in the text after Eq. (3): the system "represents the minimum number of terms necessary" for 4D hyperchaos. No search or argument is given | **partial**: abstract, introduction, Sec. 2, start of Sec. 3, Table 2 |
| Yu et al., Mathematics 11(3), 701 (2023) | System 3; says it is inspired by a 7-term 4D hyperchaotic system based on Sprott B by A. T. Sheet (2022), which diverges quickly at (a, b) = (0.1, 0.09) | **partial**: abstract, introduction, Sec. 2 and the exponents; the 5D memristive part not read |
| Li, Sprott, Thio and Zhu, PHYSCON 2015, "A unique signum switch for chaos and hyperchaos" | System 4; repeats that Chlouverakis and Sprott's system "contains a fifth-order nonlinearity that requires four multipliers" | **partial**: abstract, introduction, Sec. 2 |
| Sahle, thesis, Tübingen 2005 (handle 10900/48866) | See the Sahle section above. Aim: a hyperchaotic flow with "minimal number of variables and nonlinear terms" | **partial**: front matter, Ch. 1, the start of Ch. 4; the text layer lost digits and Greek letters |
| Meyer, Bünner, Kittel and Parisi, "Hyperchaos in the generalized Rössler system", arXiv chao-dyn/9906028 (1999) | The N-dimensional generalised Rössler system of Baier and Sahle: `dx/dt = A x - xN e_(N-1)`, `dxN/dt = eps + b xN (x(N-1) - d)`; studies odd N only | **partial**: abstract, introduction, Sec. II-III start |
| Pham et al., Arch. Control Sci. 30(1), 23-45 (2020) | A 7-term 4D hyperjerk with one nonlinear term, chaotic (one positive exponent, 0.1036), not hyperchaotic | **partial** (abstract, introduction, one exponent line) |
| Can and Uyaroglu, Optoelectron. Adv. Mater. Rapid Commun. 9(5-6), 777-781 (2015), "A new seven-term chaotic attractor and its hyperchaos" | A 3D seven-term system and a 4D extension "hyperchaos case". Equations garbled in the text layer; not counted | **partial** (skimmed) |
| Sprott, *Elegant Chaos* (World Scientific 2010), section 6.7 "Hyperchaotic systems" | Probably lists hyperchaotic flows with term counts | **not read** (book) |
| Sheet 2022, 4D hyperchaos from Sprott B | Cited as [58] in Yu 2023; venue not identified by me | **not read** |
| Al-Rawi and Al-Azzawi, J. Appl. Nonlinear Dyn. 15(3) (2026), "Electronic circuit of new seven-term 4D hyperjerk system" | Abstract says "chaotic attractor"; paywalled | **abstract** |
| Al-Azzawi and Sheet, 5D hyperchaotic system from Sprott C (J. Appl. Eng. Math. 14(2)) | 5D, nine terms | **abstract** |
| Stankevich, Kazakov and Gonchenko, Chaos (2020), "Scenarios of hyperchaos occurrence in 4D Rössler system" | Probably has exponents for the 1979 parameters | **not read** (found only as a citation on a poster) |
| EPJ Plus 2022 (article 13360_2022_Article_2943) and Vaidyanathan papers; 11-term 4D with two quadratic terms | The exponents (1.398, 0.239, 0, -17.651) from the old lead are unchecked | **not read** |
| Singh and Roy, Nonlinear Dyn. (2017), 4D chaotic system with one nonlinear term, eight terms | Chaotic, not hyperchaotic (as far as a search snippet says) | **not read** |
| Sprott, "Elegant Chaos" catalogue of 280 examples | Same book as above | **not read** |

## C. Sprott 1994 Table I: the 19 three-dimensional cases (H0.3)

Copied from the scanned Table I (Phys. Rev. E 50, R647). Cases A to E have five terms and two quadratic nonlinearities; F to S have six terms and one. Exponent columns are (L1, 0, L3). Case A is conservative. Read from an image, so check any digit before using it as a control value. I counted terms and quadratic terms in my transcription of every row with a small script: A to E give 5 terms and 2 nonlinear terms, F to S give 6 and 1, exactly as the paper states, which is a check on the transcription of the equations (not of the exponents).

| Case | Equations | Exponents | DKY |
|---|---|---|---|
| A | `x'=y, y'=-x+yz, z'=1-y^2` | 0.014, 0, -0.014 | 3.000 |
| B | `x'=yz, y'=x-y, z'=1-xy` | 0.210, 0, -1.210 | 2.174 |
| C | `x'=yz, y'=x-y, z'=1-x^2` | 0.163, 0, -1.163 | 2.140 |
| D | `x'=-y, y'=x+z, z'=xz+3y^2` | 0.103, 0, -1.320 | 2.078 |
| E | `x'=yz, y'=x^2-y, z'=1-4x` | 0.078, 0, -1.078 | 2.072 |
| F | `x'=y+z, y'=-x+0.5y, z'=x^2-z` | 0.117, 0, -0.617 | 2.190 |
| G | `x'=0.4x+z, y'=xz-y, z'=-x+y` | 0.034, 0, -0.634 | 2.054 |
| H | `x'=-y+z^2, y'=x+0.5y, z'=x-z` | 0.117, 0, -0.617 | 2.190 |
| I | `x'=-0.2y, y'=x+z, z'=x+y^2-z` | 0.012, 0, -1.012 | 2.012 |
| J | `x'=2z, y'=-2y+z, z'=-x+y+y^2` | 0.076, 0, -2.076 | 2.037 |
| K | `x'=xy-z, y'=x-y, z'=x+0.3z` | 0.038, 0, -0.890 | 2.042 |
| L | `x'=y+3.9z, y'=0.9x^2-y, z'=1-x` | 0.061, 0, -1.061 | 2.057 |
| M | `x'=-z, y'=-x^2-y, z'=1.7+1.7x+y` | 0.044, 0, -1.044 | 2.042 |
| N | `x'=-2y, y'=x+z^2, z'=1+y-2z` | 0.076, 0, -2.076 | 2.037 |
| O | `x'=y, y'=x-z, z'=x+xz+2.7y` | 0.049, 0, -0.319 | 2.154 |
| P | `x'=2.7y+z, y'=-x+y^2, z'=x+y` | 0.087, 0, -0.481 | 2.181 |
| Q | `x'=-z, y'=x-y, z'=3.1x+y^2+0.5z` | 0.109, 0, -0.609 | 2.179 |
| R | `x'=0.9-y, y'=0.4+z, z'=xy-z` | 0.062, 0, -1.062 | 2.058 |
| S | `x'=-x-4y, y'=x+z^2, z'=1+x` | 0.188, 0, -1.188 | 2.151 |

Other 3D references from the same papers: Lorenz `x'=-sigma x+sigma y, y'=rx-xz-y, z'=xy-bz`; Rössler 1976 `x'=-y-z, y'=x+ay, z'=b+xz-cz` (seven terms, one nonlinear term; Sprott 1994 page 1 and Sprott-Linz Eq. 2); Sprott 1997 jerk flow above (five terms, one nonlinear term).

## D. Chlouverakis and Sprott 2006: every case (H0.2)

All are hyperjerks `x^(n) + ... = f(...)`. "Terms" is my count after writing the flow as n first-order equations. Exponents and initial conditions as published. Dimension n = 4 unless noted.

| Eq. | Equation as printed | Terms | Nonlinear | Exponents | Type |
|---|---|---|---|---|---|
| (7) | `x'''' + x''' + 5.2 x'' + 2.7 x' = 4.5(x^2 - 1)` | 8 | 1 | (0.185, 0, -0.483, -0.7), DL 2.38 | chaos |
| (8) | `x'''' + x''' + 10.4 x'' + 8.4 x' = -9.3 x^3 + 2.4 x^2 + 13.6 x - 1` | 10 | 2 | (0.4, 0, -0.23, -1.18), DL 3.15 | chaos |
| (9) | `x'''' + x''' + 2.6 x'' + 2.4 x' = 1.9 x - tan^-1(200 x)` | 8 | 1 (not polynomial) | (0.2, 0, -0.165, -1.03), DL 3.03 | chaos |
| (10a) | `x'''' + x''' + 1.87 x'' + 2.34 x' = 5.45 x + x^2 x''` | 8 | 1 | (0.37, 0, -0.53, -0.84), DL 2.7 | chaos |
| (10b) | `x'''' + 0.25 x''' + 2.2 x'' + 1.2 x' = -1.9 x - 3.2 x^2 x''` | 8 | 1 | (0.284, 0, -0.108, -0.425), DL 3.42 | chaos |
| (11) | `x'''' + x''' + 3 x'' + 2.95 x' = -3.93 sin(x) x'' - 1.47` | 8 | 1 (not polynomial) | (1.94, 0, -0.4, -2.54), DL 3.6 | chaos |
| (12) | `x'''' + 0.1 x''' - (3x - 4x'')(x'')^2 + (x')^3 = -2x^3` | 8 | 4 | (0.141, 0.0168, 0, -0.257), DL 3.61 | **hyperchaos** |
| (13) | `x'''' + x''' x^4 + A x'' + x' + x = 0` | 7 | 1 (degree 5) | (0.132, 0.035, 0, -1.25), A = 3.6, DL 3.13, D2 = 3.126 +- 0.184; chaos to hyperchaos near A about 3.98 | **hyperchaos** |
| (14) | `x'''' + x''' + A x'' + x' = g(x)`, `g(x) = x^2 - 1` | 8 (my count) | 1 | largest exponent 0.078 at A = 2.42, DKY 2.17 | chaos, "simplest" |
| (15) | `x'''' + x''' + A x'' + (x')^2 + x = 0` | 7 (paper says seven) | 1 | largest exponent 0.031 at A = 3.48, DKY 2.068 | chaos, "second simplest" |
| (17) n=5 | `x^(5) + x'''' + 7.068 x''' + 3.94 x'' + 9.17 x' = 3.9(x^2 - 1)` | 10 | 1 | (0.159, 0, -0.242, -0.315, -0.6), DL 2.65 | chaos |
| (18) n=5 | `x^(5) + x'''' + 7.278 x''' + 4 x'' + 9.19 x' = -7.9 x + 2.06 tan^-1(200 x)` | 10 | 1 (not polynomial) | (0.257, 0, -0.217, -0.36, -0.68), DL 3.11 | chaos |
| (19) n=5 | `x^(5) + 0.46 x'''' G(x) + 8.5 x''' + 3.4 x'' + 8.9 x' + 5.2 x - 1.256 G(x) = 0`, G a step function | 10 | 2 (not polynomial) | (0.0637, 0.0213, 0, -0.038, -0.454), DL 4.1 | hyperchaos, 5D |

## E. Searches for review articles, minimal-term claims, and systematic searches

Queries run (WebSearch, 2026-10-26, in this order):

1. `Rössler 1979 "An equation for hyperchaos" Physics Letters A 71 155 equations`
2. `Wilczak Serrano Barrio "Rössler" hyperchaos computer assisted proof SIAM Journal on Applied Dynamical Systems`
3. `Sprott Linz "Algebraically simple chaotic flows" International Journal of Chaos Theory and Applications 2000 pdf`
4. `四维超混沌系统 单个非线性项 物理学报 58 5201 2009`
5. `Kopp 2025 seven-term four-dimensional hyperchaotic system tanh no equilibria TWMS Journal of Applied and Engineering Mathematics`
6. `simple 4D hyperchaotic system seven terms Sprott C MDPI Mathematics 2023`
7. `Sahle Generalized Rössler system hyperchaos dissertation Tübingen 2005 hdl 10900/48866`
8. `review of hyperchaotic systems survey simplest hyperchaotic flow fewest terms four-dimensional`
9. `"simplest hyperchaotic" flow minimal number of terms 4D autonomous ODE one nonlinearity six terms`
10. `systematic computer search hyperchaos four-dimensional ODE random coefficients two positive Lyapunov exponents brute-force search Sprott`
11. `hyperchaotic system with six terms four-dimensional minimal algebraically simple hyperchaos`
12. `hyperchaos "fewest terms" OR "minimal number of terms" OR "simplest" four-dimensional hyperchaotic 2024 2025 arXiv` (extended mode)
13. `Sprott Elegant Chaos section 6.7 hyperchaotic systems simplest four-dimensional hyperchaotic flow terms`
14. `machine learning OR genetic algorithm OR evolutionary search discover new hyperchaotic systems automatically polynomial ODE Lyapunov spectrum`
15. `Rössler hyperchaos 1979 a=0.25 b=3 c=0.5 d=0.05 Lyapunov exponents 0.112 0.02 0 -25 four-dimensional` (the three exponent numbers in this query were my guess, used as a probe; the search did not confirm them and I do not use them)
16. `Chlouverakis Sprott "Chaotic hyperjerk systems" hyperchaotic seven terms conjectured simplest hyperchaotic flow cited`
17. `sprott.physics.wisc.edu hyperchaos simple four-dimensional system two positive Lyapunov exponents Li Sprott` (limited to the Sprott site)

Findings.

- **Dedicated review of hyperchaotic systems: none found.** Queries 8, 11 and 13 returned only individual papers and a Wikipedia article. The closest thing to a review is Sprott's book section 6.7, which I could not open. Reviews may exist that my queries did not reach.
- **Claims of a minimal term count for a 4D hyperchaotic flow:** two. (a) Chlouverakis and Sprott 2006, Eq. (13), "conjectured to be the simplest hyperchaotic flow" (7 terms, one nonlinearity; read in full). (b) Kopp and Samuilik 2025: 7 terms is the minimum number of terms necessary (read in part; no search or proof given). Li and Sprott 2014 and Li et al. 2015 repeat (a) as "may be the algebraically simplest hyperchaotic snap system".
- **Systematic computer search for hyperchaos: none found.** What exists: Sprott 1994 (random search for 3D chaos, 10^8 samples), Sprott 1997 (search of jerk flows for chaos), Chlouverakis and Sprott 2006 (numerical search of hyperjerks, each case "optimized relative to the largest Lyapunov exponent"; hyperchaos found as a by-product and by hand), and the genetic-programming papers already listed below, which look for chaos, not hyperchaos. Nothing I found searches 4D polynomial flows for hyperchaos ordered by term count.
- Everything else found in these queries is hand-built (adding a controller, a memristor or an extra variable to a known 3D system).

## F. Search methods already published (leads from the seeding session; not read in H0)

| Method | Source | Snippet | Read? |
|---|---|---|---|
| Genetic programming for Lorenz-like chaos | Pan and Das 2018 (arXiv 1803.00052); "When Darwin meets Lorenz" (arXiv 1409.7842) | 150+ structures from the Lorenz family; fitness is the largest exponent | not read |
| LLM-guided GP for equations from data | LLM-ODE (arXiv 2603.20910) | Recovers known systems from data; does not design new flows | not read |
| Rare-event search for hidden attractors | Elisov, hidden-attractor-search (GitHub, MIT) | Different goal: find a hidden attractor in a known system | not read |
| Baier and Sahle flow | (year "2014" is from the seeding lead and unchecked) | N-dimensional flow with the maximal number of positive exponents | not read |

## G. Catalogue and tools to compare against (not read; licences not checked)

- dysts (GilpinLab, about 135 systems, arXiv 2110.05266): the standard benchmark list of known chaotic systems. Not compared in H0; part of H5.
- lyapax (JAX, Lyapunov spectra, v0.2.0), jitcode and jitcdde (C-compiled, Python), ChaosTools.jl (Julia), pyribs (quality diversity). The lyapax page is the author's own comparison and its claims about other tools are unchecked.

## H. Claims in earlier project files that this reading changes or checks

- `LITERATURE.md` (old): Rössler equations "from memory, unverified" are **correct** (checked against Wilczak et al. Eq. 2 and the term count of 9 in two other papers). The original 1979 paper itself was still not read.
- `LITERATURE.md` (old): "Sources disagree on the number of nonlinear terms" for Kopp's tanh system. Resolved: the paper counts one nonlinear term and lists the absolute value separately; under the plan's definition it is 2, and it is not a polynomial system.
- `LITERATURE.md` (old): "Sprott 1997 ... later work says no simpler quadratic flow is chaotic". What I read is only Sprott's own hedged remark ("unlikely that any algebraically simpler form ... exists"). It is not a proof.
- `PLAN.md` H2: "Brute-force small 3D polynomial systems on a coarse coefficient grid, as Sprott did." Sprott 1994 used **random sampling** (about 10^8 of 10^20 sets, coefficients in -5 to 5 in steps of 0.1), not a grid, and Sprott and Linz 2000 say it was not exhaustive (it missed Rössler's 1976 system). Sprott 1997 found a 5-term, 1-nonlinear-term chaotic jerk flow that the 1994 search had missed. So the H2 gate "re-finds Sprott's published cases" can fail for sampling reasons, and a 3D rediscovery run can legitimately find cases not in Table I.
- `PLAN.md` H3(b): "general sparse quadratic 4D systems" cannot contain the 7-term/1-nonlinear record (degree 5). See the warning in the summary.
- `PLAN.md` H3(a): "Chlouverakis and Sprott ... found some hyperchaotic cases" is correct; there are exactly two n = 4 hyperchaotic cases (Eq. 12, 13) and one n = 5 case (Eq. 19, not polynomial).
- `PLAN.md` H1 positive controls: "Rössler 1979 hyperchaos (two positive exponents)". No exponent values for it were printed in any text I read; the H1 control will have to establish them (Wilczak et al. say the classical parameter values a = 0.25, b = 3, c = 0.5, d = 0.05 give robust hyperchaos).
- Old lead "11-term 4D ... (1.398, 0.239, 0, -17.651)": still unchecked.

## I. Not searched

Julia-side searches; arXiv 2025 to 2026 beyond the queries above; hyperchaos in maps; hyperchaotic circuits from the 1980s (Matsumoto, Chua, Kobayashi 1986 and later piecewise-linear work); delay systems; any record claim before 2023 other than Chlouverakis and Sprott; whether dysts contains any 4D hyperchaotic system with 7 or fewer terms; the Baier and Klein / Baier and Sahle papers.

## H. H1 cross-check of published exponents (added 2026-10-26, evaluator results)

The H1 evaluator (`docs/PREDICTIONS-H1.md`, results section) recomputed the spectra of several systems in this file at `T = 100000`:
- **Confirmed to about 0.001:** Li and Sprott 2014 (0.0708, 0.0128, -0.0001, -1.0835 against the published 0.0704, 0.0128, 0, -1.0832), Chlouverakis and Sprott Eq. 13 (0.1319, 0.0343, 0, -1.2575 against 0.132, 0.035, 0, -1.25) and Eq. 12 (0.1427, 0.0187, 0, -0.2614 against 0.141, 0.0168, 0, -0.257), and the Sprott 1994 cases B, C, F, S.
- **Rössler 1979:** no published exponents were found; we get (0.1126, 0.0207, 0, -25.10), hyperchaotic.
- **NOT reproduced: Kopp 2025 (a = 0.62, b = 0.55).** Our exponents are (0.1471, 0.0001, 0, -0.7672): chaotic with two zero exponents, not the published (0.1385, 0.0201, 0.0076, -0.7864). Cause unknown (see the H1 results section). **Update 2026-10-27:** the transcription matches the paper's Eq. (1); 145 further runs (docs/PREDICTIONS-KOPP.md) never give a second exponent above 0.0004 at any a from 0.1 to 2, so the published hyperchaos does not reproduce; likely a short or noisy run in the paper (hypothesis). Until it is resolved, do not count Kopp 2025 as a known 7-term hyperchaotic flow; the "six papers with 7 terms" in the summary at the top becomes at most five.
- **Not yet checked:** Yu 2023, PHYSCON 2015, Kopp and Samuilik 2025, Zhou 2009.
