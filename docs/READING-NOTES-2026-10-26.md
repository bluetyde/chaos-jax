# Notes from the eight owner-supplied papers

Read on 2026-10-26 while the Phase 2 pull request was in review. Depth for each paper is in
`READING-LOG.md`. Four reading agents made the first notes. I re-checked the claims marked
"checked" myself, against the text or a page image. Other claims are as the agents reported them.

## What this means for finished work

1. **The 0.96 value does not match the Phase 2 oracle.** Lea et al. 2000 give a slope of about
   0.96, from a straight-line fit to one run with averaging time 131 over r from about 25 to 50.
   Eyink et al. 2004 reuse it as 0.9600 +/- 0.0002 and call it the value at r = 28 (checked).
   The +/- is the error of the line fit. It is not a sampling error. The Phase 2 oracle gives
   1.0029 +/- 0.0021 at rho = 28, about 4% higher. Figures 5d and 6a of arXiv 1204.0159 show
   about 1.0 to 1.02. No gate was set on this. The reason for the gap is not known. A wide-range
   slope against a local slope is one possible reason. A check is proposed in the pull request
   comment and has not been run.
2. **Phase 1 Lyapunov method.** It agrees with Benettin 1980 (tangent vector, renormalise every
   s steps, log of the diagonal of R, divide by elapsed time). The two-trajectory variant is not
   described in that paper. No paper here confirms the Lorenz value 0.906 directly. Eyink et al.
   2004 state lambda_1 of about 0.906 at r = 28, and our measured 0.9064 and 0.9018 agree with it.
3. **Wolf 1985 cannot check our exponents.** Its Lorenz case has sigma = 16, R = 45.92, b = 4.
   The values are in bits per second. Converted to natural log per time unit they are about
   1.497, 0 and -22.46. The sum, -30.24 bits/s, is close to the exact trace -21 per time unit
   (-30.30 bits/s). Our sum, -13.6666, matches the exact trace -(10 + 1 + 8/3) for our parameters.
4. **Lorenz 1963** uses our parameters with no rescaling (checked on journal page 136).

## What this means for later phases

- **Phase 3.** Lea et al. 2000 measured the gradient growth directly: |d(zbar)/dr| grows like
  exp(lambda tau), lambda = 0.933 +/- 0.004 for r from 25 to 30. They call this consistent with
  lambda_1. The gradient is of order 1e4 at tau = 2.26 and order 1e100 at tau = 131 (agent report).
  This is a prior result for P3.1 and P3.3. Predictions in `docs/PREDICTIONS-3.md` should be
  written knowing it.
- **Phase 4.** Eyink et al. 2004 say the ensemble-adjoint error falls like N^(1/alpha - 1)
  because the gradient has a heavy tail from trajectories near the unstable fixed point. They
  report about 2.5e8 samples for 10% accuracy at averaging time 2. The gates in `PLAN.md`
  (relative error of the median, relative interquartile range) were set without this. Heavy
  tails make the median and interquartile range the right statistics. Do not change any gate.
- **Ruelle 1997** proves the linear-response formula for uniformly hyperbolic attractors only.
  Lorenz is not uniformly hyperbolic, so any claim that shadowing works at rho = 28 is a
  numerical result, not a theorem.
- **Phase 7.** The DifferentialEquations.jl paper is dated and says nothing about AD through
  solvers. A port will need current documentation.
