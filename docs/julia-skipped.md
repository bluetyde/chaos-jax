# Task A7 skipped: Julia is not available

Date: 2026-10-27.

`TODO.md` says to run SciMLSensitivity `NILSS` on Lorenz-63 only if `julia --version` works in the environment. If it does not, write this file and move on.

## What I checked

- Command: `julia --version`.
- Result: `julia: command not found`. `which julia` and `which juliaup` also find nothing.

## What I did not do

- I did not install Julia. The task makes the run conditional on Julia already working here. An install would also need new outbound hosts, and the owner has not allowed them.
- I did not read the SciMLSensitivity documentation or the source, and I did not write any Julia code. No number in this repository comes from Julia.

## What this leaves open

- A cross-check of the Lorenz-63 sensitivities with SciMLSensitivity `NILSS` (and `ForwardLSS`, `AdjointLSS`, `NILSAS`) is not done. Four Python codes already agree with each other (`nilss-jax`, my LSS, my NILSS and `fds`), so a Julia run would add a fifth implementation, not a different kind of method.
- To do it later, the owner would need an environment where `julia --version` works, or would need to allow the Julia download hosts and tell me to install it. I would then write predictions and gates first, as for the other tasks.
