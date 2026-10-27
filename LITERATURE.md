# Literature (leads, not yet verified)

Status of every row: SEEN IN A SEARCH RESULT ONLY on 2026-10-26 (abstracts and snippets). Nothing here was read in full. Fill the "read" column during H0 and correct anything wrong.

## Closed or mostly closed
| Topic | Source | What the snippet said | Read? |
|---|---|---|---|
| Simplest 3D quadratic chaotic flows | Sprott, "Some simple chaotic flows", Phys. Rev. E 1994 (sprott.physics.wisc.edu/pubs/paper212.htm) | A systematic search of 3D quadratic ODEs found 19 simple chaotic flows | no |
| Simplest dissipative chaotic flow | Sprott 1997 (paper227) | Third-order jerk system with one quadratic nonlinearity; later work says no simpler quadratic flow is chaotic | no |
| Review | Sprott and Linz 2000 | Says the algebraically simplest quadratic and piecewise-linear chaotic flows are probably identified | no |
| Catalogue | Sprott, "Elegant Chaos" (book) | 280 examples | no |

## Hyperchaos leads
| System | Source | Snippet | Read? |
|---|---|---|---|
| Rössler hyperchaos 1979 | Rössler, Phys. Lett. A 71, 155 (1979) | 4D; commonly used parameters a=0.25, b=3, c=0.5, d=0.05 give two positive exponents. Equations from memory (x'=-y-z, y'=x+a*y+w, z'=b+x*z, w'=-c*z+d*w): UNVERIFIED | no |
| Computer-assisted proof of Rössler hyperchaos | Wilczak, Serrano and Barrio (SIAM paper) | Cited as proving hyperchaos for the 4D system | no |
| 4th and 5th order hyperjerk | Chlouverakis and Sprott, Chaos Solitons Fractals 28 (2006) 739 | Two apparently simplest n=4 chaotic flows plus several hyperchaotic cases for n=4 and 5 | no |
| 4D simplified Lorenz, hidden attractors | Li and Sprott (sprott.physics.wisc.edu/pubs/paper406.htm) | No equilibria, hyperchaotic in some parameter regions | no |
| One nonlinear term, 4D hyperchaos | Acta Phys. Sin. 58, 5201 (doi 10.7498/aps.58.5201); J. Comput. Appl. 30(8) 2045 (2010) | Reports a 4D hyperchaotic system with one nonlinear term; term count not seen | no |
| Seven-term 4D hyperchaos with tanh | Kopp, TWMS 2025 and a related memristor paper | Four linear terms, a tanh term, an absolute-value term, a constant; no equilibria. Sources disagree on the number of nonlinear terms | no |
| Simple 7-term 4D hyperchaos from Sprott C | MDPI Mathematics 2023 | 7 terms, based on the 3D Sprott-C system | no |
| 11-term 4D, two quadratic nonlinearities | EPJ Plus 2022 and Vaidyanathan papers | Examples with published spectra such as (1.398, 0.239, 0, -17.651) for an eleven-term system | no |
| Hyperchaotic flows in arbitrary dimension | Sahle, PhD thesis, Tübingen 2005 | A minimal-variable, minimal-nonlinearity realisation | no |
| Baier and Sahle flow | 2014 paper | N-dimensional flow with the maximal number of positive exponents | no |

## Search methods already published
| Method | Source | Snippet |
|---|---|---|
| Genetic programming for Lorenz-like chaos | Pan and Das 2018 (arXiv 1803.00052); "When Darwin meets Lorenz" (arXiv 1409.7842) | 150+ structures from the Lorenz family; fitness is the largest exponent |
| LLM-guided GP for equations from data | LLM-ODE (arXiv 2603.20910) | Recovers known systems from data; does not design new flows |
| Rare-event search for hidden attractors | Elisov, hidden-attractor-search (GitHub, MIT) | Different goal: find a hidden attractor in a known system |

## Catalogue to compare against
- dysts (GilpinLab, about 135 systems, arXiv 2110.05266): the standard benchmark list of known chaotic systems.

## Tools (not read; licences not checked)
- lyapax (JAX, Lyapunov spectra, v0.2.0), jitcode and jitcdde (C-compiled, Python), ChaosTools.jl (Julia), pyribs (quality diversity). The lyapax page is the author's own comparison and its claims about other tools are unchecked.

## Not searched
- Review articles on hyperchaos (Chaos Solitons Fractals, Nonlinear Dynamics); Julia-side searches; arXiv 2025 to 2026; hyperchaos in maps; any record claim after 2023.
