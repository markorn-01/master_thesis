# Unequal-source CPU smoke test

Python 3.11.15 / JAX 0.10.2 CPU, 32³ cells, three events over t=0–0.02.
Source strengths 1 and 0.5, with the base coefficient matched to the saved
64³ equal-wind budget. Prescribed energy coefficients are 0.32825707 and
0.16412854. This short run does not test a collision or physical ancestry.

`equivalence.json` passed: three events, 21 scalar fields per event, zero
maximum scalar difference, exact histogram counts, matching sparse surfaces,
background slices and final gas state. Metadata records actual setup and
source hashes. No GPU validation is claimed. The full unequal collision is
prepared for the user's remote `.venv-iwr` run; see
`experiments/wind_bubble/UNEQUAL_BUBBLES.md`.

The t=0.02 inspection flags 256 downstream endpoints inside an injection sphere
at this early coarse state (zero unknown endpoints, zero upstream overlap).
Thus equivalence is a software check, not validation of these early local Mach
or thermodynamic samples. `event_0002.png` is an initialization inspection only.
