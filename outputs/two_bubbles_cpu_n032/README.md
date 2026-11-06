# Two-source CPU workflow trial

Local CPU/JAX 0.6.2 run: 32³, nine events t=0 to 0.16, two equal sources at
(0.35,0.5,0.5) and (0.65,0.5,0.5). This is a plumbing test before the 64³ GPU run.

equivalence.json reports all nine events passed, 21 primary history fields,
zero finite scalar differences, exact histograms, matching schema-3 catalogs,
matching pressure/density slices and final-state agreement within rtol=1e-6,
atol=1e-8. The raw final gas states were not saved. Both execution paths reuse
the validated primary local estimator; this is not independent physics validation.

At t=0.16 there are 1120 selected surface samples. Minimum detected-surface
clearance across recorded events is approximately 0.13184 (4.22 cells).
The final endpoint positions are known and do not overlap either source.
These checks do not establish freedom from boundary effects or physical accuracy.

Each source occupies 12 injection cells at this coarse resolution. With the
unchanged analytic-volume EI normalization, the discrete energy injection rate
is approximately 0.17905 per source, rather than the nominal 0.5. This is recorded
explicitly in metadata.json. Do not treat this trial as a convergence result or
assume identical expansion to a differently positioned centered source.

Static views show raw detected points and pressure/density slices. They do not
claim that colored point groups identify distinct shocks. Read the experiment
guide at experiments/wind_bubble/TWO_BUBBLES.md for scope and reproduction.

Reproduce with a fresh directory:

```bash
python -m experiments.wind_bubble.run_two_bubbles \
  --num-cells 32 --num-analysis 9 --t-end 0.16 --verify-offline \
  --output-dir outputs/two_bubbles_cpu_n032_repeat
```
