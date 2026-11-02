# 64³ reverse-candidate normal paths at t=0.20 (CPU)

Completed locally with Python 3.12.4, JAX 0.6.2, CPU backend on 2026-11-02.
This is a new simulation, not a reconstruction of saved GPU gas states.
It uses the original 21-event schedule, dt_max=0.20/21, injection radius 0.0625,
and the same hydrodynamic setup. Metadata source hashes identify the inspected
working-tree implementation; no estimator, detector or filtering rule changed.

All 608 valid reverse-candidate endpoints were inspected. Independent ray
interpolation reproduces the shared analyzer's pressure/density samples (maximum
absolute difference 2.98e-8). Independently reconstructing the six pressure
candidates reproduces the selected distances. These checks found no coordinate,
sign or pressure-choice mismatch in these paths; they are not physical validation.

Both selected endpoints are outside the Boolean shock zone and injection region
for every inspected pair. Nevertheless, median endpoint-to-endpoint changes
over a one-cell interval are approximately 30.0%/2.7% for upstream/downstream
pressure and 30.4%/18.6% for density. These are local variations, not errors
against true physical states. An extremum can hide variation when only opposite
ends of an interval are compared; the companion derived checks additionally
measure its full three-point range relative to the central value.

The representative `normal_path.png` shows a pressure minimum on the upstream
side and near-flat downstream pressure with appreciable downstream density
variation. It is chosen by median per-sample density-RH residual, not by the
cleanest-looking reconstruction. Positive distance is upstream along the fixed
normal; negative distance is downstream. Normal velocity is in the simulation
frame. Shading marks the nearest-neighbor Boolean shock zone and injection
region where present. Empty injection shading within the displayed range does
not imply the entire recorded path is outside injection.

The original downstream selection is the farthest of six pressure candidates
for all 608 points. Holding upstream fixed and hypothetically using each existing
downstream candidate gives median density-RH residuals from about -16.3% at the
nearest to -2.8% at the farthest. No alternative choice is used to replace a
measurement. This is a sensitivity check, not a proposed estimator or proof
that any alternative state is correct.

`normal_paths.npz` records all sampled rays, candidates, selected states and
geometry; `endpoint_checks.csv` contains per-point checks. No full gas arrays
are written. Readback checks passed and are in `readback_checks.json`.
For population plots, interval overlap checks, limitations and repeatable
postprocessing, see [`../reverse_path_diagnostics/README.md`](../reverse_path_diagnostics/README.md).
