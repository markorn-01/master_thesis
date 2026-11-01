# Online tracking and energy: IWR GPU validation

Validated on compgpu3, physical GPU 0 selected with CUDA_VISIBLE_DEVICES=0,
in `.venv-iwr` with Python 3.12.3 and JAX 0.6.2. The run records revision
`34642eb1c70dddbe3fcea641149024630a212707`. The GPU model is not recorded
in these metadata. All 43 focused/regression tests passed remotely in 87.662 s,
as reported in the session; the test console log was not copied.

## Run

```bash
python -m experiments.wind_bubble.run_online_bubble \
  --num-cells 32 --num-analysis 5 --t-end 0.20 \
  --save-surfaces --track-shocks --integrate-energy \
  --verify-offline --require-gpu \
  --output-dir outputs/online_temporal_iwr_compgpu3_n032
```

Use a new output directory to repeat the run. The physical injection radius
is 0.0625. The five actual times are 0, 0.05, 0.10, 0.15 and 0.20 within
float32 representation. Online output retains no full gas-state series;
offline verification intentionally retains reference states in memory.
No raw gas snapshots are saved in this folder.

## Preserved evidence and checks

The folder was copied via scp and checked locally on 2026-11-01. Original
metadata, equivalence report, histories, catalogs, histogram archives and plots
are preserved. `local_checks.json` records the independent local verification:

- All 41 primary fields agree exactly across five events.
- All 10 tracking rows (two candidates per event) agree exactly.
- All five energy rows agree exactly, including valid-interval and time-coverage fields.
- Histogram arrays and every sparse catalog array agree exactly.
- Reconstructing the histograms from the catalogs reproduces the saved counts.
- Recomputing cumulative energy from the saved per-band powers reproduces
  the energy output without invoking the streaming writer.
- All five recorded source SHA256 hashes matched the local files at import.

The remote runner separately confirmed agreement with the original offline
batch energy integrator and pointwise final gas-state agreement using
rtol=1e-6, atol=1e-8. Full gas arrays are not stored, so the final-state check
could not be repeated locally. These results compare execution paths on the
same GPU, not equality between different CPU/GPU backends.

Run `python outputs/online_temporal_iwr_compgpu3_n032/verify_import.py` from the
repository root with NumPy to repeat the local checks. Source changes will
cause its source-hash check to fail. `SHA256SUMS` contains local preservation
hashes, not a comparison against an independently supplied remote manifest.

## Physical interpretation and scope

Detected surface sample counts are 0, 200, 392, 560 and 872. The forward
candidate changes from `not_yet_detected` to `initialized` at approximately
0.05, then `continued` at the remaining three events. No reverse candidate
is detected in this coarse demonstration; its accumulated energy remains
NaN. Both-band behavior and missing/reacquired detections have synthetic tests,
but this simulation does not validate a resolved reverse-shock history.

Forward accumulated energy begins at zero at t approximately 0.05 and reaches
0.007309131668106052 in code-energy units at t approximately 0.20, over about
0.15 simulation-time units of valid integration intervals. It excludes the
unresolved initial interval and is not the total dissipated energy since
injection began or a complete energy-budget test.

Candidate labels assume the single-source radial geometry. They do not
provide general identity matching for interacting shocks. Continuity flags
are advisory: all detections update candidate state, and finite-power energy
integration does not add a continuity-based filter. Dense diagnostic grids
still transfer to the CPU. No runtime or peak-memory claim is made for these
optional products; earlier performance figures describe compact-only output.

`online_temporal_histories.png` shows radius, radial velocity, instantaneous
thermalization power and accumulated energy. The reverse legend entry has no
curve because no reverse candidate was measured.
