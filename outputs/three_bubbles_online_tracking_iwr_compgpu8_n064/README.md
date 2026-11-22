# Verified three-bubble callback run

Completed 64-cubed three-source GPU run: 17 analysis events through t = 0.16,
equal source strengths, the established discrete-budget matching prescription,
JAX 0.6.2, and incremental tracking enabled during the ordered host callbacks.
The user reports execution on compgpu2, GPU 7. The directory's compgpu8 label
was retained from the launch command; it is not hardware provenance.

`equivalence.json` records the GPU verification: all 21 scalar fields per event
agree with maximum finite difference zero; histogram counts and full tracking
records match exactly; sparse catalogs, background slices and final gas states
pass their comparisons. All 45 group observations match. The batch comparison
passes for 32 adjacent results and 64 delayed/terminal results. Result counts
are algorithm records, not physical shock counts.

`copy_checks.json` records local readback. All recorded runner-source hashes
match the local checkout. Online/offline scalar histories and tracking products
were compared again; all 680 exported NPZ fields also match exactly in a
separate readback. Full gas states were not exported for local rechecking;
their final-state comparison is the GPU runner's recorded result.

The final visualization is in `../three_bubble_online_tracking_demo_n064/`.
Its reader joins each group to the original samples, preserves the callback's
statistics, verifies mesh/catalog identity and respects relationship-availability
times. Source-overlap flags and conditional ancestry remain explicit. This
controlled run does not establish physical convergence or turbulent-box scaling.
