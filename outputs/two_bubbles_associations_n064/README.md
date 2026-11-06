# Candidate temporal associations of two-bubble surfaces

CPU postprocessing of the existing 17-event 64³ GPU sequence; no simulation rerun.
Two spatial partitions, three distance gates and two directional-coverage
thresholds give 12 association configurations. FuzzyCat was reviewed but not run.

See the [method, results and limitations](../../experiments/astrolink/TEMPORAL_ASSOCIATIONS.md).

The example figure retains two candidate parents when the outer groups become
one. It also exposes missing links during inner-surface fragmentation at .13.
These are geometric candidate relationships, not validated persistent shock
identities, source ancestry, material trajectories, or shock formation times.

`pair_scores.csv` retains rejected as well as accepted scores. Local group IDs
must be paired with event indices and segmentation settings. Raw labels for
components below the 20-sample association cutoff remain in `event_*.npz`.
