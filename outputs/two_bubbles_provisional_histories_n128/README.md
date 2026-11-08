# Provisional group histories

Open `provisional_histories.png` to see sample-count and median-Mach
histories for the three longest provisional segments. Two early histories
overlap in this symmetric setup. Segments are not physical shock IDs.

- `group_histories.csv`: all 53 retained group observations, segment numbers,
  continuation flags, sample counts and median valid Mach.
- `segments.csv`: all 25 segments, including 18 single-observation segments.
- `continuation_links.csv`: 28 inherited adjacent-event correspondences.
- `controlled_checks.json`: stipulated toy truth and retained/missed links.
- `verification.json`: checks, scope and source/input hashes.

Only unique, nonredundant candidates agreed upon at both tested distances
continue a segment. Ambiguous, multiple-parent or unsupported transitions
start a new segment. This can miss real continuations. No physical mergers
or identities are inferred, and no original matcher decisions or physics
are changed. See `experiments/astrolink/CONTINUATION_SEGMENTS.md`.
