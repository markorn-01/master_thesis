# Advisory continuation table

Start with `parent_status.csv` and `child_status.csv`. These describe
existing geometric matches; they do not change acceptance or establish
physical identity. `distance_sensitive` is a separate warning.

There are 20 parent records over two adjacent intervals and two search
distances: 10 continuation candidates, 2 ambiguous and 8 multiple-parent
support records. Eight records additionally have distance sensitivity.
Counts include repeats at two distances, not distinct physical shocks.

`control_summary.csv` and `control_details.json` retain the known synthetic
false/missed links. `verification.json` records checks and hashes.
See `experiments/astrolink/CONTINUATION_STATUS.md` for definitions, commands
and limits. This is the completed advisory-status checkpoint, not a new
physical tracking algorithm.
