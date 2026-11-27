# Four-bubble 64-cubed GPU pilot: copied and rechecked

The user ran this pilot on compgpu2 and copied its products locally. The
17-event sequence covers t=0 through t=0.16, with four equal tetrahedral
sources and the established per-source prescribed input budget.

## Copy and execution checks

Local rechecking passed:

- All 180 expected simulation/tracking files are present and readable; no
  duplicate or nested copies were found before adding this review.
- All 25 recorded source fingerprints agree with the local checkout. Remote
  absolute paths were mapped to the same repository-relative files.
- All 21 scalar-history fields agree between online and offline at 17 events;
  maximum finite scalar difference is zero. Histogram counts match exactly.
- Sparse catalogs, background slices, full tracking records and partitions
  agree under their established exact/tolerant checks.
- Independent whole-sequence batch assembly agrees with the offline tracker.
- The tracker records 89 group observations and 40 continuation segments.
  It releases its geometry buffer at close, with at most three catalogs
  during processing.
- The GPU report records final gas-state agreement. Complete gas states are
  not exported, so that particular comparison cannot be repeated locally.

`copy_checks.json` contains the local results, per-event group sizes, source
agreement and fingerprints of all original input files. The GPU's original
`equivalence.json` and metadata remain unchanged.

## Initial observations

| Time | Retained geometric groups | Comment |
| --- | ---: | --- |
| 0.01--0.09 | 4 | Four detected groups continue across these events. |
| 0.10 | 10 | Four large groups and six 32-sample fragments. |
| 0.11 | 1 | A shared four-parent combination produces one large group. |
| 0.12 | 2 | One large group and one 20-sample group. |
| 0.13 | 25 | One large group and 24 groups of 20 samples each. |
| 0.14--0.16 | 5 | One large group and four smaller groups. |

At t=0.14, four five-parent combinations link 20 of the 24 small preceding
groups to four larger groups. This is a concrete fragment/association case
for visual review; it is not proof that these are physical mergers or source
ownership. Unmatched groups and any conditional support must be interpreted
from their saved channels rather than forcing identities.

At the final event, the 9,812 detected samples comprise one 8,512-sample group
and four 325-sample groups. The minimum detected-surface boundary clearance
over the sequence is 0.12727 domain units, about 8.15 grid cells. This geometric
clearance does not by itself establish complete boundary independence.

Sampling flags remain important: downstream endpoints overlap injection
regions at early t=0.01--0.03. Upstream endpoints overlap injection regions
for 24 samples at t=0.15 and 36 at t=0.16. No endpoint positions are unknown.
Flags are advisory and do not silently remove those samples or validate
their physical shock states.

## Initial visual review

The existing generic multi-source viewer produced and visually checked:

- `review/event_0009.png`: t=0.09.
- `review/event_0013.png`: t=0.13.
- `review/event_0016.png`: t=0.16.

Each shows the original Mach-colored surface samples and physical pressure
and density slices at z=0.5078125. All source centers are off this slice;
hollow stars indicate their projected positions, not in-plane injections.
Histograms reconstructed from each catalog match its saved bins/counts.
These figures use event-specific color scales; they are initial static
inspections, not a synchronized animation or smooth grouped display.

## Scope and next decision

The four-source workflow passes execution-equivalence checks and produces
interpretable products for review. This is not a runtime benchmark, a
developed turbulent box, or independent validation of complete physical
shock detection and ancestry. The current three-source animation adapters
have case-specific checks and should not be relabeled as four-source movies.

The next bounded review is the t=0.13 to 0.14 fragment transition alongside
the original detected geometry, including unmatched and delayed-support
records. No additional GPU simulation is needed for that review.
