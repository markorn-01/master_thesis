# Where is the weaker-source inner history lost?

The saved-data audit distinguishes exported surface detections, spatial group
retention and temporal matching. It does not rerun the finder or assign
physical source ownership.

## Main result

| Time | Near-source inward probe samples | In retained groups | In discarded small components |
| --- | ---: | ---: | ---: |
| 0.00–0.14 | 0 | 0 | 0 |
| 0.15 | 12 | 0 | 12 |
| 0.16 | 56 | 24 | 32 |

At 0.15 all 12 samples form one raw 12-sample component. The unchanged cutoff
is 20 samples. At 0.16 the components have sizes 24, 6, 6, 6, 6, 2, 2, 2, 2.
Only the 24-sample component is retained as group 10 / continuation segment 12.
The actual tracking graph has no incoming event for this final group.
Eight final upstream endpoint source overlaps occur among the excluded samples;
none occur in the retained group. There are no downstream overlaps or unknown
sampling positions in the probe populations.

This is chiefly a limitation in the available exported inner-surface detections,
compounded by small-component filtering. The temporal tracker has no retained
predecessor to use. Relaxing the group cutoff could make the 12-sample patch
available at 0.15; it cannot create the missing earlier catalog detections.
No cutoff change or hypothetical link has been accepted in this audit.

## Inspection selection and sensitivity

Default: points whose nearest source is the weaker source, radius <= twice
the injection radius, and normal dot outward radial direction < -0.5. This
is a geometric inspection probe, not production classification or physical
ownership. The catalogs already exclude detected positions inside either
injection sphere. We therefore cannot use them to count all finder detections
before source-region exclusion.

The audit repeats the selection for maximum radius factors 1.5, 2 and 2.25,
and radial-alignment limits 0, -0.5 and -0.8. All nine choices give first probe
samples at event 15 and first retained samples at event 16. A larger-radius
inward selection alone can include interacting outer-surface samples; the
restricted local probe prevents interpreting those as an inner history.
This sensitivity check does not establish completeness outside the probe.

## How to read the figure

`weak_source_diagnosis.png`: top uses full 3D catalog counts. The gray line is
the corresponding geometric selection near the stronger source; purple is all
selected samples near the weaker source; teal is the retained subset.
The symlog axis shows both exact zeros and the large population difference.

Bottom: density slices near the weaker source at 0.15 and 0.16, with nearby
surface samples overlaid. Background z=0.50390625; markers use a slab with
|z-z_slice| <= one grid cell. Counts in the titles distinguish displayed slab
points from full 3D populations. Cyan crosses are below the grouping cutoff;
green circles belong to retained groups. Dashed white circle is the injection
radius; dotted white circle is the inspection radius; star is the source.
The background is gas density, not a shock-zone mask.

## What this does not establish

The sparse catalogs and 2D slices do not retain complete states, candidate
shock-zone masks, surface-selection decisions inside sources or rejected
candidate values. They cannot distinguish physical late shock formation,
source-region exclusion, numerical broadening, detection criteria or surface
selection as the cause of missing earlier samples. No physical absence of a
reverse shock is established. A targeted full-state/finder-stage inspection
would be needed to resolve that question; another temporal matcher is not
justified by these missing inputs alone.

## Artifacts and checks

- `sample_history.csv`: both source-local probe populations and quality flags.
- `components.csv`: raw component size, selected subset and retention status.
- `probe_sensitivity.csv`: all 153 event/setting evaluations.
- `verification.json`: first-event sensitivity, final incoming-edge check,
  source/input hashes, and explicit limits of the available data.

Prior validated input/source hashes are checked, raw component sizes and
retention decisions are independently recounted, and graph observations are
joined by actual event/group IDs. Seven focused tests pass, including empty
input and inward/local selection checks. The rendered figure was inspected.
No existing simulation, partition, graph or measurement was changed.

Reproduce with a new output directory:

```sh
python -m experiments.wind_bubble.inspect_weak_source \
  --output-dir outputs/two_bubbles_weak_source_review_n128
```
