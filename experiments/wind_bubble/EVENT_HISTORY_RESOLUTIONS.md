# Resolution sensitivity of two-bubble group histories

## Question and result

Does the fixed event tracker recover the same broad history at 64³ and 128³?
**The early outer-group combination and subsequent outer continuation repeat.
Inner detections occur earlier at higher resolution, and exact fragment ancestry
is not established by this comparison.**

| Observation | 64³ | 128³ |
| --- | --- | --- |
| Two early groups | t=0.01–0.07 | t=0.01–0.07 |
| Shared outer combination | t=0.07→0.08 | t=0.07→0.08 |
| Combined outer continuation | t=0.08–0.16 | t=0.08–0.16 |
| First inward-oriented patches | t=0.13 (10 patches) | t=0.10 (14 patches) |
| Two larger inner groups followed through the end | t=0.14–0.16 | t=0.11–0.16 |
| Total group observations | 39 | 53 |
| Continuation segments, including isolated observations | 15 | 21 |
| Identical event sets between the two matching distances | 16/16 intervals | 15/16 intervals |

Counts are observations and segment labels, not numbers of physical shocks.
First detection is not a measured physical formation time. Labels are assigned
independently in each run, with no forced cross-resolution group matching.

## Important limitation: inner combinations

At 64³, t=0.13→0.14 has two selected three-parent combinations at both matching
distances. However, their exclusion objective gaps span approximately
**1.03e-8 to 4.25e-8**, only slightly above the fixed numerical tie tolerance
1e-8. Thus agreement across distances does not imply strongly distinguished
parent subsets. The analogous fragment transition at 128³ occurs at
t=0.10→0.11 and is distance-sensitive, with similarly tiny gaps. Both remain
weak ancestry evidence. The three-partner cap also prevents explaining all
fragments in one combination.

## Fixed setup and what varies

The saved runs have matching source locations, injection radius, ambient state,
adiabatic index, wind speed, domain, units, output times and prescribed discrete
mass/energy rates. Each source's prescribed discrete energy rate is
0.32825707012703415. Budget matching is not proof of equal actual net gas-energy
deposition.

Tracking uses the unchanged controlled implementation: the same coverage,
exclusivity, distance cost, arity, normal angle and tie tolerance. Both runs use
matching distances 1.5/64 and 2.5/64 in domain units. The existing grouping uses
radius 1.5 **native cells**, normal threshold 60 degrees and minimum 20 samples.
Consequently its physical radius is halved at 128³, and a fixed sample cutoff
does not represent the same physical area. This compares the current full
detection/grouping/tracking pipeline across resolutions; it is not an isolated
tracker convergence study or physical convergence claim.

## Figures and measurements

- `resolution_histories.png`: complete 64³ and 128³ graphs on the same time
  axis. Green: shared continuation; purple: shared branch; dashed orange:
  distance-specific event. Vertical labels are local graph segment numbers.
- `resolution_summary.png`: group count, inward-oriented group count, largest
  group fraction of retained samples, and median valid Mach in the largest
  group. Counts and fractions are sample-weighted, not area-weighted.
- Inward orientation means a group's median dot product between its normal
  and nearest-source radial unit vector is negative. This describes geometry;
  it does not assign source ancestry or independently classify physical shocks.
- At t=0.16 both runs retain three groups. Largest-group median Mach is about
  6.618 at 64³ and 6.691 at 128³. Similarity of these values is not convergence.

## Verification and reproduction

All 32 new 64³ distance/interval comparisons pass independent group-occupancy
dynamic programming, including optimum and selected-event exclusion gaps.
Independent sparse geometry checks cover 94 generated hypotheses. The prior
128³ graph is reused without modifying its selections. Sources and inputs are
hash-checked; graph readback checks complete observation accounting and shared
event counts. Nineteen focused tests pass (resolution statistics, history
assembly and event matching).

```sh
python -m unittest tests.experiments.test_event_history_resolutions tests.experiments.test_event_history tests.experiments.test_event_tracking
python -m experiments.wind_bubble.compare_event_history_resolutions --output-dir outputs/two_bubbles_event_history_resolution_n064_n128
```

Use a new output directory. Saved outputs include the 64³ result archive and
graph, selected events with exclusion gaps, group statistics at both
resolutions, per-event summaries, figures and verification hashes. No new
hydrodynamic run or GPU is needed.
