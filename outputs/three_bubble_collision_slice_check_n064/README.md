# Preliminary check of the lower two bubbles' interaction

This reads only the existing three-bubble 64-cubed slice fields and exported
shock catalogs. It changes no solver, detector, thresholds, grouping or display
inputs. `collision_slice_check.png` shows the actual slice and a cross-layer
profile at t=0.08. Counts and all-time planar diagnostics are in `slice_summary.csv`.

## Established from saved data

The central ROI is x=0.45–0.55, y=0.335–0.415, z=0.42–0.58. It contains no
exported detected samples from t=0.07 through 0.16. Earlier, at t=0.05 and
0.06, some detections exist outside the displayed half-cell slab. Thus the
late central gap is already present before grouping or mesh fitting, and is
not solely caused by the thin-slice overlay. This ROI is outside the source
exclusion spheres; it is not a count of every possible interacting front.

Across the saved central layer density rises while the temperature proxy
P/rho falls. At t>=0.06, the in-plane term
dT/dx*drho/dx + dT/dy*drho/dy is negative in all 36 ROI cells.
The production finder requires the full 3D gradient dot product to be positive,
so this is a specific reason to investigate its thermodynamic filtering.

## Not established

The saved plane omits dT/dz*drho/dz and the velocity field. We therefore cannot
reconstruct the production 3D alignment, compression, jump or surface masks.
We cannot yet attribute the gap to one exact production criterion, declare
the layer only a contact, or prove a physical shock was missed. Shock/contact
overlap and resolved gradients may matter; these remain hypotheses.

The next diagnostic is a matching-setup three-bubble replay that records the
unchanged production masks and velocity/thermodynamic profiles in this ROI,
checking the replay products against the existing run. Do not relax the
criteria merely to populate the displayed gap.

Method reference for the gradient/contact filter:
Schaal & Springel, Shock finding on a moving-mesh I, Section 2.3.2,
https://arxiv.org/html/1407.4117 . This reference explains the criterion's
purpose; it does not classify this particular simulated layer.
