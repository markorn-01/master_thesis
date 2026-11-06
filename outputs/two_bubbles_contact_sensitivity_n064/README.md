# Two-bubble contact-stage sensitivity

CPU postprocessing of seven existing GPU-generated 64³ two-bubble catalogs.
252 AstroLink configurations/orders and 84 normal-compatible graph runs.
No new hydrodynamic run, physical ownership labels, or temporal tracking.

See [method, findings and reproduction commands](../../experiments/astrolink/CONTACT_STAGE.md).

- `contact_evolution.png`: four declared example settings; group counts,
  assigned fraction and agreement between input orders.
- `event_0004_sections.png`, `event_0008_sections.png`, `event_0016_sections.png`:
  thin display sections from full 3D clustering; original versus shuffled order.
- `event_*.npz`: all tested partitions, with original-catalog mappings and
  AstroLink hierarchy/fit information.
- `summary.csv`: all 336 runs, including four explicitly flagged fitting warnings.
- `order_stability.csv`: 336 pairwise comparisons, invariant to label renumbering.
- `metadata.json`: configuration, versions and input/source hashes.
- `verification.json`: successful readback checks; fitting warnings remain flagged.
- `SHA256SUMS`: checksums of output files other than this manifest.

Main finding: graph partitions are stable to input reordering but their merging
depends on geometry thresholds. AstroLink remains order-sensitive near contact.
Neither method has established physical shock identities during collision.
