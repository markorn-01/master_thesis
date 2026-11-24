# Known shock/contact fan: 228-case diagnostic matrix

CPU/JAX 0.6.2; production finder unchanged. Exact conserved cell averages and
synthetic broadening controls are explicitly distinguished in the metadata.
All isolated exact shock controls are recovered; isolated contacts have no
false surfaces. Six close-pair cases have no zone candidates, including two
exact-input cases where a dense contact is half a cell behind a real Mach-3
shock. Other cases can retain a biased Mach measurement.

- `summary.csv`: all 228 cases, stage counts, location/Mach scoring, and outcomes.
- `profiles/`: representative centerline fields from actual 3D analysis.
- `illustrative_zone_trace.csv`: numerical operands and decisions for the close
  exact-input example shown in the first column of the profile figure.
- `shock_contact_profiles.png`: exact physical profiles versus analyzed grid
  inputs for close/wide pairs.
- `recovery_by_separation.png`: outcomes by gap, contact density, and filter
  width for one grid phase at 64^3.
- `verification.json`: reference states, conservation residuals, outcomes, and
  source hashes. Completion does not mean every case was recovered.
- `validation.json`: local unit/control validation and artifact integrity.

The separate `../shock_contact_fixed_gap_n032_n064_n128/` study varies only
resolution for one fixed physical solution. Main interpretation and limitations:
`../../experiments/shock/SHOCK_CONTACT.md`. Neither experiment proves that a
fully resolved shock is missed in the actual bubble collision.
