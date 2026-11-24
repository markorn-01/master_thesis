# One exact physical shock/contact fan on three grids

Shock Mach 3, contact density factor 8, physical separation 1/64.
The same shock/contact positions, physical time, and Riemann origin are used
at 32^3, 64^3, and 128^3. Exact conserved averages are checked by coarsening
the fine reference. No synthetic broadening or hydrodynamic evolution is used.

Results: no zone candidates at 32^3; detection with Mach 2.236 at 64^3;
detection with the correct Mach 3 at 128^3. Thus better grid resolution restores
recovery in this particular case. CPU/JAX 0.6.2 was used.

`fixed_physical_gap.png` overlays the known physical temperature profile and
the grid inputs. `summary.csv`, `profiles/`, and `verification.json` record
the measurements and provenance. This is not a convergence result for the
evolved bubble simulation and does not establish its central layer's identity.
