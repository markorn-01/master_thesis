# Three saved times: original versus fitted display normals

Read `experiments/wind_bubble/TEMPORAL_FITTED_NORMALS.md` for results,
validation, reproduction commands and limits.

- `temporal_fitted_comparison.png`: same camera, spatial bounds and Mach
  scale; original versus two-cell normal-fit cutaways at three times.
- `central_sections.png`: actual intersections of accepted triangles with
  z=0.5. Colors are event-local group labels, not persistent identities.
- Six mesh NPZs preserve positions, physical normals, Mach and group labels;
  candidate normals and fit flags are stored under separate explicit names.
- `summary.csv`, `group_statistics.csv`, `readback.json`: quantitative results
  and checks, including input/source hashes.

The fit improves sample coverage and total piece counts at these times,
but increases open-edge counts. Invalid fits retain saved normals. No
physical diagnostic or production reconstruction is replaced. Within-group
false bridges and temporal flicker remain unvalidated.
