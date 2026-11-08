# Full-event normal comparison

Experimental display meshes for the same saved 128³ two-bubble event at
t=0.16. See `experiments/wind_bubble/EVENT_NORMAL_COMPARISON.md` for method,
results, validation and reproduction commands.

`saved.npz` uses original shock normals. `plane_2cells.npz` and
`plane_3cells.npz` use same-group local plane fits for display reconstruction.
Each preserves original refined positions, physical normals and Mach values.
`fit_valid` is false in the saved mode because no fit is performed.

Coverage and connectivity improve on some measures, but many open edges
remain. Within-group false bridges and temporal stability are not validated.
These artifacts do not replace physical diagnostic products or production
meshes. CSV group labels are event-local, not persistent shock identities.
