# GPU-validated selected-state detection audit

Complete 128³ unequal-wind replay on compgpu1, .venv-iwr, JAX 0.6.2 GPU.
All 17 original analysis events are retained. Full states captured at events
10, 12, 14, 15, 16. Ordinary saved products agree with the preceding unequal
128³ run. The production zone masks, exported cell indices and refined
positions reproduce exactly on each captured state.

`audit/verification.json` records these checks and source/full-state hashes.
The five full `.npy` states occupy 209,715,840 bytes. They are kept on disk but
excluded from Git by `audit/.gitignore`. Preserve them on the GPU host/local
machine for reproducibility. Ten compressed source-local crops (~6.2 MiB),
counts, verification and compact online products can be versioned.
The full states are reproducible with the command in
`experiments/wind_bubble/DETECTION_AUDIT.md`; a fresh output directory is required.
Do not mistake excluded large files for deleted data.

Interpretation and profiles: `outputs/two_bubbles_detection_gate_review_n128/`.
A passing audit verifies the saved execution and rejection accounting, not
physical correctness, convergence or the existence/absence of a reverse shock.
