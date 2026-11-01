# Initial strict comparison and compilation audit

This first CPU run intentionally remains recorded as **failed** in metadata.
Do not cite it as a passing workflow validation. Its point catalogs and figures
are diagnostic products from the initial implementation.

The initial online path compiled the primary payload, while the offline
reference evaluated sampled densities eagerly. Strict pointwise rtol=1e-6,
atol=1e-8 checks exposed downstream-density differences up to 2.623e-6 absolute
and 3.522e-6 relative. Histograms and scalar histories still agreed.

`compilation_audit.py` reproduces eager-versus-compiled density sampling on
**identical** stored input states, with JAX 0.6.2 on CPU. The resulting
`compilation_audit.json` reproduces the event-by-event differences seen in this
first comparison. This isolates compilation mode as the cause for this case.
Run the audit from the repository root in a compatible environment.

The offline reference was then changed to compile the primary payload for
positions/offsets and sampled densities, while retaining the established host
analyzer for Mach and flux reference fields. The numerical tolerances were not
relaxed. The subsequent passing experiment and its figures are in:

`../online_sparse_cpu_validation_n032_compiled_reference/`
