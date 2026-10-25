# Online shock analysis: IWR validation

Both experiments used JAX 0.6.2 in `.venv-iwr`, physical H100 GPU 3 on
compgpu11, at commit `bbc48664b0aacfde9c1e798cffd3513a75713958`.
The physical injection radius was 0.0625 and the five explicit analysis times
were 0, 0.05, 0.10, 0.15 and 0.20 (within float32 representation).

| Grid | History fields per event | Maximum scalar difference | Histogram counts | Final gas state |
| --- | ---: | ---: | --- | --- |
| 32³ | 41 | 0 | Exact match | Within rtol=1e-6, atol=1e-8 |
| 64³ | 41 | 0 | Exact match | Within rtol=1e-6, atol=1e-8 |

The six compact-product and shared-analyzer tests passed remotely in 71.926 s.
The copied CSV fields and histogram arrays were rechecked locally and matched
exactly between each online/reference pair. Recorded analyzer and runner source
hashes matched the local files at the time of transfer verification.

Result directories:

- `online_single_bubble_iwr_n032/`
- `online_single_bubble_iwr_n064/`

The online runs retain no full gas snapshot series. The verification runs
intentionally retain small reference snapshot series in memory. This first
implementation transfers dense diagnostic grids to the CPU for reduction;
only compact histories and histograms are written to disk.

These checks establish online/offline agreement, not physical convergence or
performance improvement. Sampled density contrasts can be below one at low
resolution, and positive finite samples alone do not establish a resolved
compressive density jump. Runtime overhead and peak-memory usage remain to be
measured with controlled baselines.
