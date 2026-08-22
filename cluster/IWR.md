# IWR GPU workflow

The IWR GPU environment uses the repository-local `.venv-iwr` environment.
Run long experiments inside `tmux` so they survive a dropped SSH connection.

## Final-state convergence aggregation

The committed GPU convergence study contains all four resolutions. Rebuild its
CSV and figure without rerunning the simulations with:

```bash
cd ~/master_thesis
source .venv-iwr/bin/activate

python experiments/wind_bubble/run_wind_bubble_convergence.py \
  --resolutions 64 128 256 512 \
  --aggregate-only \
  --output-dir outputs/wind_bubble_convergence_gpu
```

## Phase 2 energy-dissipation run

Start with the existing 20-snapshot `64^3` single-bubble experiment:

```bash
cd ~/master_thesis
source .venv-iwr/bin/activate
tmux new -s wind-energy-64

export CUDA_VISIBLE_DEVICES=0
python experiments/wind_bubble/check_gpu_environment.py

XLA_PYTHON_CLIENT_PREALLOCATE=false \
python experiments/wind_bubble/run_single_bubble.py \
  --num-cells 64 \
  --num-snapshots 20 \
  --t-end 0.2 \
  --num-injection-cells 4 \
  --output-dir outputs/single_bubble_energy_n064
```

The Phase 2 analysis creates:

```text
outputs/single_bubble_energy_n064/shock_energy_histories.csv
outputs/single_bubble_energy_n064/shock_energy_histories.png
```

Before running a higher resolution, inspect these CSV diagnostics:

- `forward_valid_flux_fraction` and `reverse_valid_flux_fraction` should be
  close to one whenever the corresponding shock is detected;
- `forward_surface_area_vs_sphere` and `reverse_surface_area_vs_sphere` should
  be close to one for the approximately spherical baseline bubble;
- cumulative energy must be interpreted from the first resolved detection of
  each shock; missing detections are not silently integrated as zero flux;
- `combined_dissipation_to_injected_energy` is a budget diagnostic, not an
  enforced normalization.

After the `64^3` diagnostics are accepted, repeat at `128^3` with a physically
fixed injection radius:

```bash
tmux new -s wind-energy-128
export CUDA_VISIBLE_DEVICES=0

XLA_PYTHON_CLIENT_PREALLOCATE=false \
python experiments/wind_bubble/run_single_bubble.py \
  --num-cells 128 \
  --num-snapshots 20 \
  --t-end 0.2 \
  --num-injection-cells 8 \
  --output-dir outputs/single_bubble_energy_n128
```

Do not schedule temporal `256^3` or `512^3` energy runs until the `64^3` and
`128^3` surface areas, flux coverage, dissipation rates, and cumulative-energy
histories have been compared.
