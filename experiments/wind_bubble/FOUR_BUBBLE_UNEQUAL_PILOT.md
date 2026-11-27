# One bounded unequal-strength four-bubble pilot

## Question and setup

Does the existing online grouping/history workflow remain interpretable when
four sources have unequal strengths? Compare with the completed equal case
in `outputs/four_bubbles_iwr_compgpu2_n064/`.

Change only strengths, in the existing source order, from `[1, 1, 1, 1]` to
`[1.5, 1, 0.75, 0.75]`. Their sum remains four: the prescribed total mass and
mechanical-energy coefficients are unchanged. Wind speed remains 10, so mass
and prescribed power change together. This does not guarantee identical net
gas-energy deposition under the state-dependent EI conversion.

Keep the tetrahedral positions, radius 0.0625, ambient state, 64³ grid,
17 events over `0 ≤ t ≤ 0.16`, source activation from t=0, detector,
grouping/tracking criteria and diagnostic products unchanged. Do not tune
parameters to obtain an expected number of groups. This is not a developed
turbulent box or a performance benchmark.

The original two-source metadata is the source-budget reference; do not
substitute four-source metadata for it. Preflight predicts 160 injection cells
per source and prescribed energy rates approximately
`[0.492386, 0.328257, 0.246193, 0.246193]`.

## Local validation

56 focused tests passed, covering the read-only fragment review, existing
delayed support and streaming products, and unequal-source injection/budget
accounting. The three-event 32³ CPU smoke run passed complete online/offline
and batch tracking checks; see `outputs/four_bubbles_unequal_cpu_smoke_n032/`.
It does not reach the collision regime.

NumPy-only preflight:

```sh
python -m experiments.wind_bubble.run_four_bubbles \
  --preflight-only --num-cells 64 --num-analysis 17 --t-end 0.16 \
  --source-strengths 1.5 1 0.75 0.75 --track-shocks --verify-offline
```

## GPU procedure after committing and pushing

On the remote checkout, preserve any untracked previous outputs that would
conflict with the pull. Leave the stash saved after pulling:

```sh
cd ~/master_thesis
git stash push -u -m "Preserve GPU outputs before unequal four-source pilot" -- outputs/
git pull --ff-only
export THESIS_PY="$HOME/.venvs/master-thesis-py312/bin/python"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi -i 0
"$THESIS_PY" -m experiments.wind_bubble.check_gpu_environment
"$THESIS_PY" -m unittest \
  tests.experiments.test_four_bubbles \
  tests.experiments.test_streaming_fragment_review \
  tests.experiments.test_delayed_support
```

GPU 0 is an example selection, not a reservation. Launch only on a working,
free GPU after these checks pass. Use a fresh output directory:

```sh
"$THESIS_PY" -m experiments.wind_bubble.run_four_bubbles \
  --num-cells 64 --num-analysis 17 --t-end 0.16 \
  --source-strengths 1.5 1 0.75 0.75 \
  --track-shocks --verify-offline --require-gpu \
  --output-dir outputs/four_bubbles_unequal_iwr_compgpu2_n064
cat outputs/four_bubbles_unequal_iwr_compgpu2_n064/equivalence.json
```

## Completion and stopping rule

Require completed metadata and all 17 events; online/offline scalar,
histogram, catalog and slice comparisons; exact full tracking records;
independent batch agreement; and final gas-state agreement. Copy outputs
locally and check source fingerprints before interpreting differences.

Then compare equal/unequal cases at matching times: flow slices and detected
surfaces, Mach/density-ratio distributions, group observations and accepted
versus conditional relationships. Inspect source-overlap/unknown-endpoint
flags and boundary clearance. Do not assume symmetry, six-fragment families,
or a physical source-owner identity. The fragment-review tool intentionally
rejects windows without consensus across its four evidence channels.

Deliver one equal/unequal comparison and a short statement of what changes
and what remains ambiguous. Stop there to review the result. Unequal spacing,
delayed activation, more sources and higher resolution are separate later
decisions.
