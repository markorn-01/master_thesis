# Running the 256³ wind bubble on Helix

These steps are for the bwForCluster Helix login node. Do not run the
simulation directly on a login node.

## One-time setup

Clone the repository and enter it, then create a dedicated virtual environment:

```bash
module purge
module load devel/python
module load devel/cuda
python3 -m venv .venv-helix
.venv-helix/bin/python -m pip install --upgrade pip
.venv-helix/bin/python -m pip install -r cluster/requirements-helix.txt
```

The repository root is used as `PYTHONPATH` by the experiment runner, so the
package does not need a separate editable installation.

## Short GPU smoke test

Request a GPU interactively rather than testing on the login node:

```bash
salloc --partition=gpu-single --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --gres=gpu:A100:1 --time=00:20:00 --mem=8gb
module load devel/python
module load devel/cuda
JAX_PLATFORMS=cuda .venv-helix/bin/python \
  experiments/wind_bubble/check_gpu_environment.py
exit
```

The output must say `Default backend : gpu` and `Matrix test : PASS`.

## Submit the 256³ run

From the repository root:

```bash
sbatch cluster/helix_gpu_256.slurm
squeue --me
```

The script requests one FP64-capable GPU with at least 70 GB of device memory,
96 GB of host memory, and 24 hours. It writes four restart checkpoints and
automatically resumes from the latest one if the job is submitted again.

Monitor a running job with:

```bash
sstat --format=JobId,AveCPU,AveRSS,MaxRSS -j JOB_ID
srun --jobid=JOB_ID --overlap nvidia-smi
```

Inspect the final state with:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
cat slurm-wind256-JOB_ID.out
```

Successful analysis creates:

```text
outputs/wind_bubble_convergence/n256/metrics.json
outputs/wind_bubble_convergence/wind_bubble_convergence.csv
outputs/wind_bubble_convergence/wind_bubble_convergence.png
```

To rebuild the comparison after copying all resolution results into the same
output directory:

```bash
.venv-helix/bin/python \
  experiments/wind_bubble/run_wind_bubble_convergence.py \
  --resolutions 64 128 256 --aggregate-only \
  --output-dir outputs/wind_bubble_convergence
```

Do not request 512³ until the 256³ peak device-memory usage and runtime are
known. The raw five-variable 512³ float32 state is 2.5 GiB, but the solver's
temporary arrays make the actual peak substantially larger.
