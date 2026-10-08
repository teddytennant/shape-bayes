#!/usr/bin/env bash
#
# Smoke run for Shape-Bayes (arXiv 2610.09032). Copied from templates/sbatch.sh
# with the two placeholders filled in.
#
# __VENV_SETUP__   : reuse the shared CUDA jax venv
# __RUN_COMMAND__  : 20 Adam steps of the shape posterior loss, 1x H200

#SBATCH --job-name=arxiv-impl-shape-bayes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail
cd "$(dirname "$0")"
ROOT="${SLURM_SUBMIT_DIR:-/work/ttennant1/arxiv-impl/20261008T102003Z}"
exec > "$ROOT/shape-bayes.out" 2> "$ROOT/shape-bayes.err"

PY="$HOME/arxiv-jax/venv/bin/python"
export JAX_PLATFORMS=cuda
unset JAX_PLATFORM_NAME || true
export PYTHONPATH="$ROOT/shape-bayes/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/shape-bayes"

"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
"$PY" smoke_gpu.py
rc=$?

echo "exit $rc"
exit $rc
