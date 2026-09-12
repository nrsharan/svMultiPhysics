#!/bin/bash -l
#SBATCH --job-name=svmp-tests
#SBATCH --account=balzadlb_0000
#SBATCH --partition=cpu_filler
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-core=1
#SBATCH --output=/lustre/nurans63/svmp-frosch/logs/%x-%j.out
#
# Runs the svMultiPhysics tests (default: tests/test_def_diffu.py, which runs
# each case on 1, 3 and 4 processes) with the build from build-job.sh.
#   sbatch --export=ALL,TESTS="test_def_diffu.py" BuildScripts/Elysium-RUB/test-job.sh
# VENV is a Python environment with pytest, meshio and numpy:
#   python3 -m venv ~/dev/svmultiphysics/venv-tests
#   ~/dev/svmultiphysics/venv-tests/bin/pip install pytest meshio numpy
set -o pipefail
unset SLURM_EXPORT_ENV

SVMP_SOURCE_DIR=${SVMP_SOURCE_DIR:-$HOME/dev/svmultiphysics/svMultiPhysics-frosch}
VENV=${VENV:-$HOME/dev/svmultiphysics/venv-tests}
TESTS=${TESTS:-test_def_diffu.py}

source $SVMP_SOURCE_DIR/BuildScripts/Elysium-RUB/env.sh

cd $SVMP_SOURCE_DIR/tests
$VENV/bin/python -m pytest $TESTS -v -rs
