#!/bin/bash -l
#SBATCH --job-name=svmp-def-diffu
#SBATCH --account=balzadlb_0000
#SBATCH --partition=cpu_filler
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-core=1
#SBATCH --output=/lustre/nurans63/svmp-frosch/logs/%x-%j.out
#
# Runs a deformation-diffusion test case with a given preconditioner on a
# scratch copy of tests/cases/def_diffu under RUN_ROOT (the cases refer to
# each other's meshes), then summarizes the Newton and linear convergence.
#   sbatch --ntasks=16 BuildScripts/Elysium-RUB/run-case-job.sh plaque_short trilinos-frosch [LS tolerance] [LS max iterations]
# Arguments: case (plaque_short), preconditioner (trilinos-frosch), and
# optionally values replacing the case's linear solver <Tolerance> 1e-6 and
# <Max_iterations> 100.
# Pass options as arguments, not with sbatch --export=ALL,...: that copies
# the submitting shell's environment into the job, and a partial module setup
# there (e.g. from a non-interactive ssh command) leaves the modules in
# env.sh unknown and MPI failing in MPI_Init.
set -o pipefail
unset SLURM_EXPORT_ENV

SVMP_SOURCE_DIR=${SVMP_SOURCE_DIR:-$HOME/dev/svmultiphysics/svMultiPhysics-frosch}
EXE=${EXE:-$SVMP_SOURCE_DIR/build/svMultiPhysics-build/bin/svmultiphysics}
CASE=${1:-plaque_short}
PREC=${2:-trilinos-frosch}
LSTOL=${3:-}
LSMAXIT=${4:-}
RUN_ROOT=${RUN_ROOT:-/lustre/nurans63/svmp-frosch/runs}
NP=${SLURM_NTASKS:-1}

source $SVMP_SOURCE_DIR/BuildScripts/Elysium-RUB/env.sh

T=$RUN_ROOT/${CASE}_${PREC}_np${NP}_${SLURM_JOB_ID:-local}
mkdir -p $T
rsync -a --exclude '*-procs' $SVMP_SOURCE_DIR/tests/cases/def_diffu/ $T/
cd $T/$CASE
sed -i -E "s|<Preconditioner> *[a-z0-9-]+ *</Preconditioner>|<Preconditioner> $PREC </Preconditioner>|" solver.xml
[ -n "$LSTOL" ] && sed -i -E "s|<Tolerance> *1e-6 *</Tolerance>|<Tolerance> $LSTOL </Tolerance>|" solver.xml
[ -n "$LSMAXIT" ] && sed -i -E "s|<Max_iterations> *100 *</Max_iterations>|<Max_iterations> $LSMAXIT </Max_iterations>|" solver.xml
LS=$(sed -n '/<LS /,/<\/LS>/p' solver.xml)
echo "=== $CASE  prec=$PREC  np=$NP  LS tol=$(echo "$LS" | grep -o '<Tolerance>[^<]*' | sed 's/<Tolerance>//')  maxit=$(echo "$LS" | grep -o '<Max_iterations>[^<]*' | sed 's/<Max_iterations>//')  node=$(hostname)  dir=$T/$CASE"

start=$(date +%s)
srun --mpi=pmi2 $EXE solver.xml > run.log 2>&1
rc=$?
wall=$(( $(date +%s) - start ))

# Peak memory of the largest rank, from Slurm's accounting of this job step
# (/usr/bin/time is not installed on the compute nodes).
rss=$(sacct -j ${SLURM_JOB_ID}.0 -n -o MaxRSS 2>/dev/null | head -1 | tr -d ' ')
echo "EXIT $rc  wall $wall s  peak memory of the largest rank: ${rss:-n/a}"
echo "errors/NaN/exceptions: $(grep -a -c -i 'error\|nan\|terminat\|exception' run.log)"
echo "linear solves: $(grep -a -c -E '^ *DD +[0-9]+-[0-9]+' run.log) total, $(grep -a -c 'linear system solution has not converged' run.log) not converged"
echo "Krylov its per solve (min/mean/max): $(grep -a -E '^ *DD +[0-9]+-[0-9]+' run.log | sed -E 's/.*\] *[!\[] *([0-9]+) .*/\1/' | awk '{n++; s+=$1; if(min==""||$1<min)min=$1; if($1>max)max=$1} END{if(n) printf "%d / %.1f / %d", min, s/n, max}')"
echo "Newton iterations:"
grep -a -E '^ *DD +[0-9]+-[0-9]+' run.log | cut -c1-120 | sed 's/^/  /'
