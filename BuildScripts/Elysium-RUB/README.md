# Building and running svMultiPhysics on Elysium (HPC@RUB)

Slurm job scripts for the Elysium cluster of the Ruhr-Universität Bochum,
used to build svMultiPhysics with the `trilinos-frosch` preconditioner and to
run the deformation-diffusion cases (2026-09). Computations, compiling
included, are not allowed on the login nodes, so everything runs as a job.

| Script | What it does |
|---|---|
| `env.sh` | `ml load intel-oneapi-compilers intel-oneapi-mpi intel-oneapi-mkl cmake` (sourced by the jobs) |
| `build-job.sh` | builds Interface2 (MKL backend) and svMultiPhysics on a compute node, about 2 min on 24 cores |
| `test-job.sh` | runs `tests/test_def_diffu.py` (each case on 1, 3 and 4 processes) |
| `run-case-job.sh` | runs a `tests/cases/def_diffu` case with a given preconditioner on a scratch copy under `/lustre`, and summarizes the Newton and linear convergence and the peak memory |

```
cd ~/dev/svmultiphysics/svMultiPhysics-frosch
sbatch BuildScripts/Elysium-RUB/build-job.sh
sbatch BuildScripts/Elysium-RUB/test-job.sh
sbatch --ntasks=16 BuildScripts/Elysium-RUB/run-case-job.sh plaque_short trilinos-frosch
```

Logs go to `/lustre/nurans63/svmp-frosch/logs`, runs to
`/lustre/nurans63/svmp-frosch/runs`; the paths, account (`balzadlb_0000`)
and partition (`cpu_filler`, at most 3 hours) are set at the top of each
script.

## Stack

- Intel oneAPI 2024.1 compilers through the Intel MPI wrappers
  (`mpiicc -cc=icx`, `mpiicpc -cxx=icpx`, with `-fp-model=precise`), Intel MPI
  2021.14.2, MKL 2024.0 (LP64, sequential).
- Trilinos: `~/opt/intel-llvm/trilinos-sv` (global ordinal `int`, with
  ShyLU_DDFROSch and PARDISO_MKL, MKL LP64). With LP64 MKL, FROSch's use of
  Amesos2's umbrella header compiles; with `MKL_ILP64` it would not.
- VTK: `~/opt/intel-llvm/vtk` (9.7.0, static).
- Interface2: develop, built by `build-job.sh` into
  `~/opt/intel-llvm/interface2-latest`. The repository is private; without
  GitHub credentials on the cluster it can be copied as a git bundle
  (`git bundle create Interface2.bundle develop`, then `git clone` the bundle).
- svMultiPhysics: a direct build of `Code/` into
  `build/svMultiPhysics-build`, the layout `tests/conftest.py` expects. The
  test meshes are git LFS files:
  `GIT_LFS_SKIP_SMUDGE=1 git clone --branch <branch> https://github.com/nrsharan/svMultiPhysics.git`
  and `git lfs pull --include "tests/cases/def_diffu/**"`.
- Tests: a Python environment with pytest, meshio and numpy
  (`python3 -m venv ~/dev/svmultiphysics/venv-tests`).

## Pitfalls

- Do not submit with `sbatch --export=ALL,...`: that copies the submitting
  shell's environment into the job, and a partial module setup there (a
  non-interactive `ssh host 'sbatch ...'` does not initialize the modules)
  leaves the modules of `env.sh` "unknown" and every rank fails in
  `MPI_Init`. The run options are therefore script arguments.
- `git` is not installed on the compute nodes.
- Loading the modules prints an Lmod warning about hidden dependencies
  (`gcc-runtime`, `glibc`); it is harmless.
