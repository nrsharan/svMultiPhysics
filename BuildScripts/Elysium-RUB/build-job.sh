#!/bin/bash -l
#SBATCH --job-name=svmp-build
#SBATCH --account=balzadlb_0000
#SBATCH --partition=cpu_filler
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=/lustre/nurans63/svmp-frosch/logs/%x-%j.out
#
# Builds Interface2 and svMultiPhysics (with the trilinos-frosch
# preconditioner) on a compute node: calculations, compiling included, are
# not allowed on the login nodes.
#   sbatch BuildScripts/Elysium-RUB/build-job.sh
# Paths can be overridden with sbatch --export=ALL,VAR=value:
#   SVMP_SOURCE_DIR        svMultiPhysics checkout
#   BUILD_DIR              svMultiPhysics build directory; the default is the
#                          layout tests/conftest.py expects
#   INTERFACE2_SOURCE_DIR  Interface2 checkout (develop)
#   INTERFACE2_DIR         Interface2 install prefix
#   BUILD_INTERFACE2=OFF   use an existing Interface2 install
#   TRILINOS_DIR           Trilinos install with global ordinal int, and
#                          ShyLU_DDFROSch for trilinos-frosch
#   VTK_DIR                VTK CMake package directory
# Everything uses the Intel oneAPI compilers through the Intel MPI wrappers
# (mpiicc -cc=icx, mpiicpc -cxx=icpx) with -fp-model=precise, as Trilinos was.
set -eo pipefail
unset SLURM_EXPORT_ENV

SVMP_SOURCE_DIR=${SVMP_SOURCE_DIR:-$HOME/dev/svmultiphysics/svMultiPhysics-frosch}
BUILD_DIR=${BUILD_DIR:-$SVMP_SOURCE_DIR/build/svMultiPhysics-build}
INTERFACE2_SOURCE_DIR=${INTERFACE2_SOURCE_DIR:-$HOME/dev/interface2/Interface2-latest}
INTERFACE2_BUILD_DIR=${INTERFACE2_BUILD_DIR:-$HOME/dev/interface2/build_intel_latest}
INTERFACE2_DIR=${INTERFACE2_DIR:-$HOME/opt/intel-llvm/interface2-latest}
TRILINOS_DIR=${TRILINOS_DIR:-$HOME/opt/intel-llvm/trilinos-sv}
VTK_DIR=${VTK_DIR:-$HOME/opt/intel-llvm/vtk/lib64/cmake/vtk-9.7}
NJOBS=${SLURM_CPUS_PER_TASK:-8}

source $SVMP_SOURCE_DIR/BuildScripts/Elysium-RUB/env.sh

C_FLAGS="-cc=icx -fp-model=precise"
CXX_FLAGS="-cxx=icpx -fp-model=precise"
MKL_ARGS=(-DMKL_DIR=$MKLROOT/lib/cmake/mkl -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential)

echo "== $(date)  $(hostname)  $NJOBS cores"
# git is not installed on the compute nodes
commit() { command -v git > /dev/null && git -C $1 log --oneline -1; }

if [ "${BUILD_INTERFACE2:-ON}" = ON ]; then
  echo "== Interface2 $(commit $INTERFACE2_SOURCE_DIR) -> $INTERFACE2_DIR"
  cmake -S $INTERFACE2_SOURCE_DIR -B $INTERFACE2_BUILD_DIR -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=mpiicc -DCMAKE_C_FLAGS="$C_FLAGS" \
        -DCMAKE_CXX_COMPILER=mpiicpc -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
        -DINTERFACE2_MATH_BACKEND=MKL "${MKL_ARGS[@]}" \
        -DINTERFACE2_BUILD_TESTING=OFF -DBUILD_DOCS=OFF \
        -DCMAKE_INSTALL_PREFIX=$INTERFACE2_DIR
  cmake --build $INTERFACE2_BUILD_DIR -j $NJOBS
  cmake --install $INTERFACE2_BUILD_DIR
fi

echo "== svMultiPhysics $(commit $SVMP_SOURCE_DIR) -> $BUILD_DIR"
cmake -S $SVMP_SOURCE_DIR/Code -B $BUILD_DIR -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=mpiicc -DCMAKE_C_FLAGS="$C_FLAGS" \
      -DCMAKE_CXX_COMPILER=mpiicpc -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
      -DSV_USE_MPI=ON -DSV_USE_NOTIMER=ON \
      -DVTK_DIR=$VTK_DIR \
      -DSV_USE_INTERFACE2=ON -DInterface2_DIR=$INTERFACE2_DIR/lib64/cmake/Interface2 \
      -DSV_USE_TRILINOS=ON -DTrilinos_DIR=$TRILINOS_DIR/lib64/cmake/Trilinos \
      "${MKL_ARGS[@]}" \
      "-DCMAKE_PREFIX_PATH=$INTERFACE2_DIR;$TRILINOS_DIR"
grep -E "^SV_TRILINOS_HAS_FROSCH" $BUILD_DIR/CMakeCache.txt
cmake --build $BUILD_DIR -j $NJOBS --target svmultiphysics
ls -la $BUILD_DIR/bin/svmultiphysics
echo "== done $(date)"
