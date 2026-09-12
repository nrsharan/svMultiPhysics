#!/bin/bash
# svMultiPhysics on macOS / Apple silicon, as used for the local builds of
# 2026-09. These are direct builds of Code/ (no superbuild); the program is
# <build directory>/bin/svmultiphysics.
#   TRILINOS=frosch (default)  Trilinos with FROSch, ~/opt/trilinos-16.1.0-int-frosch
#                              (build-frosch; all preconditioners incl. trilinos-frosch)
#   TRILINOS=int               Trilinos without FROSch, ~/opt/trilinos-16.1.0-int
#                              (build-trilinos)
#   TRILINOS=OFF               no Trilinos, fsils only (build)
#   TRILINOS=<prefix>          any other Trilinos install
#   INTERFACE2=OFF             without the Interface2/AceGen library (no
#                              deformation-diffusion equation)
#   VTK_DIR, INTERFACE2_DIR    default to ~/opt/vtk (configure-VTK.sh) and
#                              ~/opt/interface2 (Interface2's
#                              config_scripts/do-config-interface2-macos-accelerate.sh)
# Compilers: Apple clang (the CMake default); CMake finds Homebrew's Open MPI.
#
# Usage, from anywhere:
#   TRILINOS=frosch <this script> build-frosch && make -C build-frosch -j4
#
# Tests: tests/conftest.py expects the superbuild layout
# build/svMultiPhysics-build/bin/svmultiphysics and reads that directory's
# CMakeCache.txt (SV_USE_TRILINOS, SV_USE_INTERFACE2, SV_TRILINOS_HAS_FROSCH).
# For a direct build, link it temporarily and run pytest from tests/ in a
# Python environment with pytest, meshio and numpy:
#   ln -s ../build-frosch build/svMultiPhysics-build
#   (cd tests && python -m pytest test_def_diffu.py)
#   rm build/svMultiPhysics-build

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR=${1:?usage: $0 <build directory>}
TRILINOS=${TRILINOS:-frosch}
VTK_DIR=${VTK_DIR:-$HOME/opt/vtk/lib/cmake/vtk-9.7}
INTERFACE2_DIR=${INTERFACE2_DIR:-$HOME/opt/interface2}

ARGS=(-DCMAKE_BUILD_TYPE=Release -DSV_USE_MPI=ON -DSV_USE_NOTIMER=ON -DVTK_DIR=$VTK_DIR)
PREFIX_PATH=""

if [ "${INTERFACE2:-ON}" = ON ]; then
  ARGS+=(-DSV_USE_INTERFACE2=ON -DInterface2_DIR=$INTERFACE2_DIR/lib/cmake/Interface2)
  PREFIX_PATH=$INTERFACE2_DIR
fi

case $TRILINOS in
  frosch) TRILINOS_PREFIX=$HOME/opt/trilinos-16.1.0-int-frosch ;;
  int)    TRILINOS_PREFIX=$HOME/opt/trilinos-16.1.0-int ;;
  OFF)    TRILINOS_PREFIX="" ;;
  *)      TRILINOS_PREFIX=$TRILINOS ;;
esac
if [ -n "$TRILINOS_PREFIX" ]; then
  ARGS+=(-DSV_USE_TRILINOS=ON -DTrilinos_DIR=$TRILINOS_PREFIX/lib/cmake/Trilinos)
  PREFIX_PATH="${PREFIX_PATH:+$PREFIX_PATH;}$TRILINOS_PREFIX"
fi

cmake -S "$REPO/Code" -B "$BUILD_DIR" -G "Unix Makefiles" "${ARGS[@]}" "-DCMAKE_PREFIX_PATH=$PREFIX_PATH"
