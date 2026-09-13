#!/bin/bash -l
# Trilinos 16.1.0 for svMultiPhysics on macOS / Apple silicon (M4), as used
# for the local installs of 2026-09:
#   FROSCH=ON (default)  ~/opt/trilinos-16.1.0-int-frosch, with FROSch for the
#                        trilinos-frosch preconditioner
#   FROSCH=OFF           ~/opt/trilinos-16.1.0-int
#
# Notes:
#   - Tpetra global ordinal = int: svMultiPhysics's Trilinos interface
#     (trilinos_impl.h) uses GO = int, and Tpetra supports only one global
#     ordinal per build, so this cannot share an install with a long long
#     build;
#   - only the packages svMultiPhysics uses (Tpetra, Belos, Ifpack2, MueLu,
#     Amesos2/KLU2, NOX) and their dependencies;
#   - FROSch: enable the subpackage ShyLU_DDFROSch (with Xpetra), not its
#     parent ShyLU_DD, which pulls in ShyLU_DDCore and needs Epetra (off here).
# Toolchain and third-party libraries (install-dependencies-via-spack.sh):
# Homebrew Open MPI, Apple Accelerate via veclibfort, Open MPI-built spack
# HDF5/ParMETIS.
#
# On a cluster with MKL: trilinos_impl.h avoids Amesos2's umbrella header
# Amesos2.hpp because its PardisoMKL part does not compile with MKL_ILP64, but
# FROSch includes it; check this first when enabling FROSch there. The Linux
# reference configuration is Docker/ubuntu/dockerfile.
#
# Usage: run from an empty build directory, then `ninja install`.
TYPE=RELEASE
FROSCH=${FROSCH:-ON}

BASE_DIR=${TRILINOS_SOURCE_DIR:-$HOME/dev/Trilinos/Trilinos-16.1.0} # trilinos-release-16-1-0
if [ "$FROSCH" = ON ]; then
  INSTALL_DIR=${TRILINOS_INSTALL_DIR:-$HOME/opt/trilinos-16.1.0-int-frosch}
  FROSCH_ARGS=(-D Trilinos_ENABLE_ShyLU_DDFROSch:BOOL=ON
               -D Trilinos_ENABLE_Xpetra:BOOL=ON)
else
  INSTALL_DIR=${TRILINOS_INSTALL_DIR:-$HOME/opt/trilinos-16.1.0-int}
  FROSCH_ARGS=()
fi

export PATH=$HOME/dev/spack/bin:$PATH

MPI_BIN_DIR=$(dirname $(which mpiexec))

HDF5=$(spack location -i hdf5+mpi ^openmpi)
BOOST=$(spack location -i boost)
PARMETIS=$(spack location -i parmetis ^openmpi)
METIS=$(spack location -i metis)
VECLIBFORT=$(spack location -i veclibfort)

rm -rf CMake*

cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE:STRING=$TYPE \
    -D CMAKE_INSTALL_PREFIX:STRING=$INSTALL_DIR \
    -D CMAKE_C_FLAGS:STRING="-O3 -Wno-format-security -DNDEBUG -Wno-deprecated" \
    -D CMAKE_CXX_FLAGS:STRING="-O3 -Wno-invalid-specialization -Wno-format-security -DNDEBUG -Wno-deprecated" \
    -D CMAKE_Fortran_FLAGS:STRING="-O2" \
    -D MPI_BIN_DIR=$MPI_BIN_DIR \
    -D CMAKE_C_COMPILER=mpicc \
    -D CMAKE_CXX_COMPILER=mpicxx \
    -D CMAKE_Fortran_COMPILER=mpifort \
    -D CMAKE_CXX_STANDARD:STRING=17 \
    -D Trilinos_ENABLE_Fortran:BOOL=ON \
    -D BUILD_SHARED_LIBS:BOOL=OFF \
    -D Trilinos_ENABLE_Tpetra:BOOL=ON \
    -D Trilinos_ENABLE_Belos:BOOL=ON \
    -D Trilinos_ENABLE_Ifpack2:BOOL=ON \
    -D Trilinos_ENABLE_MueLu:BOOL=ON \
    -D Trilinos_ENABLE_Amesos2:BOOL=ON \
    -D Trilinos_ENABLE_NOX:BOOL=ON \
    "${FROSCH_ARGS[@]}" \
    -D Trilinos_ENABLE_Epetra:BOOL=OFF \
    -D Trilinos_ENABLE_OpenMP:BOOL=OFF \
    -D Tpetra_INST_INT_INT:BOOL=ON \
    -D Tpetra_INST_INT_LONG_LONG:BOOL=OFF \
    -D TPL_ENABLE_MPI:BOOL=ON \
    -D TPL_ENABLE_DLlib:BOOL=OFF \
    -D TPL_ENABLE_Pthread:BOOL=OFF \
    -D TPL_ENABLE_Matio:BOOL=OFF \
    -D TPL_ENABLE_BLAS:BOOL=ON \
    -D TPL_ENABLE_LAPACK:BOOL=ON \
    -D TPL_BLAS_LIBRARIES:STRING="$VECLIBFORT/lib/libvecLibFort.dylib" \
    -D TPL_LAPACK_LIBRARIES:STRING="$VECLIBFORT/lib/libvecLibFort.dylib" \
    -D TPL_ENABLE_METIS:BOOL=ON \
    -D METIS_LIBRARY_DIRS:PATH=$METIS/lib \
    -D METIS_INCLUDE_DIRS:PATH=$METIS/include \
    -D TPL_ENABLE_ParMETIS:BOOL=ON \
    -D ParMETIS_LIBRARY_DIRS:PATH="$PARMETIS/lib;$METIS/lib" \
    -D ParMETIS_INCLUDE_DIRS:PATH="$PARMETIS/include;$METIS/include" \
    -D TPL_ENABLE_Boost:BOOL=ON \
    -D Boost_INCLUDE_DIRS:PATH=$BOOST/include \
    -D TPL_ENABLE_HDF5:BOOL=ON \
    -D HDF5_LIBRARY_DIRS:PATH=$HDF5/lib \
    -D HDF5_INCLUDE_DIRS:PATH=$HDF5/include \
    -D Trilinos_ENABLE_TESTS:BOOL=OFF \
    -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
$BASE_DIR
