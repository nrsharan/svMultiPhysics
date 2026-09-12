# Building svMultiPhysics on macOS / Apple silicon

The scripts used for the local svMultiPhysics stack on an Apple M4 (macOS 26,
2026-09), in build order. Each configure script states its options and default
paths at the top.

1. `install-dependencies-via-spack.sh`: Homebrew toolchain (Apple clang,
   gfortran, Open MPI 5.0.9) and the spack-built libraries Trilinos uses
   (veclibfort, METIS, ParMETIS, HDF5, Boost).
2. `configure-VTK.sh`: VTK 9.7.0, static, reading/writing only, to `~/opt/vtk`.
3. Interface2 (for the `deformation-diffusion` equation): its
   `config_scripts/do-config-interface2-macos-accelerate.sh`, installed to
   `~/opt/interface2`.
4. `configure-Trilinos.sh`: Trilinos 16.1.0 with global ordinal `int`, with
   FROSch (`FROSCH=ON`, default) or without.
5. `configure-svMultiPhysics.sh`: svMultiPhysics against these
   (`TRILINOS=frosch|int|OFF`), including how to run the tests.

## Notes for other machines and clusters

- svMultiPhysics's Trilinos interface uses global ordinal `int`
  (`Tpetra_INST_INT_INT=ON`); a Trilinos built for FEDDLib (`long long`) cannot
  be used, since Tpetra supports one global ordinal per build.
- The `trilinos-frosch` preconditioner needs `-DTrilinos_ENABLE_ShyLU_DDFROSch=ON`
  (the subpackage, not ShyLU_DD, which requires Epetra). svMultiPhysics reports at
  configure time whether Trilinos provides it. The Linux reference configuration
  is `Docker/ubuntu/dockerfile`.
- With Intel MKL (`MKL_ILP64`), Amesos2's umbrella header `Amesos2.hpp` does not
  compile because of its PardisoMKL part. `trilinos_impl.h` avoids it, but FROSch
  includes it, so check this first when building with FROSch on a cluster.
- On Apple silicon, BLAS/LAPACK must come from Accelerate and all MPI libraries
  from the same MPI (see `install-dependencies-via-spack.sh`).
- Memory: `trilinos-amesos2` (a full direct factorization) needs 6.7 GB on one
  process for `tests/cases/def_diffu/plaque_short`; `trilinos-frosch` solves on
  every process only its subdomain directly.
