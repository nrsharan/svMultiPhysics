# artery_dan

The artery of FEDDLib's `examples/arteries/paper_amlodipine/.../artery_dan`
(`meshes/SPP2311/Artery_dan_SCI.mesh`) with the constrained-mixture (CMM)
element, set up as the svMultiPhysics side of FEDDLib's
`examples/arteries/artery_dan_cmm`: both solve the same problem, so that the
two codes (and their FROSch preconditioners) can be compared one to one.

- Mesh: 6045 vertices, 30487 tetrahedra in seven tissue regions (domains
  15-21, the FEDDLib volume flags), quadratic (TET10) with straight edges as
  FEDDLib builds its P2 mesh: 44300 nodes.
- Boundary conditions: bottom and top fixed axially; the FEDDLib vertices
  held in x/z (flag 13) and y/z (flag 14) become the bottom triangles that
  contain them (`pin_xz`, `pin_yz`); pressure on the inner wall ramped linearly
  to 85 mmHg (11.33237 kPa) at t = 1 (`load.dat`); zero concentration on the
  inner and outer walls.
- Material: every region uses the hollow_cylinder CMM parameter set
  (PLACEHOLDER: region-specific CMM parameters are still to be defined).
- Time stepping: 50 steps of 0.02 (the load ramp).
- Linear solver: GMRES without restart, tolerance 1e-5, preconditioner
  `trilinos-frosch` (as FEDDLib's settings).

## Generating the mesh and FEDDLib's Dirichlet node sets

```
python3 ../tools/feddlib_mesh_to_svmp.py <FEDDLib>/meshes/SPP2311/Artery_dan_SCI.mesh mesh \
    --face 2:bottom --face 3:top --face 5:inner --face 4:outer \
    --pin 13:2:pin_xz --pin 14:2:pin_yz \
    --dirichlet-override <FEDDLib>/feddlib/problems/examples/arteries/artery_dan_cmm/Artery_dan_SCI_svmp_flags.txt \
    --override-flag pin_xz+outer=23 --override-flag pin_xz=13 \
    --override-flag pin_yz+outer=24 --override-flag pin_yz=14 \
    --override-flag bottom+inner=7 --override-flag bottom+outer=6 \
    --override-flag top+inner=8 --override-flag top+outer=9 \
    --override-flag bottom=2 --override-flag top=3 --override-flag inner=5 --override-flag outer=4
```

The override file gives FEDDLib a Dirichlet flag for exactly the nodes that
svMultiPhysics constrains (FEDDLib's `artery_dan_cmm/main.cpp` lists what each
flag fixes). Flags 23/24 mark the pin-face nodes on the outer wall, which also
carry the wall's concentration condition.

## Differences that remain

- Time integration: FEDDLib uses Newmark (beta = 1/4, gamma = 1/2) for the
  displacement and backward Euler for the concentration; svMultiPhysics uses
  generalized-alpha (spectral radius 0.5) for both. The concentration mass
  term of the tangent is Mc/dt in FEDDLib and gamma/(beta dt) Mc in
  svMultiPhysics.
- Newton convergence test: FEDDLib (NOX) stops on the update norm
  (1e-7), svMultiPhysics on the residual (1e-5, relative).
- Linear solver: FEDDLib preconditions GMRES from the right, svMultiPhysics
  from the left (both stop on the true residual); FEDDLib's FROSch uses a
  two-block coarse space (displacement with rotations, concentration), like
  svMultiPhysics's `trilinos-frosch-block`, while `trilinos-frosch` is a single
  block with translations only. FEDDLib keeps the coarse basis of the first
  matrix for the whole run ("Reuse: Coarse Basis") and drops basis entries
  below 1e-5 ("Phi: Dropping Threshold"); svMultiPhysics recomputes the basis
  for every matrix and drops below 1e-8 (FROSch's default).

## Results (Elysium, cpu nodes)

All runs converge in all 50 steps, with 3 Newton iterations per step in
svMultiPhysics and 3.8 in FEDDLib. GMRES iterations are the mean per linear
solve. Wall times of the same run vary by up to about 10% between nodes.

| Run | 16 ranks: wall, GMRES its | 32 ranks: wall, GMRES its |
|---|---|---|
| svMultiPhysics `trilinos-frosch`, FROSch set up for every solve | 3972 s, 25.1 | 1576 s, 33.0 |
| svMultiPhysics `trilinos-frosch`, setup kept (current) | 4063 s, 25.1 | 1522 s, 33.0 |
| svMultiPhysics `trilinos-frosch-block`, set up for every solve | 4517 s, 24.1 | 1685 s, 30.3 |
| svMultiPhysics `trilinos-frosch-block`, setup kept (current) | 4123 s, 24.1 | 1368 s, 30.3 |
| svMultiPhysics `trilinos-frosch`, `<Diagonal_scaling> false` | 3739 s, 24.8 | 1611 s, 32.1 |
| svMultiPhysics `trilinos-frosch`, coarse basis kept (`frosch_recycle_coarse_basis.xml`) | 3588 s, 27.1 | 1438 s, 35.8 |
| FEDDLib `artery_dan_cmm` (coarse basis kept) | 3233 s, 203.7 | 1735 s, 266.0 |
| FEDDLib, coarse basis recomputed | 3464 s, 156.5 | 1545 s, 192.2 |

Keeping the FROSch setup gives the same iterations as setting it up anew.
Where the time goes (timer summary printed at the end of the run), 32 ranks:

- svMultiPhysics `trilinos-frosch` (1603 s): FROSch `compute` 1457 s for 150
  matrices (overlapping subdomains 776 s, coarse space 681 s), GMRES 104 s,
  graph and matrix creation 9 s, assembly and the rest about 30 s; FROSch
  `initialize` 0.4 s, once.
- FEDDLib, coarse basis recomputed (1545 s): FROSch `compute` 724 s for 191
  matrices, GMRES 577 s.

On 16 ranks: svMultiPhysics (3905 s) FROSch `compute` 3695 s (overlapping
subdomains 2150 s, coarse space 1545 s), GMRES 135 s, graph and matrix 13 s;
FEDDLib with the coarse basis kept (3233 s) FROSch `compute` 1840 s for 191
matrices, GMRES 978 s.

So the number of GMRES iterations hardly matters for svMultiPhysics; the
FROSch `compute` for every Newton iteration does. The overlapping subdomains
have about the same size in both codes (32 ranks, after one layer of overlap:
12406 degrees of freedom on average, 15824 at most, in svMultiPhysics; 12151
and 15036 in FEDDLib), yet a FROSch `compute` takes 9.7 s in svMultiPhysics and
3.8 s in FEDDLib, and svMultiPhysics needs 6 to 8 times fewer GMRES
iterations. Neither the diagonal scaling, the stopping test, the kept coarse
basis nor the Trilinos build (both optimized) explains this; the systems
themselves do differ (below).

### The two systems

The first Newton system of both codes (16 ranks; written with
`SVMP_TRILINOS_WRITE_SYSTEM` and `FEDD_WRITE_SYSTEM`, compared with
`../tools/compare_systems.py`, nodes matched by their coordinates):

- The concentration is zero in both runs (zero initial and wall values), so the
  displacement-concentration coupling of the tangent is zero. FEDDLib stores no
  coupling blocks; svMultiPhysics stores all 4x4 dof couplings of every node
  pair (19.3 million entries against FEDDLib's 12.1 million), the coupling ones
  as zeros. FROSch factorizes the stored pattern: without them
  (`SVMP_FROSCH_DROP_ZEROS`), FROSch `compute` takes about half the time on
  hollow_cylinder_short, with the same iterations.
- Displacement block: same pattern; on the rows without Dirichlet conditions
  svMultiPhysics's values are 1.185e-4 times FEDDLib's (relative difference
  1e-3 after this factor, from the time integration).
- Concentration block: same pattern, values differ by 30% beyond a common
  factor (mass term Mc/dt in FEDDLib, gamma/(beta dt) Mc in svMultiPhysics).
- Dirichlet conditions: FEDDLib replaces the Dirichlet rows by rows of the
  identity but keeps the Dirichlet columns (its displacement block is
  nonsymmetric, |A - A^T| / |A| = 0.24, and its Dirichlet diagonal entries are
  1 against a median of 426); svMultiPhysics removes the rows and columns and
  scales the system symmetrically. FROSch finds FEDDLib's Dirichlet rows itself
  (rows with a single nonzero entry).
