# artery_dan

An artery with the constrained-mixture (CMM) element of the
`deformation-diffusion` equation.

- Mesh (`mesh/`, from `Artery_dan_SCI.mesh`): 6045 vertices, 30487 tetrahedra
  in seven tissue regions (domains 15-21), quadratic (TET10) with straight
  edges: 44300 nodes.
- Boundary conditions: bottom and top fixed axially; two pin faces (`pin_xz`,
  `pin_yz`: the bottom triangles that contain two pinned vertices) remove the
  rigid-body modes; pressure on the inner wall ramped linearly to 85 mmHg
  (11.33237 kPa) at t = 1 (`load.dat`, follower pressure); zero concentration
  on the inner and outer walls.
- Material: every region uses the hollow_cylinder CMM parameter set
  (PLACEHOLDER: region-specific CMM parameters are still to be defined).
- Time stepping: 50 steps of 0.02 (the load ramp).
- Linear solver: GMRES without restart, tolerance 1e-5, preconditioner
  `trilinos-frosch-block` with `frosch_block.xml` (coarse basis entries below
  1e-5 dropped, the coarse basis of the first matrix kept), the fastest setup
  measured (below). To run another preconditioner, also remove the
  `<Configuration_file>` line: the file is for the two-block preconditioner
  (rotations of the displacements), which `trilinos-frosch` cannot use.

## Performance (Elysium, cpu nodes)

All runs converge in all 50 steps with 3 Newton iterations per step. GMRES
iterations are the mean per linear solve. Wall times of the same run vary by
up to about 10% between nodes.

| Run | 16 ranks: wall, GMRES its | 32 ranks: wall, GMRES its |
|---|---|---|
| `trilinos-frosch`, FROSch set up for every solve | 3972 s, 25.1 | 1576 s, 33.0 |
| `trilinos-frosch`, setup kept, symbolic factorization reused | 4063 s, 25.1 | 1522 s, 33.0 |
| `trilinos-frosch-block`, set up for every solve | 4517 s, 24.1 | 1685 s, 30.3 |
| `trilinos-frosch-block`, setup kept, symbolic factorization reused | 4123 s, 24.1 | 1368 s, 30.3 |
| `trilinos-frosch`, `<Diagonal_scaling> false` | 3739 s, 24.8 | 1611 s, 32.1 |
| `trilinos-frosch`, coarse basis of the first matrix kept | 3588 s, 27.1 | 1438 s, 35.8 |
| `trilinos-frosch`, symbolic factorization not reused (the default now) | | 711 s, 33.0 |
| `trilinos-frosch`, reused, stored zeros dropped (`SVMP_FROSCH_DROP_ZEROS`) | 1843 s, 25.1 | 718 s, 33.0 |
| `trilinos-frosch-block` with `frosch_block.xml` (this case's setup) | 1608 s, 26.8 | 702 s, 34.7 |
| `trilinos-frosch-block`, only coarse basis entries below 1e-5 dropped | | 714 s, 30.3 |
| `trilinos-frosch-block`, `frosch_block.xml` and stored zeros dropped | | 676 s, 34.7 |

Keeping the FROSch setup gives the same iterations as setting it up anew.
Where the time goes (timer summary printed at the end of the run, before the
symbolic factorization reuse was switched off):

- 32 ranks (1603 s): FROSch `compute` 1457 s for 150 matrices (overlapping
  subdomains 776 s, coarse space 681 s), GMRES 104 s, graph and matrix
  creation 9 s, assembly and the rest about 30 s; FROSch `initialize` 0.4 s,
  once.
- 16 ranks (3905 s): FROSch `compute` 3695 s (overlapping subdomains 2150 s,
  coarse space 1545 s), GMRES 135 s, graph and matrix creation 13 s.

With "Reuse: Symbolic Factorization" (FROSch's default), FROSch updates the
overlapping subdomain matrices of every new matrix entry by entry, which is
slower than extracting and factorizing them anew, the more so for the long
rows of the matrix (up to 315 entries: all 4x4 dof couplings of every node
pair are stored, the displacement-concentration ones as zeros while the
concentration is zero). Not reusing it (now the default) or dropping the
stored zeros halves the run on 32 ranks (711 s and 718 s instead of 1522 s,
same iterations); both remove the same cost. On 32 ranks the setups without
symbolic factorization reuse are all within 676-718 s, i.e. within the
variation between nodes; on 16 ranks `trilinos-frosch-block` with
`frosch_block.xml` is the fastest (1608 s against 1843 s).
