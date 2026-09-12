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
  from the left; FEDDLib's FROSch uses a two-block coarse space
  (displacement with rotations, concentration), svMultiPhysics's
  `trilinos-frosch` a single block with translations only.
