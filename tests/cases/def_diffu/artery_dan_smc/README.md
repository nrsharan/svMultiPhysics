# artery_dan_smc

A long run of the artery of `../artery_dan` (same mesh, Dirichlet conditions
and pressure ramp) with the smooth-muscle element without the constrained
mixture (`<CCBActiveGandR>`, Interface2's
`DeformationDiffusionSmoothMuscleActiveGrowthReorientationTetrahedra3D10`)
in all seven tissue regions, and a drug concentration applied on the walls.

- Mesh: `../artery_dan/mesh` (TET10, 44300 nodes), regions 15-21: 15
  adventitia, 16 media, 17 degenerated media, 18 lipid, 19 and 20
  calcifications, 21 fibrous cap. Each region has its own parameter set
  (mainly Kappa, MuA, Alpha1-5, D0 and ActiveStartTime differ).
- Loads: pressure on the inner wall ramped linearly to 85 mmHg (11.33237
  kPa) over the first second (`../artery_dan/load.dat`, follower pressure);
  concentration 0 on the inner and outer walls, 2 from t = 860 (`conc.dat`).
- Switches (`<Time_segments>`), in regions 15, 16, 17 and 21 (regions 18-20
  stay passive):
  - reorientation in [1, 20), [220, 240) and [840, 860);
  - growth in [540, 840), the growth orientation initialized at t = 540;
  - in the media (16, 17) the active response in [20, 220), [240, 540) and
    [860, 1500), the active stretches initialized at t = 20.
- Rate acceleration (`<Rate_acceleration>`): until t = 220 the rate
  parameters LambdaBarCDotMax, LambdaBarCDotMin, Eta, K3, K4, K7, Beta1,
  Gamma6, KDotMin, KDotMax, LambdaBarDotPMin and LambdaBarDotPMax are
  multiplied by 20 and Gamma2 and Gamma5 divided by 20.
- Time stepping: 11 segments, 17958 steps to t = 1500:

  | Time | Step | Steps |
  |---|---|---|
  | [0, 1) | 0.02 | 50 (pressure ramp) |
  | [1, 20) | 0.5 | 38 |
  | [20, 50) | 0.1 | 300 |
  | [50, 80) | 0.15 | 200 |
  | [80, 110) | 0.2 | 150 |
  | [110, 220) | 0.25 | 440 |
  | [220, 240) | 0.5 | 40 |
  | [240, 540) | 0.2 | 1500 |
  | [540, 840) | 0.025 | 12000 (growth) |
  | [840, 860) | 0.5 | 40 |
  | [860, 1500) | 0.2 | 3200 (drug) |

- Linear solver: GMRES with `trilinos-frosch-block` and
  `../artery_dan/frosch_block.xml`.
- Output: one XDMF/HDF5 pair (`result.xdmf`, `result.h5`) with the results
  of every 100th step: displacement, concentration and the element quantities
  MisesStress, SCirc, SAxial, SRadial, W, Growth, Stretch1, Stretch2, nC1,
  nC2, nD1, nD2, DetF, DetFe and DetFg.
- The element history is not in svMultiPhysics's restart files, so a
  continued simulation would start growth and remodeling anew; the case
  writes no restart files and must run in one job (on Elysium, the `cpu`
  partition allows 7 days).

Run on Elysium, e.g. on two nodes:

```
sbatch --partition=cpu --time=7-00:00:00 --nodes=2 --ntasks=96 \
    BuildScripts/Elysium-RUB/run-case-job.sh artery_dan_smc trilinos-frosch-block
```
