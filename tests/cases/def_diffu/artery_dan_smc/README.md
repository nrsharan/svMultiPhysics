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

- Adaptive time stepping (`<Adaptive_time_stepping> true`): every step above
  is the largest step its segment may take, so the table gives the steps and
  the step count of a run in which none of them fails. A time step that fails
  -- the element cannot compute its state, or the Newton iteration reaches
  `<Max_iterations>` without meeting its tolerance -- is repeated from the
  state it started from with half the step, down to
  `<Minimum_time_step_size> 1e-4`, below which the run stops with an error;
  after 5 time steps in a row that converge the step is doubled again, up to
  the segment's size. Every repeated step is reported as `[adaptive] t = ...`
  and their number at the end of the run. An earlier run of this case stopped
  in the growth phase at t = 718.725 ("Growth: divergence in elem= 281 gp= 4
  ... DeltaT= 0.025") and had to be repeated from a restart file with the
  growth step halved throughout; only the steps that fail are halved now.

- Linear solver: GMRES with `trilinos-frosch-block` and
  `../artery_dan/frosch_block.xml`.
- Structure: quasi-static, `<Include_inertia> false </Include_inertia>`: the
  element's dynamic residual and mass matrix are left out; the parameters,
  Density included, are unchanged. With the inertia of Density 1 (kg/mm^3) the wall
  oscillated with a period of about 1 s after the pressure ramp, and the
  runs diverged near t = 2 whatever the time step.
- Predictor: `<Predictor> same_displacement </Predictor>`: every time step
  starts from the displacement of the previous one. The default predictor
  extrapolates it with the previous velocity, which at the jump of the time
  step from 0.02 to 0.5 after the pressure ramp (t = 1) started Newton far
  from the solution and diverged.
- Time integration: `<Spectral_radius_of_infinite_time_step> 0.0`, i.e.
  alpha_f = 1: the element and the loads are evaluated at t_{n+1}.
  svMultiPhysics evaluates the element at the intermediate state
  u_{n+alpha_f} but the pressure at t_{n+1}; without inertia, alpha_f < 1
  would enforce equilibrium at the intermediate state.
- Output: one XDMF/HDF5 pair (`result.xdmf`, `result.h5`) with the results
  of every 100th step: displacement, concentration and the element quantities
  MisesStress, SCirc, SAxial, SRadial, W, Growth, Stretch1, Stretch2, nC1,
  nC2, nD1, nD2, DetF, DetFe and DetFg.
- Restart files (svMultiPhysics's, with the element history) are written every
  500 steps as `stFile_<step>.bin` (`stFile_last.bin` is the latest). To
  continue after a stop, run the same case with `<Continue_previous_simulation>
  true </Continue_previous_simulation>` on the same number of processes; it
  starts from `stFile_last.bin` (copy an earlier `stFile_<step>.bin` there to
  continue from that step) and adds to the XDMF/HDF5 result files.

Run on Elysium, e.g. on two nodes:

```
sbatch --partition=cpu --time=7-00:00:00 --nodes=2 --ntasks=96 \
    BuildScripts/Elysium-RUB/run-case-job.sh artery_dan_smc trilinos-frosch-block
```
