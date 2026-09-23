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
  concentration 0 on the inner and outer walls, 2 from t = 4660 (`conc.dat`).
- Switches (`<Time_segments>`), in regions 15, 16, 17 and 21 (regions 18-20
  stay passive):
  - reorientation in [1, 20), [4020, 4040) and [4640, 4660);
  - growth in [4340, 4640), the growth orientation initialized at t = 4340;
  - in the media (16, 17) the active response in [20, 4020), [4040, 4340) and
    [4660, 5300), the active stretches initialized at t = 20.
- No rate acceleration. The earlier schedule multiplied the rate parameters
  by 20 (and divided Gamma2, Gamma5) until t = 220 to reach the active steady
  state in 200 s; with the corrected elements (Interface2 464d23f) the active
  response takes steps of a few hundred seconds, so it runs the 4000 s out at
  the true rates instead -- 200 s at 20 times the rate is the 4000 s here.
- Time stepping: 8 segments, about 12080 steps to t = 5300:

  | Time | Step | Steps |
  |---|---|---|
  | [0, 1) | 0.02 | 50 (pressure ramp) |
  | [1, 20) | 5 | 4 |
  | [20, 4020) | 500 | 8 (active response, to a steady state) |
  | [4020, 4040) | 5 | 4 |
  | [4040, 4340) | 100 | 3 |
  | [4340, 4640) | 0.025 | 12000 (growth) |
  | [4640, 4660) | 5 | 4 |
  | [4660, 5300) | 100 | 7 (drug) |

- Adaptive time stepping (`<Adaptive_time_stepping> true`): every step above
  is the largest step its segment may take, so the table gives the steps and
  the step count of a run in which none of them fails. A time step that fails
  -- the element cannot compute its state, the linear solver breaks down, or
  the Newton iteration reaches `<Max_iterations>` without meeting its
  tolerance -- is repeated from the
  state it started from with half the step, down to
  `<Minimum_time_step_size> 1e-4`, below which the run stops with an error;
  after 5 time steps in a row that converge the step is doubled again, up to
  the segment's size. Every repeated step is reported as `[adaptive] t = ...`
  and their number at the end of the run. A run of the earlier schedule
  stopped in the growth phase at t = 718.725 ("Growth: divergence in elem=
  281 gp= 4 ... DeltaT= 0.025") and had to be repeated from a restart file
  with the growth step halved throughout; only the steps that fail are
  halved now, which is why the growth segment may take 0.25.

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
  step from 0.2 to 5 after the pressure ramp (t = 1) started Newton far
  from the solution and diverged.
- Time integration: `<Spectral_radius_of_infinite_time_step> 0.0`, i.e.
  alpha_f = 1: the element and the loads are evaluated at t_{n+1}.
  svMultiPhysics evaluates the element at the intermediate state
  u_{n+alpha_f} but the pressure at t_{n+1}; without inertia, alpha_f < 1
  would enforce equilibrium at the intermediate state.
- Output: one XDMF/HDF5 pair (`result.xdmf`, `result.h5`) with the results
  of every step -- the active phase is only eight of them: displacement, concentration and the element quantities
  MisesStress, SCirc, SAxial, SRadial, W, Growth, Stretch1, Stretch2, nC1,
  nC2, nD1, nD2, DetF, DetFe and DetFg.
- Restart files (svMultiPhysics's, with the element history) are written every
  100 steps as `stFile_<step>.bin` (`stFile_last.bin` is the latest). To
  continue after a stop, run the same case with `<Continue_previous_simulation>
  true </Continue_previous_simulation>` on the same number of processes; it
  starts from `stFile_last.bin` (copy an earlier `stFile_<step>.bin` there to
  continue from that step) and adds to the XDMF/HDF5 result files.

Run on Elysium, e.g. on two nodes:

```
sbatch --partition=cpu --time=7-00:00:00 --nodes=2 --ntasks=96 \
    BuildScripts/Elysium-RUB/run-case-job.sh artery_dan_smc trilinos-frosch-block
```
