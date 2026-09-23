# artery_dan_cmm_drug

The long run of `../artery_dan_smc` (same mesh, loads, switch schedule and
time stepping) with the constrained-mixture element
(`<CCBActiveCMMGandR>`, Interface2's
`DeformationDiffusionConstrainedMixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10`)
in all seven tissue regions.

## Parameters (assumptions)

The regions' parameter sets are those of the smooth-muscle element; the
constrained-mixture element has 71 parameters, of which

- the 47 it shares by name with the smooth-muscle element (fA, Alpha1-5,
  Kappa, MuA, Beta1-2, D0, the calcium/phosphorylation rates, kEtaPlus,
  mEtaPlus, ActiveStartTime, Density, ...) take each region's values of
  `../artery_dan_smc` (e.g. kEtaPlus 0.001, not the 0.6 of the
  hollow_cylinder set);
- the 24 constrained-mixture parameters take the values of the
  `../hollow_cylinder` set in every region: ElastinFrac, CollagenFrac,
  SMCFrac (0.370370370, 0.370370370, 0.259259259), TElastin, TCollagen, TSMC,
  TC (1000), KElastin, KCollagen, KSMC (1), IHomo (50), QMax (1), QMin (0.2),
  AlphaMech, AlphaBio (1), PCirc, PAx, PRad (1), GNInit (1.008), NCOpt (0.6);
- the smooth-muscle-only parameters (ThetaPlus/Minus1-3, KTheta, MTheta,
  growthDirectionBool) have no counterpart and are not used.

Switches (`<Time_segments>`), in regions 15, 16, 17 and 21:

- ReorientationBool in [1, 20), [4020, 4040) and [4640, 4660);
- GroundGrowthBool and SMCGrowthBool in [4340.01, 4640), the growth
  orientation initialized when they switch on, in the first step of the
  growth segment. Switched on at the segment's start instead, i.e. in the
  last step before it, the element's local growth iteration diverged
  ("Growth: divergence ... DeltaT= 0.2" in ErrorLog.txt) and the run stopped;
- in the media (16, 17) ActiveBool in [20, 4020), [4040, 4340) and
  [4660, 5300), the active stretches initialized at t = 20;
- CollRemodelingBool off.

## Other settings

- Loads as in `../artery_dan_smc` (pressure ramped to 85 mmHg over the first
  second, drug concentration 2 on the walls from t = 4660), no rate
  acceleration, and its time stepping except for the pressure ramp, which
  takes steps of 0.05 instead of 0.2 (20 steps; about 1250 steps to
  t = 5300). The earlier schedule needed 0.005 here: with that parameter set
  the first quasi-static load step of 0.02 diverged (Newton overshoots by
  +38 dB and does not recover), while 0.005 and 0.0025 converged (7, then
  4-5 Newton iterations per step). With the corrected elements the ten times
  larger step is to be tried, and adaptive time stepping halves it if it
  fails.
- Adaptive time stepping (`<Adaptive_time_stepping> true`): the step of every
  segment is the largest step that segment may take, and a time step that
  fails -- the element cannot compute its state, the linear solver breaks
  down, or the Newton iteration reaches `<Max_iterations>` without meeting its
  tolerance -- is repeated from
  the state it started from with half the step, down to
  `<Minimum_time_step_size> 1e-4`, below which the run stops with an error;
  after 5 time steps in a row that converge the step is doubled again, up to
  the segment's size. The two places this case works around by hand, the
  first load step (a smaller pressure ramp step) and the switch-on of growth
  (4340.01 rather than the growth segment's start), are both failures of
  that kind: with
  adaptive time stepping they would be repeated with a smaller step instead.
  The workarounds are kept, so that the run takes the steps the table gives
  unless something else fails.
- Time integration: `<Spectral_radius_of_infinite_time_step> 0.0`, i.e.
  alpha_f = 1: the element and the loads are evaluated at t_{n+1}, as for
  `../artery_dan_smc`.
- Structure: quasi-static, `<Include_inertia> false </Include_inertia>`: the
  element's dynamic residual and mass matrix are left out; the parameters,
  Density included, are unchanged. With the inertia of Density 1 (kg/mm^3) the wall
  oscillated with a period of about 1 s after the pressure ramp, and the
  runs diverged near t = 2 whatever the time step.
- `<Predictor> same_displacement </Predictor>`.
- Linear solver: GMRES with `trilinos-frosch-block` and its default FROSch
  settings (coarse basis recomputed for every matrix: the time step varies
  25-fold).
- Output: `result.xdmf`/`result.h5`, every step: displacement,
  concentration, MisesStress, SCirc, SAxial, SRadial, W, PhiElastin,
  PhiCollagen, PhiSMC, Stretch1, Stretch2, nC1, nC2, nD1, nD2, DetF, DetFe and
  DetFg.
- Restart files (svMultiPhysics's, with the element history) are written every
  100 steps as `stFile_<step>.bin` (`stFile_last.bin` is the latest). To
  continue after a stop, run the same case with `<Continue_previous_simulation>
  true </Continue_previous_simulation>` on the same number of processes; it
  starts from `stFile_last.bin` (copy an earlier `stFile_<step>.bin` there to
  continue from that step) and adds to the XDMF/HDF5 result files.
- Run, e.g.

```
sbatch --partition=cpu --time=7-00:00:00 --nodes=2 --ntasks=96 --exclusive \
    BuildScripts/Elysium-RUB/run-case-job.sh artery_dan_cmm_drug trilinos-frosch-block
```
