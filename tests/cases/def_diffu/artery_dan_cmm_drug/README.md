# artery_dan_cmm_drug

The long run of `../artery_dan_smc` (same mesh, loads, switch schedule, rate
acceleration and time stepping) with the constrained-mixture element
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

- ReorientationBool in [1, 20), [220, 240) and [840, 860);
- GroundGrowthBool and SMCGrowthBool in [540, 840), the growth orientation
  initialized at t = 540;
- in the media (16, 17) ActiveBool in [20, 220), [240, 540) and [860, 1500),
  the active stretches initialized at t = 20;
- CollRemodelingBool off.

## Other settings

- Loads, time stepping and rate acceleration as in `../artery_dan_smc`
  (pressure ramped to 85 mmHg over the first second, drug concentration 2 on
  the walls from t = 860, 17958 steps to t = 1500).
- Structure: quasi-static, `<Include_inertia> false </Include_inertia>`: the
  element's dynamic residual and mass matrix are left out; the parameters,
  Density included, are unchanged. With the inertia of Density 1 (kg/mm^3) the wall
  oscillated with a period of about 1 s after the pressure ramp, and the
  runs diverged near t = 2 whatever the time step.
- `<Predictor> same_displacement </Predictor>`.
- Linear solver: GMRES with `trilinos-frosch-block` and its default FROSch
  settings (coarse basis recomputed for every matrix: the time step varies
  25-fold).
- Output: `result.xdmf`/`result.h5`, every 100th step: displacement,
  concentration, MisesStress, SCirc, SAxial, SRadial, W, PhiElastin,
  PhiCollagen, PhiSMC, Stretch1, Stretch2, nC1, nC2, nD1, nD2, DetF, DetFe and
  DetFg.
- No restart files (the element history is not in them): run in one job, e.g.

```
sbatch --partition=cpu --time=7-00:00:00 --nodes=2 --ntasks=96 --exclusive \
    BuildScripts/Elysium-RUB/run-case-job.sh artery_dan_cmm_drug trilinos-frosch-block
```
