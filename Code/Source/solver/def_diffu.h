// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEF_DIFFU_H
#define DEF_DIFFU_H

#include "ComMod.h"
#include "SolutionStates.h"

namespace def_diffu {

/// @brief Assemble the coupled deformation-diffusion equation
/// (EquationType::phys_def_diffu): a monolithic 4-dof/node (3 displacement +
/// 1 concentration) equation whose element residual and tangent are
/// computed entirely by the Interface2/AceGen CCB constrained-mixture
/// active growth-and-remodeling element (see ace_gen_cmm_smc_element.h).
void construct_def_diffu(ComMod& com_mod, CepMod& cep_mod, const mshType& lM, const SolutionStates& solutions);

/// @brief Commit the trial per-element history produced by the just-converged
/// time step (ComMod::ccbActiveCmmGandrHistoryUpdated) into the persistent,
/// converged history (ComMod::ccbActiveCmmGandrHistory) that the next time
/// step's construct_def_diffu() calls will read. Call this once per
/// successfully converged time step, alongside svMultiPhysics's own
/// Ao=An/Yo=Yn/Do=Dn commit (see main.cpp's iterate_solution()).
void commit_history(ComMod& com_mod);

/// @brief Zero the velocity/acceleration fields (solutions.old, i.e. what
/// the NEXT time step's predictor reads as Yo/Ao) for every dof this
/// equation owns, across every equation instance of phys_def_diffu.
///
/// def_diffu.cpp assembles this equation quasi-statically (see its header
/// comment) and never reads Yn/An/Yo/Ao for anything physical -- Interface2
/// always sees accelerations=0 and a plain backward-Euler concentration
/// rate computed directly from Dn/Do. But Integrator::predictor() still
/// unconditionally forms each new step's initial Dn guess as
/// Do+Yn*dt+An*coef (via the generic Newmark predictor formula, since
/// phys_def_diffu has no predictor-side special case), and Yn/An are left
/// however Integrator::corrector()'s phys_def_diffu branch bookkeeping last
/// updated them (a byproduct of that branch, not meaningful state -- see
/// its comment in Integrator.cpp). Zeroing Yo/Ao after every converged step
/// keeps that predictor formula reducing to a clean quasi-static restart,
/// Dn=Do, rather than extrapolating from Newton-iteration bookkeeping
/// noise. Call this once per successfully converged time step, alongside
/// commit_history().
///
/// [NOTE] If/when this element is used with any of the active/growth/
/// remodeling Bool domain-data flags on (genuine time-dependent kinetics),
/// this reset is still fine to keep -- it only affects the Newmark Y/A
/// bookkeeping this equation doesn't otherwise use, not the model's own
/// internal time integration (which lives entirely in Interface2's history
/// state and domain-data time constants).
void reset_dynamics(ComMod& com_mod, SolutionStates& solutions);

};

#endif
