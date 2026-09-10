// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEF_DIFFU_H
#define DEF_DIFFU_H

#include "CmMod.h"
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

/// @brief Start-of-time-step update, called once com_mod.time holds the new
/// time t_{n+1}: set every CCB domain-data flag that has <Time_segments> to
/// 1 inside and 0 outside its intervals, and run the one-time
/// initialization (Time_segments 'initialization' attribute) of a flag's
/// first switch-on from the converged state of the previous time step.
void advance_time_step(ComMod& com_mod, const CmMod& cm_mod, const SolutionStates& solutions);

};

#endif
