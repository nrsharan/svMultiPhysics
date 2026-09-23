// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEF_DIFFU_H
#define DEF_DIFFU_H

#include "CmMod.h"
#include "ComMod.h"
#include "SolutionStates.h"

#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace def_diffu {

/// @brief The state of the deformation-diffusion elements at the start of a
/// time step: the element history and the domain-data flags with
/// <Time_segments>. With adaptive time stepping a time step that fails is
/// repeated from this state with a smaller time step size (see main.cpp's
/// iterate_solution()), which advance_time_step() and the element would
/// otherwise not see: it switches the flags on, runs their one-time
/// initialization and writes the result into the converged history.
struct StepState {
  /// @brief The converged element history of every mesh
  /// (ComMod::ccbActiveCmmGandrHistory).
  std::map<std::string, std::vector<double>> history;

  /// @brief The trial element history of every mesh
  /// (ComMod::ccbActiveCmmGandrHistoryUpdated).
  std::map<std::string, std::vector<double>> historyUpdated;

  /// @brief The value of every domain-data flag with <Time_segments>, in the
  /// order the equations, domains and flags are visited.
  std::vector<double> flagValues;

  /// @brief Whether the one-time initialization of every one of those flags
  /// has run, in the same order.
  std::vector<char> flagInitialized;
};

/// @brief The state of the deformation-diffusion elements at the current
/// time, to repeat a time step from (see StepState).
StepState save_state(const ComMod& com_mod);

/// @brief Put the state saved by save_state() back.
void restore_state(ComMod& com_mod, const StepState& state);

/// @brief Assemble the coupled deformation-diffusion equation
/// (EquationType::phys_def_diffu): a monolithic 4-dof/node (3 displacement +
/// 1 concentration) equation whose element residual and tangent are
/// computed entirely by one of the Interface2/AceGen CCB active
/// growth-and-remodeling elements (see ace_gen_cmm_smc_element.h).
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

/// @brief Output fields of the element post-processing quantities of
/// equation eq (ace_gen_cmm_smc::post_fields() of its first
/// deformation-diffusion domain; all its domains use the same element), or
/// none if it has no such domain.
std::vector<ace_gen_cmm_smc::PostField> post_fields(const eqType& eq);

/// @brief Nodal values of the element post-processing quantities of equation
/// iEq on mesh lM at the current state, for the VTK output: 'values' is
/// resized to (number of quantities, lM.nNo). The elements' weighted
/// contributions (ace_gen_cmm_smc::post_process()) are summed at every node
/// over the elements sharing it, on all processors, and divided by the summed
/// weights, which are kept in row 0. Nodes without weight -- those of other
/// equations' elements, and all nodes before the first time step, when there
/// is no element state yet -- get 0. Must be called on every processor.
void nodal_post_data(const ComMod& com_mod, const mshType& lM, const SolutionStates& solutions, const int iEq,
                     Array<double>& values);

/// @brief Number of values of the element history this process adds to its
/// restart record: the history of every local element of every mesh with
/// deformation-diffusion elements (ComMod::ccbActiveCmmGandrHistory).
std::size_t restart_history_size(const ComMod& com_mod);

/// @brief Write this process's element history to its restart record, after
/// svMultiPhysics's own data: the history of the time step just converged
/// (ComMod::ccbActiveCmmGandrHistoryUpdated, committed into
/// ccbActiveCmmGandrHistory only when the next time step starts), mesh by mesh
/// in the local element order. A restart must use the same number of
/// processes, as for the rest of the record.
void write_restart_history(ComMod& com_mod, std::ostream& restart_file);

/// @brief Read the element history written by write_restart_history() into
/// both ComMod::ccbActiveCmmGandrHistory and ccbActiveCmmGandrHistoryUpdated.
void read_restart_history(ComMod& com_mod, std::ifstream& restart_file);

};

#endif
