// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Assembly for EquationType::phys_def_diffu: a monolithic 4-dof/node (3
// displacement + 1 concentration) equation whose element residual and
// tangent are computed entirely by the Interface2/AceGen CCB
// constrained-mixture active growth-and-remodeling element. There is no
// inner Gauss-point loop: Interface2 integrates all Gauss points internally
// in one compute() call per element.
//
// Time integration uses svMultiPhysics's generalized-alpha scheme, as for
// the struct equation (see sv_struct.cpp): the element is evaluated at the
// intermediate state (displacement and concentration from Dg, acceleration
// from Ag, concentration rate from Yg), its residual includes the inertia
// term Rdyn, and the tangent is taken with respect to the acceleration
// unknown: am*Mu + af*gam*dt*Mc + af*beta*dt^2*K, with K the element's
// d(residual)/d(displacement, concentration). The displacement and
// concentration dofs are then advanced by the Newmark relations in
// Integrator::corrector(), like any other second-order equation.
//
// Domain-data flags with time segments (<Time_segments> in
// <CCBActiveCMMGandR>) are set at the start of every time step by
// advance_time_step(), which also runs the one-time initialization of a
// flag's first switch-on, as FEDDLib's initializeActiveResponse() and
// initializeGrowth() do.

#include "def_diffu.h"

#include "ace_gen_cmm_smc_element.h"
#include "all_fun.h"
#include "consts.h"
#include "LinearAlgebra.h"
#include "time_segments.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace def_diffu {

namespace {

/// Set up the per-element history of mesh lM on first use by tiling the
/// generic per-Gauss-point placeholder (ace_gen_cmm_smc::initial_history)
/// across every element. This does not need to be geometry-aware: the AceGen
/// kernel overwrites the geometry-dependent fiber/growth-orientation entries
/// (a11-a23) and the growth tensor (ag11-ag33) itself, inside its own Task 3
/// compute() on the very first call (gated on time==timeIncrement), so the
/// placeholder's values there are discarded.
void initialize_history(ComMod& com_mod, const mshType& lM, const int iEq)
{
  auto& meshHistory = com_mod.ccbActiveCmmGandrHistory[lM.name];
  auto& meshHistoryUpdated = com_mod.ccbActiveCmmGandrHistoryUpdated[lM.name];

  if (!meshHistory.empty()) {
    return;
  }

  const auto& eq = com_mod.eq[iEq];
  int historyLengthPerElement = -1;

  for (int e = 0; e < lM.nEl; e++) {
    const auto& dmn = eq.dmn[all_fun::domain(com_mod, lM, iEq, e)];

    if (dmn.phys != consts::EquationType::phys_def_diffu) {
      continue;
    }

    if (historyLengthPerElement < 0) {
      historyLengthPerElement = dmn.ccb_active_cmm_gandr_info.historyLengthPerElement;
      if (historyLengthPerElement > 0) {
        meshHistory.assign(static_cast<std::size_t>(historyLengthPerElement) * lM.nEl, 0.0);
      }
    } else if (dmn.ccb_active_cmm_gandr_info.historyLengthPerElement != historyLengthPerElement) {
      throw std::runtime_error(
          "[construct_def_diffu] Mesh '" + lM.name + "' has multiple "
          "CCBActiveCMMGandR domains with different history lengths "
          "(e.g. different Integration_code values); this is not supported.");
    }

    if (historyLengthPerElement == 0) {
      continue;
    }

    auto elemHistory = ace_gen_cmm_smc::initial_history(dmn.ccb_active_cmm_gandr_info);

    std::copy(elemHistory.begin(), elemHistory.end(),
              meshHistory.begin() + static_cast<std::size_t>(e) * historyLengthPerElement);
  }

  meshHistoryUpdated = meshHistory;
}

/// Element input for element e of mesh lM at the state (D, Y, A): D gives
/// the displacements and concentrations, A the accelerations and Y the
/// concentration rates.
ace_gen_cmm_smc::ElementInput element_input(const ComMod& com_mod, const eqType& eq, const dmnType& dmn,
                                            const mshType& lM, const int e, const Array<double>& D,
                                            const Array<double>& Y, const Array<double>& A,
                                            const std::vector<double>& meshHistory,
                                            const int historyLengthPerElement)
{
  const int nsd = com_mod.nsd;

  ace_gen_cmm_smc::ElementInput input;
  input.integrationCode = dmn.ccb_active_cmm_gandr_integration_code;
  input.subIterationTolerance = dmn.ccb_active_cmm_gandr_subiteration_tolerance;
  input.domainData = dmn.ccb_active_cmm_gandr_domain_data;
  input.timeIncrement = com_mod.dt;
  input.time = com_mod.time;
  // [NOTE] Local (per-rank, per-mesh) element index, not a globally
  // unique element ID across MPI ranks/meshes. Interface2 uses this only
  // for its own error reporting, not for anything that affects the
  // computed residual/tangent, so this is acceptable.
  input.elementID = e;

  if (historyLengthPerElement > 0) {
    auto begin = meshHistory.begin() + static_cast<std::size_t>(e) * historyLengthPerElement;
    input.history.assign(begin, begin + historyLengthPerElement);
  }

  for (int a = 0; a < lM.eNoN; a++) {
    int Ac = lM.IEN(a,e);

    for (int i = 0; i < nsd; i++) {
      input.positions(i,a) = com_mod.x(i,Ac);
      input.displacements(i,a) = D(eq.s+i, Ac);
      input.accelerations(i,a) = A(eq.s+i, Ac);
    }

    input.concentrations(a) = D(eq.s+nsd, Ac);
    input.rates(a) = Y(eq.s+nsd, Ac);
  }

  return input;
}

/// One-time initialization of the elements of domain iDmn when a flag first
/// switches on, evaluated at the converged state of the previous time step.
void initialize_flag(ComMod& com_mod, const int iEq, const int iDmn, const std::string& initialization,
                     const SolutionStates& solutions)
{
  const auto& Do = solutions.old.get_displacement();
  const auto& Yo = solutions.old.get_velocity();
  const auto& Ao = solutions.old.get_acceleration();

  const auto& eq = com_mod.eq[iEq];
  const auto& dmn = eq.dmn[iDmn];
  const int historyLengthPerElement = dmn.ccb_active_cmm_gandr_info.historyLengthPerElement;

  if (historyLengthPerElement == 0) {
    return;
  }

  for (const auto& lM : com_mod.msh) {
    initialize_history(com_mod, lM, iEq);

    auto& meshHistory = com_mod.ccbActiveCmmGandrHistory[lM.name];
    auto& meshHistoryUpdated = com_mod.ccbActiveCmmGandrHistoryUpdated[lM.name];

    if (meshHistory.empty()) {
      continue;
    }

    for (int e = 0; e < lM.nEl; e++) {
      if (all_fun::domain(com_mod, lM, iEq, e) != iDmn) {
        continue;
      }

      auto input = element_input(com_mod, eq, dmn, lM, e, Do, Yo, Ao, meshHistory, historyLengthPerElement);
      std::vector<double> history = (initialization == "active_stretches")
          ? ace_gen_cmm_smc::history_with_active_stretches(input)
          : ace_gen_cmm_smc::history_with_growth_orientation(input);

      auto offset = static_cast<std::size_t>(e) * historyLengthPerElement;
      std::copy(history.begin(), history.end(), meshHistory.begin() + offset);
      std::copy(history.begin(), history.end(), meshHistoryUpdated.begin() + offset);
    }
  }
}

/// Value of a domain-data flag, or -1 if the element has no such parameter.
double flag_value(const dmnType& dmn, const std::string& name)
{
  const auto& names = dmn.ccb_active_cmm_gandr_info.domainDataNames;
  auto it = std::find(names.begin(), names.end(), name);

  if (it == names.end()) {
    return -1.0;
  }

  return dmn.ccb_active_cmm_gandr_domain_data[it - names.begin()];
}

}

void construct_def_diffu(ComMod& com_mod, CepMod& cep_mod, const mshType& lM, const SolutionStates& solutions)
{
  using namespace consts;

  const auto& Ag = solutions.intermediate.get_acceleration();
  const auto& Yg = solutions.intermediate.get_velocity();
  const auto& Dg = solutions.intermediate.get_displacement();

  const int cEq = com_mod.cEq;
  auto& eq = com_mod.eq[cEq];
  auto& cDmn = com_mod.cDmn;
  const double dt = com_mod.dt;

  // Derivatives of the intermediate acceleration, rate and state with
  // respect to the acceleration unknown (see sv_struct.cpp).
  const double am = eq.am;
  const double afv = eq.af * eq.gam * dt;
  const double afu = eq.af * eq.beta * dt * dt;

  const int eNoN = lM.eNoN; // 10 for this Tet10 element
  const int dof = eq.dof;   // 4 (3 displacement + 1 concentration)

  Vector<int> ptr(eNoN);
  Array<double> lR(dof, eNoN);
  Array3<double> lK(dof*dof, eNoN, eNoN);

  // Per-element history storage for this mesh (see the long comment on
  // ComMod::ccbActiveCmmGandrHistory).
  initialize_history(com_mod, lM, cEq);
  auto& meshHistory = com_mod.ccbActiveCmmGandrHistory[lM.name];
  auto& meshHistoryUpdated = com_mod.ccbActiveCmmGandrHistoryUpdated[lM.name];

  int historyLengthPerElement = -1;

  for (int e = 0; e < lM.nEl; e++) {
    cDmn = all_fun::domain(com_mod, lM, cEq, e);
    auto& dmn = eq.dmn[cDmn];

    if (dmn.phys != EquationType::phys_def_diffu) {
      continue;
    }

    if (historyLengthPerElement < 0) {
      historyLengthPerElement = dmn.ccb_active_cmm_gandr_info.historyLengthPerElement;
    } else if (dmn.ccb_active_cmm_gandr_info.historyLengthPerElement != historyLengthPerElement) {
      throw std::runtime_error(
          "[construct_def_diffu] Mesh '" + lM.name + "' has multiple "
          "CCBActiveCMMGandR domains with different history lengths "
          "(e.g. different Integration_code values); this is not supported.");
    }

    auto input = element_input(com_mod, eq, dmn, lM, e, Dg, Yg, Ag, meshHistory, historyLengthPerElement);

    for (int a = 0; a < eNoN; a++) {
      ptr(a) = lM.IEN(a,e);
    }

    auto output = ace_gen_cmm_smc::compute(input);

    lR = 0.0;
    lK = 0.0;

    for (int a = 0; a < eNoN; a++) {
      for (int i = 0; i < dof; i++) {
        lR(i,a) = output.lR(i,a) + output.lRdyn(i,a);
      }
    }

    for (int a = 0; a < eNoN; a++) {
      for (int b = 0; b < eNoN; b++) {
        for (int idx = 0; idx < dof*dof; idx++) {
          lK(idx,a,b) = afu*output.lKState(idx,a,b) + afv*output.lKRate(idx,a,b) + am*output.lKMass(idx,a,b);
        }
      }
    }

    if (historyLengthPerElement > 0) {
      std::copy(output.historyUpdated.begin(), output.historyUpdated.end(),
                meshHistoryUpdated.begin() + static_cast<std::size_t>(e) * historyLengthPerElement);
    }

    // meshHistoryUpdated (the trial state from this Newton iteration) is
    // committed into meshHistory once this time step converges -- see
    // commit_history(), called from main.cpp's iterate_solution().

    eq.linear_algebra->assemble(com_mod, eNoN, ptr, lK, lR);
  }
}

void commit_history(ComMod& com_mod) {
  for (auto& [meshName, updated] : com_mod.ccbActiveCmmGandrHistoryUpdated) {
    com_mod.ccbActiveCmmGandrHistory[meshName] = updated;
  }
}

void advance_time_step(ComMod& com_mod, const CmMod& cm_mod, const SolutionStates& solutions)
{
  const double time = com_mod.time;
  const double previous_time = time - com_mod.dt;
  const bool master = com_mod.cm.mas(cm_mod);

  for (int iEq = 0; iEq < com_mod.nEq; iEq++) {
    auto& eq = com_mod.eq[iEq];

    if (eq.phys != consts::EquationType::phys_def_diffu) {
      continue;
    }

    for (int iDmn = 0; iDmn < eq.nDmn; iDmn++) {
      auto& dmn = eq.dmn[iDmn];

      if (dmn.phys != consts::EquationType::phys_def_diffu) {
        continue;
      }

      // As in FEDDLib, the one-time initializations of this step's first
      // switch-ons are evaluated with the previous step's flag values; the
      // new values are assigned afterwards.
      for (auto& flag : dmn.ccb_active_cmm_gandr_flag_segments) {
        if (!time_segments::in_intervals(flag.intervals, time) || flag.initialized) {
          continue;
        }
        flag.initialized = true;

        // A restart after the first switch-on continues from the restarted
        // state (as in FEDDLib).
        const double first_start = flag.intervals.front()[0];
        const bool restarted_after_start = previous_time > first_start &&
                                           !time_segments::approx_equal(previous_time, first_start);

        if (flag.initialization != "none" && !restarted_after_start) {
          if (master) {
            std::cout << " [def_diffu] t = " << time << ": " << flag.initialization
                      << " initialization for " << flag.name << std::endl;
          }
          initialize_flag(com_mod, iEq, iDmn, flag.initialization, solutions);
        }
      }

      bool changed = false;

      for (auto& flag : dmn.ccb_active_cmm_gandr_flag_segments) {
        const double value = time_segments::in_intervals(flag.intervals, time) ? 1.0 : 0.0;
        double& current = dmn.ccb_active_cmm_gandr_domain_data[flag.index];

        if (value != current) {
          changed = true;
          if (master) {
            std::cout << " [def_diffu] t = " << time << ": " << flag.name << " = " << value << std::endl;
          }
        }
        current = value;
      }

      if (changed && master && flag_value(dmn, "ActiveBool") == 1.0 && flag_value(dmn, "ReorientationBool") == 1.0) {
        std::cout << " [def_diffu] WARNING: active response and reorientation are switched on at the same time." << std::endl;
      }
    }
  }
}

};
