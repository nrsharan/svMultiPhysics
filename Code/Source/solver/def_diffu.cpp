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

#include "def_diffu.h"

#include "ace_gen_cmm_smc_element.h"
#include "all_fun.h"
#include "consts.h"
#include "LinearAlgebra.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace def_diffu {

void construct_def_diffu(ComMod& com_mod, CepMod& cep_mod, const mshType& lM, const SolutionStates& solutions)
{
  using namespace consts;

  const auto& Ag = solutions.intermediate.get_acceleration();
  const auto& Yg = solutions.intermediate.get_velocity();
  const auto& Dg = solutions.intermediate.get_displacement();

  const int nsd = com_mod.nsd;
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
  // ComMod::ccbActiveCmmGandrHistory). Lazily initialized on first use, once
  // the owning domain's historyLengthPerElement is known.
  auto& meshHistory = com_mod.ccbActiveCmmGandrHistory[lM.name];
  auto& meshHistoryUpdated = com_mod.ccbActiveCmmGandrHistoryUpdated[lM.name];

  if (meshHistory.empty()) {
    // One-time history initialization: tile the generic per-Gauss-point
    // placeholder (ace_gen_cmm_smc::initial_history) across every element.
    // This does NOT need to be geometry-aware: the AceGen kernel
    // unconditionally overwrites the geometry-dependent fiber/growth-
    // orientation entries (a11-a23) and the growth tensor (ag11-ag33)
    // itself, from scratch, inside its own Task 3 compute() on the very
    // first call (gated on time==timeIncrement), so the placeholder's values
    // there are discarded.
    int historyLengthPerElement = -1;

    for (int e = 0; e < lM.nEl; e++) {
      cDmn = all_fun::domain(com_mod, lM, cEq, e);
      auto& dmn = eq.dmn[cDmn];

      if (dmn.phys != EquationType::phys_def_diffu) {
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

    ace_gen_cmm_smc::ElementInput input;
    input.integrationCode = dmn.ccb_active_cmm_gandr_integration_code;
    input.subIterationTolerance = dmn.ccb_active_cmm_gandr_subiteration_tolerance;
    input.domainData = dmn.ccb_active_cmm_gandr_domain_data;
    input.timeIncrement = dt;
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

    for (int a = 0; a < eNoN; a++) {
      int Ac = lM.IEN(a,e);
      ptr(a) = Ac;

      for (int i = 0; i < nsd; i++) {
        input.positions(i,a) = com_mod.x(i,Ac);
        input.displacements(i,a) = Dg(eq.s+i, Ac);
        input.accelerations(i,a) = Ag(eq.s+i, Ac);
      }

      input.concentrations(a) = Dg(eq.s+nsd, Ac);
      input.rates(a) = Yg(eq.s+nsd, Ac);
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

};
