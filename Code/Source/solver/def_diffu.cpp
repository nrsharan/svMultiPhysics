// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

// Assembly for EquationType::phys_def_diffu: a monolithic 4-dof/node (3
// displacement + 1 concentration) equation whose element residual and
// tangent are computed entirely by the Interface2/AceGen CCB
// constrained-mixture active growth-and-remodeling element. There is no
// inner Gauss-point loop: Interface2 integrates all Gauss points internally
// in one compute() call per element.
//
// Time integration is quasi-static, as in FEDDLib's implementation of this
// element (feddlib/core/AceFemAssembly/specific/
// AssembleFE_SCI_SMC_CMM_Active_Growth_Reorientation_def.hpp): the
// accelerations are zero and the element's inertia residual (Rdyn) and mass
// matrix (Mu) are not assembled. The tangent is d(residual)/d(state):
// Kuu/Kuc/Kcu unscaled and Kcc + Mc/dt, with the concentration rate taken by
// backward Euler from the last converged state. The Newton unknown is
// therefore the state increment itself, so Integrator::corrector() applies
// Dn -= R directly for phys_def_diffu instead of the Newmark relations.

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

  // Quasi-static assembly (see the header comment above): use the FULL
  // current Newton iterate (Dn), not svMultiPhysics's generalized-alpha
  // blended 'Dg' intermediate, and the last CONVERGED state (Do) to form a
  // plain backward-Euler concentration rate -- exactly matching FEDDLib's
  // reference implementation of this element's sibling.
  const auto& Dn = solutions.current.get_displacement();
  const auto& Do = solutions.old.get_displacement();

  const int nsd = com_mod.nsd;
  const int cEq = com_mod.cEq;
  auto& eq = com_mod.eq[cEq];
  auto& cDmn = com_mod.cDmn;
  const double dt = com_mod.dt;

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
        input.displacements(i,a) = Dn(eq.s+i, Ac);
        input.accelerations(i,a) = 0.0;
      }

      double cNew = Dn(eq.s+nsd, Ac);
      double cOld = Do(eq.s+nsd, Ac);
      input.concentrations(a) = cNew;
      input.rates(a) = (cNew - cOld) / dt;
    }

    auto output = ace_gen_cmm_smc::compute(input);

    lR = 0.0;
    lK = 0.0;

    // lRdyn/lKMass (Rdyn/Mu) are intentionally never added -- see the header
    // comment. lKState is used unscaled (the Newton unknown is the state
    // increment itself); lKRate only ever populates the concentration-
    // concentration block (Mc), where dividing by dt turns it into the
    // backward-Euler capacitance term Mc/dt added to Kcc.
    for (int a = 0; a < eNoN; a++) {
      for (int i = 0; i < dof; i++) {
        lR(i,a) = output.lR(i,a);
      }
    }

    for (int a = 0; a < eNoN; a++) {
      for (int b = 0; b < eNoN; b++) {
        for (int idx = 0; idx < dof*dof; idx++) {
          lK(idx,a,b) = output.lKState(idx,a,b) + output.lKRate(idx,a,b)/dt;
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

void reset_dynamics(ComMod& com_mod, SolutionStates& solutions) {
  auto& Yo = solutions.old.get_velocity();
  auto& Ao = solutions.old.get_acceleration();

  for (auto& eq : com_mod.eq) {
    if (eq.phys != consts::EquationType::phys_def_diffu) {
      continue;
    }
    for (int a = 0; a < com_mod.tnNo; a++) {
      for (int i = eq.s; i <= eq.e; i++) {
        Yo(i,a) = 0.0;
        Ao(i,a) = 0.0;
      }
    }
  }
}

};
