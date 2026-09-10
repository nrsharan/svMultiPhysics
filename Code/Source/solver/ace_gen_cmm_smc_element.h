// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ACE_GEN_CMM_SMC_ELEMENT_H
#define ACE_GEN_CMM_SMC_ELEMENT_H

// This header isolates every dependency on the external Interface2/AceGen
// library behind a small, svMultiPhysics-native API. It wraps
// AceGenInterface::DeformationDiffusionConstrainedMixtureModelSmoothMuscle
// ActiveGrowthReorientationTetrahedra3D10 (the "CCB" constrained-mixture
// active growth-and-remodeling element), which computes a complete 10-node
// tet (30 displacement + 10 concentration DOF) element system in one call.
//
// svMultiPhysics is built with Interface2 support only when configured with
// -DSV_USE_INTERFACE2=ON (see Code/Source/solver/CMakeLists.txt); the
// SV_HAVE_INTERFACE2 preprocessor macro reflects that at compile time. Every
// function here is safe to call regardless of how svMultiPhysics was built:
// if support was not compiled in, they throw std::runtime_error with a
// clear message instead of the Interface2 calls silently being unavailable.
//
// The point of this header is that no other svMultiPhysics translation unit
// needs to know about Interface2's API, its DOF/array layout, or its
// domain-data name quirks (see the .cpp for one such quirk) -- callers only
// deal in svMultiPhysics's own per-node dof-block convention: 3 displacement
// components followed by 1 concentration component, for each of the
// element's 10 nodes (i.e. this equation's dof=nsd+1=4 local layout).

#include "Array.h"
#include "Array3.h"
#include "Vector.h"

#include <map>
#include <string>
#include <vector>

namespace ace_gen_cmm_smc {

/// Element metadata queried once at setup time (independent of any
/// particular element's nodal data).
struct ElementInfo {
  /// Number of history/internal variables persisted per element (currently
  /// 39 per Gauss point for this AceGen kernel). Read at runtime rather
  /// than hardcoded, so nothing here needs to change if the AceGen model is
  /// regenerated with a different history layout.
  int historyLengthPerElement = 0;

  /// Number of Gauss points the element integrates internally.
  int numberOfGaussPoints = 0;

  /// The element's domain-data parameter names, in the exact order the
  /// 'domainData' array passed to compute() must use. Cleaned of AceGen's
  /// internal Mathematica-symbol markup (see the .cpp for details).
  std::vector<std::string> domainDataNames;
};

/// Query element metadata for the given integration code (18 -> 4 Gauss
/// points, 19 -> 5 Gauss points). Throws std::runtime_error if
/// svMultiPhysics was not built with Interface2 support.
ElementInfo get_element_info(int integrationCode);

/// Build the ordered domain-data array compute() expects, by looking up
/// each of 'info.domainDataNames' in 'named_params' (typically
/// dmnType::ccb_active_cmm_gandr_params, populated from the
/// <CCBActiveCMMGandR> XML block). Throws std::runtime_error naming the
/// first missing parameter, rather than silently defaulting it, since a
/// missing material parameter is a user input error that must not pass
/// silently into a nonlinear solid mechanics solve.
std::vector<double> build_domain_data(const ElementInfo& info,
                                       const std::map<std::string, double>& named_params);

/// Build the initial (t=0, pre-solve) history vector for one element, by
/// tiling the AceGen notebook's own per-Gauss-point initial-history vector
/// (Mathematica "SingleGP") across info.numberOfGaussPoints. Interface2 has
/// no API to query this, so it is hardcoded here (see the .cpp) from the
/// notebook this kernel was generated from. Returns an empty vector if
/// info.historyLengthPerElement is 0. Throws std::runtime_error if
/// info.historyLengthPerElement is nonzero but not exactly 39 *
/// info.numberOfGaussPoints -- i.e. if the AceGen model's history layout
/// has changed since this was hardcoded and this needs to be updated to
/// match.
std::vector<double> initial_history(const ElementInfo& info);

/// Compute the fully-initialized history for ONE specific element: starts
/// from initial_history()'s generic per-Gauss-point placeholder, then runs
/// Interface2's Task 8 ("InitGrowth", exposed as
/// initializeGrowthOrientationVectors()) using THIS element's own
/// reference-configuration nodal positions to fill in the geometry-
/// dependent fiber/growth-orientation tensor entries (a11-a23, ag11-ag33)
/// that initial_history() leaves at 0 as a placeholder. This is
/// necessarily per-element (not a single mesh-wide constant), since fiber/
/// growth orientation varies spatially -- e.g. for a cylindrical geometry,
/// the AceGen kernel derives circumferential/radial/axial directions from
/// each Gauss point's (x,y) position, assuming (0,0) is the structure's
/// centerline axis.
///
/// 'positions' follows the same (3,10) svMultiPhysics per-node convention
/// as ElementInput::positions (reference-configuration coordinates).
/// 'domainData' must already be ordered per ElementInfo::domainDataNames
/// (see build_domain_data()). Throws std::runtime_error if svMultiPhysics
/// was not built with Interface2 support.
std::vector<double> initialize_element_history(const ElementInfo& info,
                                                const Array<double>& positions,
                                                const std::vector<double>& domainData,
                                                double subIterationTolerance,
                                                double timeIncrement,
                                                double time,
                                                int integrationCode,
                                                int elementID);

/// Inputs for one element evaluation, in svMultiPhysics's per-node
/// dof-block convention.
struct ElementInput {
  // Note: sized via a constructor (not brace-init default member
  // initializers) so that Vector<T>'s initializer_list constructor isn't
  // accidentally selected in place of the "N zero-initialized entries"
  // constructor (the classic vector<int>{10} vs. vector<int>(10) trap).
  ElementInput()
      : positions(3, 10), displacements(3, 10), accelerations(3, 10),
        concentrations(10), rates(10) {}

  Array<double> positions;      // (3,10) reference-configuration coordinates
  Array<double> displacements;  // (3,10) current nodal displacements
  Array<double> accelerations;  // (3,10) current nodal accelerations
  Vector<double> concentrations; // (10) current nodal concentrations
  Vector<double> rates;          // (10) current nodal concentration rates

  /// Domain-data values, ordered to match ElementInfo::domainDataNames.
  std::vector<double> domainData;

  /// Converged history from the previous time step (size
  /// ElementInfo::historyLengthPerElement; may be empty).
  std::vector<double> history;

  double subIterationTolerance;
  double timeIncrement;
  double time;
  int integrationCode;
  int elementID;
};

/// Outputs of one element evaluation, packed into svMultiPhysics's local
/// dense-array convention: lR/lRdyn are (dof,eNoN) with dof=4 (3
/// displacement components then 1 concentration component), and the
/// dof*dof x eNoN x eNoN tangent blocks use flat row index i*dof+j for
/// row-component i, column-component j. All blocks are the element's raw,
/// unscaled contributions; how they are combined into the Newton system is
/// up to the caller (see def_diffu.cpp).
struct ElementOutput {
  ElementOutput()
      : lR(4, 10), lRdyn(4, 10), lKState(16, 10, 10), lKRate(16, 10, 10),
        lKMass(16, 10, 10) {}

  /// Interface2's Rint (displacement rows) and Rc (concentration row): the
  /// element's internal residual.
  Array<double> lR;         // (4,10)

  /// Interface2's Rdyn (displacement rows only; concentration row is
  /// zero): the inertia residual computed from ElementInput::accelerations.
  Array<double> lRdyn;      // (4,10)

  /// d(residual)/d(displacement, concentration), i.e. Interface2's Kuu, Kuc,
  /// Kcu, Kcc placed at their (i,j) dof-block positions.
  Array3<double> lKState;   // (16,10,10)

  /// d(residual)/d(concentration rate), i.e. Interface2's Mc placed at dof
  /// block (3,3).
  Array3<double> lKRate;    // (16,10,10)

  /// d(Rdyn)/d(acceleration), i.e. Interface2's mass matrix Mu (density
  /// included) placed at the displacement-displacement (i,j in 0..2) dof
  /// blocks; concentration rows/columns are zero.
  Array3<double> lKMass;    // (16,10,10)

  /// Trial (not-yet-committed) updated history for this element.
  std::vector<double> historyUpdated;
};

/// Run one element evaluation through Interface2 and repack the raw
/// residual/tangent output into svMultiPhysics's local dense-array
/// convention (see ElementOutput). Throws std::runtime_error if
/// svMultiPhysics was not built with Interface2 support.
ElementOutput compute(const ElementInput& input);

} // namespace ace_gen_cmm_smc

#endif // ACE_GEN_CMM_SMC_ELEMENT_H
