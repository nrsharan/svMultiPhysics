// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ACE_GEN_CMM_SMC_ELEMENT_H
#define ACE_GEN_CMM_SMC_ELEMENT_H

// This header isolates every dependency on the external Interface2/AceGen
// library behind a small, svMultiPhysics-native API. It wraps the two
// Interface2 "CCB" active growth-and-remodeling elements, each of which
// computes a complete 10-node tet (30 displacement + 10 concentration DOF)
// element system in one call:
//
//   - AceGenInterface::DeformationDiffusionConstrainedMixtureModelSmoothMuscle
//     ActiveGrowthReorientationTetrahedra3D10, the constrained-mixture element
//     (Model::ConstrainedMixture, parameters in <CCBActiveCMMGandR>);
//   - AceGenInterface::DeformationDiffusionSmoothMuscleActiveGrowth
//     ReorientationTetrahedra3D10, the smooth-muscle element without the
//     constrained mixture (Model::SmoothMuscle, parameters in
//     <CCBActiveGandR>).
//
// Both elements have the same interface and the same local node numbering;
// their parameters, history and post-processing quantities differ and are
// queried from the element (get_element_info()), except for the initial
// history, which is hardcoded per model (see the .cpp).
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

/// The Interface2 element a domain uses.
enum class Model {
  /// Constrained-mixture element, parameters in <CCBActiveCMMGandR>.
  ConstrainedMixture,
  /// Smooth-muscle element without the constrained mixture, parameters in
  /// <CCBActiveGandR>.
  SmoothMuscle
};

/// The XML element holding the parameters of 'model'.
std::string xml_block_name(Model model);

/// Element metadata queried once at setup time (independent of any
/// particular element's nodal data).
struct ElementInfo {
  /// The element.
  Model model = Model::ConstrainedMixture;

  /// Number of history/internal variables persisted per element (39 per
  /// Gauss point for the constrained-mixture element, 34 for the
  /// smooth-muscle element). Read at runtime rather than hardcoded.
  int historyLengthPerElement = 0;

  /// Number of Gauss points the element integrates internally.
  int numberOfGaussPoints = 0;

  /// The element's domain-data parameter names, in the exact order the
  /// 'domainData' array passed to compute() must use. Cleaned of AceGen's
  /// internal Mathematica-symbol markup (see the .cpp for details).
  std::vector<std::string> domainDataNames;

  /// The element's post-processing quantity names, in the row order of
  /// post_process()'s result. The first ("Volume") is the nodal weight the
  /// other quantities are accumulated with.
  std::vector<std::string> postDataNames;
};

/// One output field made of post-processing quantities: 'components'
/// consecutive rows of post_process()'s result, starting at 'firstRow'.
struct PostField {
  std::string name;
  int firstRow = 0;
  int components = 1;
};

/// Group the element's post-processing quantities into output fields by
/// name: nine consecutive names <P>xx, <P>xy, <P>xz, <P>yx, ..., <P>zz form
/// the 3x3 tensor <P> (row-major, e.g. the Cauchy stress S), three
/// consecutive names <P>1, <P>2, <P>3 the vector <P> (e.g. the fiber
/// direction a1 from a11, a12, a13), and every other name is a scalar field.
/// The nodal weight "Volume" is not a field. Throws std::runtime_error if
/// the first name is not "Volume".
std::vector<PostField> post_fields(const ElementInfo& info);

/// Query element metadata of 'model' for the given integration code (18 ->
/// 4 Gauss points, 19 -> 5 Gauss points). Throws std::runtime_error if
/// svMultiPhysics was not built with Interface2 support.
ElementInfo get_element_info(int integrationCode, Model model = Model::ConstrainedMixture);

/// Build the ordered domain-data array compute() expects, by looking up
/// each of 'info.domainDataNames' in 'named_params' (typically
/// dmnType::ccb_active_cmm_gandr_params, populated from the element's XML
/// block). Throws std::runtime_error naming the first missing parameter,
/// rather than silently defaulting it, since a missing material parameter is
/// a user input error that must not pass silently into a nonlinear solid
/// mechanics solve.
std::vector<double> build_domain_data(const ElementInfo& info,
                                       const std::map<std::string, double>& named_params);

/// Build the initial (t=0, pre-solve) history vector for one element, by
/// tiling the element's per-Gauss-point initial-history vector across
/// info.numberOfGaussPoints. Interface2 has no API to query this, so it is
/// hardcoded per model (see the .cpp). Returns an empty vector if
/// info.historyLengthPerElement is 0. Throws std::runtime_error if
/// info.historyLengthPerElement is nonzero but not the hardcoded length per
/// Gauss point times info.numberOfGaussPoints -- i.e. if the AceGen model's
/// history layout has changed since this was hardcoded and this needs to be
/// updated to match.
std::vector<double> initial_history(const ElementInfo& info);

/// Compute the fully-initialized history for ONE specific element: starts
/// from initial_history()'s generic per-Gauss-point placeholder, then runs
/// Interface2's Task 8 ("InitGrowth", exposed as
/// initializeGrowthOrientationVectors()) using THIS element's own
/// reference-configuration nodal positions to fill in the geometry-
/// dependent growth-orientation tensor entries that initial_history() leaves
/// at 0 as a placeholder. This is necessarily per-element (not a single
/// mesh-wide constant), since fiber/growth orientation varies spatially --
/// e.g. for a cylindrical geometry, the AceGen kernel derives
/// circumferential/radial/axial directions from each Gauss point's (x,y)
/// position, assuming (0,0) is the structure's centerline axis.
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

  /// The element (ElementInfo::model of the domain).
  Model model = Model::ConstrainedMixture;

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

/// Converged history of one element (input.history) with the active
/// stretches at every Gauss point (Lambdaa1/Lambdaa2 of the
/// constrained-mixture element, LambdaA1/LambdaA2 of the smooth-muscle
/// element) replaced by the element's current Gauss-point fiber stretches
/// (Interface2's getGaussPointStretches()), used when the active response
/// first switches on.
std::vector<double> history_with_active_stretches(const ElementInput& input);

/// History returned by Interface2's initializeGrowthOrientationVectors() for
/// the element state in 'input', used when growth first switches on.
std::vector<double> history_with_growth_orientation(const ElementInput& input);

/// Run the element's post-processing task (Interface2's postProcess()) for
/// the element state in 'input'. Returns (ElementInfo::postDataNames.size(),
/// 10) in svMultiPhysics's local node order: row 0 is node a's weight
/// ("Volume") and row k its weighted contribution to quantity k, as the
/// AceGen task accumulates them over the Gauss points. The nodal value of
/// quantity k is the sum of row k over the elements sharing the node divided
/// by the sum of row 0 (see def_diffu::nodal_post_data()).
Array<double> post_process(const ElementInput& input);

} // namespace ace_gen_cmm_smc

#endif // ACE_GEN_CMM_SMC_ELEMENT_H
