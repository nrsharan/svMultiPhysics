// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "ace_gen_cmm_smc_element.h"

#include <cctype>
#include <stdexcept>

#ifdef SV_HAVE_INTERFACE2
#include "aceinterface.hpp"
#endif

namespace ace_gen_cmm_smc {

namespace {

/// Initial per-Gauss-point history state, from the AceGen/Mathematica
/// notebook's own "SingleGP" vector (as supplied by the model author; see
/// SMC_Coupled_Active_Growth_Reorientation_2026_09_04.nb). This has no
/// Interface2 dependency -- it is pure hardcoded data, not queried from the
/// library -- so it lives outside the SV_HAVE_INTERFACE2 guard below.
///
/// Order (1-based Mathematica indices, as given by the notebook):
///   1  a11              2  a12              3  a13
///   4  a21              5  a22              6  a23
///   7  RhoRe            8  RhoRcoll         9  ag11
///  10  ag12            11  ag13            12  ag21
///  13  ag22            14  ag23            15  ag31
///  16  ag32            17  ag33            18  MuR1np1
///  19  MuR2np1         20  MuN1np1         21  MuN2np1
///  22  LambdaBarC1     23  LambdaBarC2     24  nA1
///  25  nA2             26  nB1             27  nB2
///  28  nC1             29  nC2             30  nD1
///  31  nD2             32  Lambdaa1        33  Lambdaa2
///  34  k251            35  k252            36  LambdaBarP1
///  37  LambdaBarP2     38  RhoRSMC         39  nBar
constexpr int kHistoryValuesPerGaussPoint = 39;

/// 0-based positions of Lambdaa1 and Lambdaa2 in the per-Gauss-point history
/// (entries 32 and 33 of the notebook order above).
constexpr int kHistoryIndexLambdaa1 = 31;
constexpr int kHistoryIndexLambdaa2 = 32;
constexpr double kInitialHistorySingleGP[kHistoryValuesPerGaussPoint] = {
    0.0,     0.0,     0.0,      // a11, a12, a13
    0.0,     0.0,     0.0,      // a21, a22, a23
    0.0,     0.0,     0.0,      // RhoRe, RhoRcoll, ag11
    0.0,     0.0,     0.0,      // ag12, ag13, ag21
    0.0,     0.0,     0.0,      // ag22, ag23, ag31
    0.0,     0.0,     0.0,      // ag32, ag33, MuR1np1
    0.0,     0.0,     0.0,      // MuR2np1, MuN1np1, MuN2np1
    1.0,     1.0,     1.0,      // LambdaBarC1, LambdaBarC2, nA1
    1.0,     0.0,     0.0,      // nA2, nB1, nB2
    0.0,     0.0,     0.0,      // nC1, nC2, nD1
    0.0,     1.0,     1.0,      // nD2, Lambdaa1, Lambdaa2
    1.82758, 1.82758, 1.0,      // k251, k252, LambdaBarP1
    1.0,     0.0,     0.0,      // LambdaBarP2, RhoRSMC, nBar
};

#ifdef SV_HAVE_INTERFACE2

using AceGenInterface::DeformationDiffusionConstrainedMixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10;

/// Interface2's raw domain-data names for this element mix two
/// inconsistent conventions (discovered by inspecting
/// SMC_Interface_Active_Growth_CMM_Reorientation.c's 'gdcs' table):
///
///   - Most entries put a Mathematica pretty-printed form before the dash
///     and the clean, C-identifier-safe name after it, e.g.
///     "k$[Eta]$Plus -kEtaPlus" -> "kEtaPlus".
///   - At least one entry ("fA -Fibre angle") instead puts the clean name
///     before the dash and a human-readable, space-containing description
///     after it.
///
/// This function takes the post-dash, pre-underscore substring (the
/// FEDDLib convention for this AceGen naming scheme), then falls back to
/// the pre-dash token if that substring is empty or contains whitespace
/// (never valid in an actual parameter name).
std::string clean_domain_data_name(const std::string& raw) {
  auto dash = raw.find('-');
  std::string candidate = (dash == std::string::npos) ? raw : [&] {
    auto underscore = raw.find('_', dash);
    return (underscore == std::string::npos)
               ? raw.substr(dash + 1)
               : raw.substr(dash + 1, underscore - dash - 1);
  }();

  bool looks_invalid = candidate.empty() ||
      candidate.find(' ') != std::string::npos;

  if (looks_invalid && dash != std::string::npos) {
    candidate = raw.substr(0, dash);
    while (!candidate.empty() &&
           std::isspace(static_cast<unsigned char>(candidate.back()))) {
      candidate.pop_back();
    }
  }

  return candidate;
}

/// Maps AceGen's local tet10 node numbering (the element's 'rnodes' table in
/// SMC_Interface_Active_Growth_CMM_Reorientation.c) to svMultiPhysics's
/// local numbering as returned by mshType::IEN(a,e): svMultiPhysics node
/// kAceGenNodeToVtkNode[a] is AceGen's node 'a'. svMultiPhysics's mesh
/// loader reorders the element nodes (its IEN corners give a negative
/// signed volume for gmsh tet10 meshes), so the table is expressed in IEN
/// order, not in the mesh file's order.
///
/// Two conditions must both hold: each AceGen node must receive the
/// physical node matching its reference coordinate, and the corner
/// assignment must be an even permutation of a positively oriented one.
/// An odd permutation still places nodal values at plausible locations but
/// gives an inverted element (negative Jacobian), whose kinematics become
/// increasingly wrong with deformation. The table was checked against the
/// analytic signed volume of real mesh elements, using Interface2's
/// postProcess() nodal volume weights at zero displacement, through the
/// def_diffu.cpp assembly path.
constexpr int kAceGenNodeToVtkNode[10] = {2, 0, 3, 1, 6, 7, 9, 5, 4, 8};

void flatten_positions(const Array<double>& src, double* dst) {
  for (int a = 0; a < 10; a++) {
    int v = kAceGenNodeToVtkNode[a];
    for (int i = 0; i < src.nrows(); i++) {
      dst[3 * a + i] = src(i, v);
    }
  }
}

#endif // SV_HAVE_INTERFACE2

} // namespace

ElementInfo get_element_info(int integrationCode) {
#ifndef SV_HAVE_INTERFACE2
  throw std::runtime_error(
      "The 'deformation-diffusion' equation requires svMultiPhysics to be "
      "built with Interface2 support (configure with -DSV_USE_INTERFACE2=ON "
      "and point CMAKE_PREFIX_PATH/Interface2_DIR at your Interface2 "
      "install).");
#else
  ElementInfo info;

  // The (integrationCode) constructor is for querying history length and
  // Gauss point count only (see aceinterface.hpp).
  DeformationDiffusionConstrainedMixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10
      sizing_elem(integrationCode);
  info.historyLengthPerElement = sizing_elem.getHistoryLength();
  info.numberOfGaussPoints = sizing_elem.getNumberOfGaussPoints();

  // The no-arg constructor is for querying domain-data/post-data names and
  // counts only.
  DeformationDiffusionConstrainedMixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10
      naming_elem;
  int numberOfDomainData = naming_elem.getNumberOfDomainData();
  char** rawNames = naming_elem.getDomainDataNames();

  info.domainDataNames.resize(numberOfDomainData);
  for (int i = 0; i < numberOfDomainData; i++) {
    info.domainDataNames[i] = clean_domain_data_name(std::string(rawNames[i]));
  }

  return info;
#endif
}

std::vector<double> build_domain_data(const ElementInfo& info,
                                       const std::map<std::string, double>& named_params) {
  std::vector<double> domainData(info.domainDataNames.size());

  for (std::size_t i = 0; i < info.domainDataNames.size(); i++) {
    const auto& name = info.domainDataNames[i];
    auto it = named_params.find(name);
    if (it == named_params.end()) {
      throw std::runtime_error(
          "The 'deformation-diffusion' equation's CCBActiveCMMGandR "
          "material parameters are missing required parameter '" + name +
          "'. Add a <" + name + "> ... </" + name +
          "> element under <CCBActiveCMMGandR> in the Domain's XML.");
    }
    domainData[i] = it->second;
  }

  return domainData;
}

std::vector<double> initial_history(const ElementInfo& info) {
  if (info.historyLengthPerElement == 0) {
    return {};
  }

  if (info.historyLengthPerElement != kHistoryValuesPerGaussPoint * info.numberOfGaussPoints) {
    throw std::runtime_error(
        "ace_gen_cmm_smc::initial_history: historyLengthPerElement (" +
        std::to_string(info.historyLengthPerElement) + ") is not " +
        std::to_string(kHistoryValuesPerGaussPoint) + " * numberOfGaussPoints (" +
        std::to_string(info.numberOfGaussPoints) + "). The hardcoded initial "
        "history seed no longer matches this AceGen kernel's history layout "
        "and needs to be updated.");
  }

  std::vector<double> seed;
  seed.reserve(info.historyLengthPerElement);
  for (int g = 0; g < info.numberOfGaussPoints; g++) {
    seed.insert(seed.end(), std::begin(kInitialHistorySingleGP), std::end(kInitialHistorySingleGP));
  }
  return seed;
}

std::vector<double> initialize_element_history(const ElementInfo& info,
                                                const Array<double>& positions,
                                                const std::vector<double>& domainData,
                                                double subIterationTolerance,
                                                double timeIncrement,
                                                double time,
                                                int integrationCode,
                                                int elementID) {
#ifndef SV_HAVE_INTERFACE2
  throw std::runtime_error(
      "The 'deformation-diffusion' equation requires svMultiPhysics to be "
      "built with Interface2 support (configure with -DSV_USE_INTERFACE2=ON "
      "and point CMAKE_PREFIX_PATH/Interface2_DIR at your Interface2 "
      "install).");
#else
  if (info.historyLengthPerElement == 0) {
    return {};
  }

  double positionsFlat[30];
  flatten_positions(positions, positionsFlat);

  double displacements[30] = {0.0};
  double concentrations[10] = {0.0};
  double accelerations[30] = {0.0};
  double rates[10] = {0.0};

  std::vector<double> domainDataCopy = domainData;
  // Placeholder history (see initial_history()) -- Task 8 only overwrites
  // the geometry-dependent orientation entries; everything else passes
  // through unchanged, so this baseline must already hold the correct
  // non-orientation initial values (e.g. LambdaBarC1=1, k251=1.82758, ...).
  std::vector<double> historyPlaceholder = initial_history(info);

  DeformationDiffusionConstrainedMixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10
      elem(positionsFlat, displacements, concentrations, accelerations, rates,
           domainDataCopy.data(), historyPlaceholder.data(), subIterationTolerance,
           timeIncrement, time, integrationCode, elementID);

  return elem.initializeGrowthOrientationVectors();
#endif
}

ElementOutput compute(const ElementInput& input) {
#ifndef SV_HAVE_INTERFACE2
  throw std::runtime_error(
      "The 'deformation-diffusion' equation requires svMultiPhysics to be "
      "built with Interface2 support (configure with -DSV_USE_INTERFACE2=ON "
      "and point CMAKE_PREFIX_PATH/Interface2_DIR at your Interface2 "
      "install).");
#else
  double positions[30];
  double displacements[30];
  double concentrations[10];
  double accelerations[30];
  double rates[10];

  flatten_positions(input.positions, positions);
  flatten_positions(input.displacements, displacements);
  flatten_positions(input.accelerations, accelerations);
  for (int a = 0; a < 10; a++) {
    int v = kAceGenNodeToVtkNode[a];
    concentrations[a] = input.concentrations(v);
    rates[a] = input.rates(v);
  }

  // Interface2 leaves 'domainData'/'history' non-const void*-like raw
  // pointers; copy locally so we can safely pass mutable pointers without
  // aliasing the caller's storage.
  std::vector<double> domainData = input.domainData;
  std::vector<double> history = input.history;

  DeformationDiffusionConstrainedMixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10
      elem(positions, displacements, concentrations, accelerations, rates,
           domainData.data(), history.data(), input.subIterationTolerance,
           input.timeIncrement, input.time, input.integrationCode,
           input.elementID);

  int errorCode = elem.compute(/*computeTangent=*/true);
  if (errorCode != 0) {
    throw std::runtime_error(
        "The Interface2/AceGen CCB constrained-mixture element failed to "
        "converge (element " + std::to_string(input.elementID) +
        ", error code " + std::to_string(errorCode) + ").");
  }

  double* Rint = elem.getResiduumVectorRint(); // 30
  double* Rdyn = elem.getResiduumVectorRdyn(); // 30 -- real inertia residual (rho*a)
  double* Rc = elem.getResiduumVectorRc();     // 10
  double** Kuu = elem.getStiffnessMatrixKuu(); // 30x30
  double** Kuc = elem.getStiffnessMatrixKuc(); // 30x10
  double** Kcu = elem.getStiffnessMatrixKcu(); // 10x30
  double** Kcc = elem.getStiffnessMatrixKcc(); // 10x10
  double** Mu = elem.getMassMatrixMu();        // 30x30 -- real mass matrix (density baked in)
  double** Mc = elem.getMassMatrixMc();        // 10x10
  double* historyUpdated = elem.getHistoryUpdated();

  ElementOutput out;

  constexpr int dof = 4;

  // All indices below are AceGen's own local node numbering; every write
  // into 'out' remaps through kAceGenNodeToVtkNode so 'out' ends up indexed
  // in svMultiPhysics's own (VTK) local node numbering, matching 'ptr' in
  // def_diffu.cpp.
  for (int a = 0; a < 10; a++) {
    int va = kAceGenNodeToVtkNode[a];
    for (int i = 0; i < 3; i++) {
      out.lR(i, va) = Rint[3 * a + i];
      out.lRdyn(i, va) = Rdyn[3 * a + i];
    }
    out.lR(3, va) = Rc[a];
  }

  for (int a = 0; a < 10; a++) {
    int va = kAceGenNodeToVtkNode[a];
    for (int b = 0; b < 10; b++) {
      int vb = kAceGenNodeToVtkNode[b];
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          out.lKState(i * dof + j, va, vb) = Kuu[3 * a + i][3 * b + j];
          out.lKMass(i * dof + j, va, vb) = Mu[3 * a + i][3 * b + j];
        }
        out.lKState(i * dof + 3, va, vb) = Kuc[3 * a + i][b];
        out.lKState(3 * dof + i, va, vb) = Kcu[a][3 * b + i];
      }
      out.lKState(3 * dof + 3, va, vb) = Kcc[a][b];
      out.lKRate(3 * dof + 3, va, vb) = Mc[a][b];
    }
  }

  out.historyUpdated.assign(historyUpdated,
                             historyUpdated + input.history.size());

  return out;
#endif
}

#ifdef SV_HAVE_INTERFACE2
namespace {

/// Interface2 input arrays for one element, in AceGen's node order.
struct RawElementInput {
  double positions[30];
  double displacements[30];
  double accelerations[30];
  double concentrations[10];
  double rates[10];
  std::vector<double> domainData;
  std::vector<double> history;
};

RawElementInput raw_element_input(const ElementInput& input) {
  RawElementInput raw;
  flatten_positions(input.positions, raw.positions);
  flatten_positions(input.displacements, raw.displacements);
  flatten_positions(input.accelerations, raw.accelerations);
  for (int a = 0; a < 10; a++) {
    int v = kAceGenNodeToVtkNode[a];
    raw.concentrations[a] = input.concentrations(v);
    raw.rates[a] = input.rates(v);
  }
  raw.domainData = input.domainData;
  raw.history = input.history;
  return raw;
}

} // namespace
#endif

std::vector<double> history_with_active_stretches(const ElementInput& input) {
#ifndef SV_HAVE_INTERFACE2
  throw std::runtime_error("The 'deformation-diffusion' equation requires svMultiPhysics to be built with Interface2 support.");
#else
  RawElementInput raw = raw_element_input(input);
  DeformationDiffusionConstrainedMixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10
      elem(raw.positions, raw.displacements, raw.concentrations, raw.accelerations, raw.rates,
           raw.domainData.data(), raw.history.data(), input.subIterationTolerance,
           input.timeIncrement, input.time, input.integrationCode, input.elementID);

  std::vector<double> stretches = elem.getGaussPointStretches();
  std::vector<double> history = input.history;
  const int nGP = static_cast<int>(stretches.size()) / 2;

  if (nGP == 0 || history.size() != static_cast<std::size_t>(nGP) * kHistoryValuesPerGaussPoint) {
    throw std::runtime_error("ace_gen_cmm_smc::history_with_active_stretches: unexpected history or stretch size.");
  }

  for (int g = 0; g < nGP; g++) {
    history[g * kHistoryValuesPerGaussPoint + kHistoryIndexLambdaa1] = stretches[2 * g];
    history[g * kHistoryValuesPerGaussPoint + kHistoryIndexLambdaa2] = stretches[2 * g + 1];
  }

  return history;
#endif
}

std::vector<double> history_with_growth_orientation(const ElementInput& input) {
#ifndef SV_HAVE_INTERFACE2
  throw std::runtime_error("The 'deformation-diffusion' equation requires svMultiPhysics to be built with Interface2 support.");
#else
  RawElementInput raw = raw_element_input(input);
  DeformationDiffusionConstrainedMixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10
      elem(raw.positions, raw.displacements, raw.concentrations, raw.accelerations, raw.rates,
           raw.domainData.data(), raw.history.data(), input.subIterationTolerance,
           input.timeIncrement, input.time, input.integrationCode, input.elementID);

  std::vector<double> history = elem.initializeGrowthOrientationVectors();

  if (history.size() != input.history.size()) {
    throw std::runtime_error("ace_gen_cmm_smc::history_with_growth_orientation: unexpected history size.");
  }

  return history;
#endif
}

} // namespace ace_gen_cmm_smc
