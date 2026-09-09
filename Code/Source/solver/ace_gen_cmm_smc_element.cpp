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

/// AceGen's local tet10 node numbering (see this element's ReferenceNodes /
/// 'rnodes' table in SMC_Interface_Active_Growth_CMM_Reorientation.c) is
/// NOT the same as svMultiPhysics's own tet10 local node numbering -- and,
/// importantly, svMultiPhysics's OWN local node numbering (i.e. what
/// mshType::IEN(a,e) actually returns for a=0..9, after its mesh loader has
/// read and internally reordered the mesh file) is *also* not simply the
/// "textbook" VTK quadratic-tetra convention (corners (0,0,0),(1,0,0),
/// (0,1,0),(0,0,1) then mid-edge nodes in edge order (0,1),(1,2),(2,0),
/// (0,3),(1,3),(2,3) -- see FE/Basis/NodeOrderingConventions.cpp) applied
/// directly to the mesh file's own node order: empirically, for every
/// element of a real gmsh-generated mesh checked here, evaluating the
/// signed-volume formula directly on svMultiPhysics's own IEN(0,e)..
/// IEN(3,e) corners gives a *negative* volume (confirmed for all 920
/// elements of one test mesh, not just one element -- so this is a
/// uniform, mesh-wide relabeling, not a per-element quirk). This array
/// gives, for each AceGen local node index, the corresponding
/// svMultiPhysics local node index (as returned by IEN) -- i.e.
/// svMultiPhysics node kAceGenNodeToVtkNode[a] IS AceGen's node 'a'.
/// Without this permutation, nodal data would be silently attached to the
/// wrong physical node.
///
/// Getting this right requires two independent things to both be correct,
/// and it is easy to get one right and the other wrong (as happened during
/// development -- see below): (1) each AceGen local index must receive the
/// physical corner/edge-midpoint that actually matches its own reference
/// coordinate (a value-correctness question), and (2) the OVERALL
/// corner assignment must be an EVEN permutation of a positively-oriented
/// reference assignment, out of the 24 possible corner assignments only 12
/// preserve a positive Jacobian determinant; the other 12 silently produce
/// an inside-out (mirrored) element with a *negative* Jacobian everywhere.
/// This second failure mode is subtle: nodal values still land at
/// plausible-looking locations and small-strain behavior can look
/// approximately fine, but the element is geometrically invalid and the
/// hyperelastic kinematics become wrong in a way that grows with
/// deformation, eventually causing severe nonlinearity/divergence.
///
/// This exact table was derived and verified in two stages against GROUND
/// TRUTH (an analytic signed volume computed independently from a real
/// element's own corner coordinates -- never assumed from documentation),
/// using Interface2's own postProcess() reported nodal volume weights
/// (summed, at zero displacement) as the adapter-side measurement:
///   1. A first candidate ({1,2,3,0,5,9,8,4,6,7}), derived by matching
///      AceGen's reference-node coordinates against the mesh file's own
///      (pre-svMultiPhysics-reordering) node order, turned out to be an
///      orientation-reversing (odd) permutation of that order -- caught by
///      comparing against the mesh file's own volume (positive) and
///      against FEDDLib's working reference for this element's sibling
///      (AssembleFE_SCI_SMC_Active_Growth_Reorientation_def.hpp), which
///      uses that same pre-reordering node order with NO permutation at
///      all and gets the correct sign. Swapping one corner pair fixed the
///      parity relative to that order, giving {2,1,3,0,5,8,9,6,4,7}.
///   2. That "fixed" table was then found to STILL give a negative volume
///      once actually exercised through svMultiPhysics's real assembly
///      path (def_diffu.cpp, using mshType::IEN-ordered positions) --
///      because svMultiPhysics's own internal node order is itself an odd
///      permutation (corners 0 and 1 swapped, verified against real
///      mesh-loader output) of the order used in stage 1. Composing that
///      known corner swap (and its corresponding mid-edge relabeling) with
///      stage 1's table gives the array below, which was verified to
///      reproduce the correct positive volume when exercised through the
///      real svMultiPhysics assembly path, for a real mesh element.
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

double debug_reference_volume(const Array<double>& positions, const std::vector<double>& domainData,
                               int integrationCode, double subIterationTolerance) {
#ifndef SV_HAVE_INTERFACE2
  throw std::runtime_error("Interface2 support not built in.");
#else
  double positionsFlat[30];
  flatten_positions(positions, positionsFlat);
  double displacements[30] = {0.0};
  double accelerations[30] = {0.0};
  double concentrations[10] = {0.0};
  double rates[10] = {0.0};

  std::vector<double> domainDataCopy = domainData;
  std::vector<double> history = initial_history(get_element_info(integrationCode));

  DeformationDiffusionConstrainedMixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10
      elem(positionsFlat, displacements, concentrations, accelerations, rates,
           domainDataCopy.data(), history.data(), subIterationTolerance,
           1.0, 1.0, integrationCode, 0);

  double** post = elem.postProcess(displacements, concentrations, history.data(), rates, accelerations);
  double volSum = 0.0;
  for (int a = 0; a < 10; a++) volSum += post[a][0];
  return volSum;
#endif
}

} // namespace ace_gen_cmm_smc
