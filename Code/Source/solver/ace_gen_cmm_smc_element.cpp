// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "ace_gen_cmm_smc_element.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <type_traits>

#ifdef SV_HAVE_INTERFACE2
#include "aceinterface.hpp"
#endif

namespace ace_gen_cmm_smc {

namespace {

[[noreturn]] void no_interface2() {
  throw std::runtime_error(
      "The 'deformation-diffusion' equation requires svMultiPhysics to be "
      "built with Interface2 support (configure with -DSV_USE_INTERFACE2=ON "
      "and point CMAKE_PREFIX_PATH/Interface2_DIR at your Interface2 "
      "install).");
}

/// Initial per-Gauss-point history state of the constrained-mixture element,
/// from the AceGen/Mathematica notebook's own "SingleGP" vector (as supplied
/// by the model author; see
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
constexpr int kCmmHistoryValuesPerGaussPoint = 39;
constexpr double kCmmInitialHistorySingleGP[kCmmHistoryValuesPerGaussPoint] = {
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

/// Initial per-Gauss-point history state of the smooth-muscle element, as
/// in the model author's element test (Interface2 test/smc_parameters.hpp).
/// Its fiber (a11-a23) and growth-orientation (Ag11-Ag33) entries are
/// placeholders: the element sets the fibers itself on its first step, and
/// the growth orientation is set when growth first switches on.
///
/// Order (0-based):
///   0  LambdaBarC1      1  LambdaBarC2      2  nA1       3  nA2
///   4  nB1              5  nB2              6  nC1       7  nC2
///   8  nD1              9  nD2             10  LambdaA1 11  LambdaA2
///  12  k251            13  k252            14  LambdaBarP1
///  15  LambdaBarP2     16  Theta1          17  Theta2   18  Theta3
///  19-27  Ag11 ... Ag33                    28-33  a11, a12, a13, a21, a22, a23
constexpr int kSmcHistoryValuesPerGaussPoint = 34;
constexpr double kSmcInitialHistorySingleGP[kSmcHistoryValuesPerGaussPoint] = {
    1.0,     1.0,                                     // LambdaBarC1, LambdaBarC2
    1.0,     1.0,     0.0, 0.0,                       // nA1, nA2, nB1, nB2
    0.0,     0.0,     0.0, 0.0,                       // nC1, nC2, nD1, nD2
    1.0,     1.0,                                     // LambdaA1, LambdaA2
    1.82758, 1.82758,                                 // k251, k252
    1.0,     1.0,                                     // LambdaBarP1, LambdaBarP2
    1.0,     1.0,     1.0,                            // Theta1, Theta2, Theta3
    0.0,     0.0,     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // Ag11 ... Ag33
    0.0,     0.0,     0.0, 0.0, 0.0, 0.0,             // a11, a12, a13, a21, a22, a23
};

/// Per-Gauss-point history layout of an element: its initial values and the
/// 0-based positions of the two active stretches.
struct HistoryLayout {
  const double* initial;
  int valuesPerGaussPoint;
  int activeStretch1;
  int activeStretch2;
};

/// The smooth-muscle element's dynamic-residual task (Rdyn) reads the density
/// from domain-data entry 61, beyond its 60 entries, while its mass matrix Mu
/// uses entry 57 ("Density"; a known issue of the generated element, see
/// Interface2's test/README.md). The domain data passed to this element are
/// therefore extended to 62 entries with the density at 61, so that Rdyn =
/// Mu*a and nothing is read out of bounds.
constexpr int kSmcDensityIndex = 57;
constexpr int kSmcRdynDensityIndex = 61;

/// The domain data as passed to the element of 'model' (see above).
std::vector<double> element_domain_data(Model model, const std::vector<double>& domainData) {
  std::vector<double> data = domainData;
  if (model == Model::SmoothMuscle && data.size() > static_cast<std::size_t>(kSmcDensityIndex)) {
    data.resize(std::max(data.size(), static_cast<std::size_t>(kSmcRdynDensityIndex + 1)), 0.0);
    data[kSmcRdynDensityIndex] = domainData[kSmcDensityIndex];
  }
  return data;
}

HistoryLayout history_layout(Model model) {
  if (model == Model::SmoothMuscle) {
    return {kSmcInitialHistorySingleGP, kSmcHistoryValuesPerGaussPoint, 10, 11};
  }
  // Lambdaa1, Lambdaa2: entries 32 and 33 of the notebook order.
  return {kCmmInitialHistorySingleGP, kCmmHistoryValuesPerGaussPoint, 31, 32};
}

#ifdef SV_HAVE_INTERFACE2

using CmmElement =
    AceGenInterface::DeformationDiffusionConstrainedMixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10;
using SmcElement = AceGenInterface::DeformationDiffusionSmoothMuscleActiveGrowthReorientationTetrahedra3D10;

/// Call 'f' with a null pointer to the element class of 'model'; the pointer
/// type selects the class (both have the same interface).
template <class F>
decltype(auto) with_element(Model model, F&& f) {
  if (model == Model::SmoothMuscle) {
    return f(static_cast<SmcElement*>(nullptr));
  }
  return f(static_cast<CmmElement*>(nullptr));
}

template <class Pointer>
using ElementOf = std::remove_pointer_t<Pointer>;

/// Interface2's raw domain-data names for these elements mix two
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
/// This function takes the post-dash, pre-underscore substring (the usual
/// reading of this AceGen naming scheme), then falls back to
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

/// Maps AceGen's local tet10 node numbering (the elements' 'rnodes' table,
/// the same in SMC_Interface_Active_Growth_CMM_Reorientation.c and
/// SMC_Interface_Active_Growth_Reorientation.c) to svMultiPhysics's local
/// numbering as returned by mshType::IEN(a,e): svMultiPhysics node
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
  // Interface2 takes non-const pointers; copies keep the caller's storage
  // from being aliased.
  raw.domainData = element_domain_data(input.model, input.domainData);
  raw.history = input.history;
  return raw;
}

#endif // SV_HAVE_INTERFACE2

} // namespace

std::string xml_block_name(Model model) {
  return model == Model::SmoothMuscle ? "CCBActiveGandR" : "CCBActiveCMMGandR";
}

ElementInfo get_element_info(int integrationCode, Model model) {
#ifndef SV_HAVE_INTERFACE2
  no_interface2();
#else
  ElementInfo info;
  info.model = model;

  with_element(model, [&](auto* tag) {
    using Element = ElementOf<decltype(tag)>;

    // The (integrationCode) constructor is for querying history length and
    // Gauss point count only (see aceinterface.hpp).
    Element sizing_elem(integrationCode);
    info.historyLengthPerElement = sizing_elem.getHistoryLength();
    info.numberOfGaussPoints = sizing_elem.getNumberOfGaussPoints();

    // The no-arg constructor is for querying domain-data/post-data names and
    // counts only.
    Element naming_elem;
    int numberOfDomainData = naming_elem.getNumberOfDomainData();
    char** rawNames = naming_elem.getDomainDataNames();

    info.domainDataNames.resize(numberOfDomainData);
    for (int i = 0; i < numberOfDomainData; i++) {
      info.domainDataNames[i] = clean_domain_data_name(std::string(rawNames[i]));
    }

    // The post-processing names are plain identifiers ("Volume", "Sxx", ...).
    char** rawPostNames = naming_elem.getPostDataNames();
    info.postDataNames.assign(rawPostNames, rawPostNames + naming_elem.getNumberOfPostData());
    return 0;
  });

  // The density workaround of element_domain_data() assumes this layout.
  if (model == Model::SmoothMuscle &&
      (info.domainDataNames.size() != kSmcRdynDensityIndex - 1 ||
       info.domainDataNames[kSmcDensityIndex] != "Density")) {
    throw std::runtime_error(
        "ace_gen_cmm_smc::get_element_info: the smooth-muscle element no longer has 60 domain-data "
        "parameters with Density at position 57; the density workaround for its dynamic residual needs "
        "to be updated.");
  }

  return info;
#endif
}

std::vector<PostField> post_fields(const ElementInfo& info) {
  const auto& names = info.postDataNames;

  if (names.empty() || names[0] != "Volume") {
    throw std::runtime_error(
        "ace_gen_cmm_smc::post_fields: the element's first post-processing "
        "quantity is not the nodal weight 'Volume'; the nodal averaging of "
        "its post-processing output needs to be updated.");
  }

  static const char* tensorSuffixes[9] = {"xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"};

  // Prefix <P> of a name ending in 'suffix', or "" if it does not end so.
  auto prefix = [](const std::string& name, const std::string& suffix) -> std::string {
    if (name.size() <= suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
      return "";
    }
    return name.substr(0, name.size() - suffix.size());
  };

  std::vector<PostField> fields;
  std::size_t i = 1;

  while (i < names.size()) {
    std::string p = prefix(names[i], tensorSuffixes[0]);
    if (!p.empty() && i + 9 <= names.size()) {
      int c = 1;
      while (c < 9 && names[i + c] == p + tensorSuffixes[c]) {
        c++;
      }
      if (c == 9) {
        fields.push_back({p, static_cast<int>(i), 9});
        i += 9;
        continue;
      }
    }

    p = prefix(names[i], "1");
    if (!p.empty() && i + 3 <= names.size() && names[i + 1] == p + "2" && names[i + 2] == p + "3") {
      fields.push_back({p, static_cast<int>(i), 3});
      i += 3;
      continue;
    }

    fields.push_back({names[i], static_cast<int>(i), 1});
    i++;
  }

  return fields;
}

std::vector<double> build_domain_data(const ElementInfo& info,
                                       const std::map<std::string, double>& named_params) {
  std::vector<double> domainData(info.domainDataNames.size());
  const std::string block = xml_block_name(info.model);

  for (std::size_t i = 0; i < info.domainDataNames.size(); i++) {
    const auto& name = info.domainDataNames[i];
    auto it = named_params.find(name);
    if (it == named_params.end()) {
      throw std::runtime_error(
          "The 'deformation-diffusion' equation's " + block +
          " material parameters are missing required parameter '" + name +
          "'. Add a <" + name + "> ... </" + name +
          "> element under <" + block + "> in the Domain's XML.");
    }
    domainData[i] = it->second;
  }

  return domainData;
}

std::vector<double> initial_history(const ElementInfo& info) {
  if (info.historyLengthPerElement == 0) {
    return {};
  }

  const HistoryLayout layout = history_layout(info.model);

  if (info.historyLengthPerElement != layout.valuesPerGaussPoint * info.numberOfGaussPoints) {
    throw std::runtime_error(
        "ace_gen_cmm_smc::initial_history: historyLengthPerElement (" +
        std::to_string(info.historyLengthPerElement) + ") is not " +
        std::to_string(layout.valuesPerGaussPoint) + " * numberOfGaussPoints (" +
        std::to_string(info.numberOfGaussPoints) + "). The hardcoded initial "
        "history seed no longer matches this AceGen kernel's history layout "
        "and needs to be updated.");
  }

  std::vector<double> seed;
  seed.reserve(info.historyLengthPerElement);
  for (int g = 0; g < info.numberOfGaussPoints; g++) {
    seed.insert(seed.end(), layout.initial, layout.initial + layout.valuesPerGaussPoint);
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
  no_interface2();
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

  std::vector<double> domainDataCopy = element_domain_data(info.model, domainData);
  // Placeholder history (see initial_history()) -- Task 8 only overwrites
  // the geometry-dependent orientation entries; everything else passes
  // through unchanged, so this baseline must already hold the correct
  // non-orientation initial values (e.g. LambdaBarC1=1, k251=1.82758, ...).
  std::vector<double> historyPlaceholder = initial_history(info);

  return with_element(info.model, [&](auto* tag) {
    using Element = ElementOf<decltype(tag)>;
    Element elem(positionsFlat, displacements, concentrations, accelerations, rates,
                 domainDataCopy.data(), historyPlaceholder.data(), subIterationTolerance,
                 timeIncrement, time, integrationCode, elementID);
    return elem.initializeGrowthOrientationVectors();
  });
#endif
}

ElementOutput compute(const ElementInput& input) {
#ifndef SV_HAVE_INTERFACE2
  no_interface2();
#else
  RawElementInput raw = raw_element_input(input);

  return with_element(input.model, [&](auto* tag) {
    using Element = ElementOf<decltype(tag)>;
    Element elem(raw.positions, raw.displacements, raw.concentrations, raw.accelerations, raw.rates,
                 raw.domainData.data(), raw.history.data(), input.subIterationTolerance,
                 input.timeIncrement, input.time, input.integrationCode, input.elementID);

    int errorCode = elem.compute(/*computeTangent=*/true);
    if (errorCode != 0) {
      throw std::runtime_error(
          "The Interface2/AceGen " + xml_block_name(input.model) + " element failed to "
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

    out.historyUpdated.assign(historyUpdated, historyUpdated + input.history.size());

    return out;
  });
#endif
}

std::vector<double> history_with_active_stretches(const ElementInput& input) {
#ifndef SV_HAVE_INTERFACE2
  no_interface2();
#else
  RawElementInput raw = raw_element_input(input);
  const HistoryLayout layout = history_layout(input.model);

  std::vector<double> stretches = with_element(input.model, [&](auto* tag) {
    using Element = ElementOf<decltype(tag)>;
    Element elem(raw.positions, raw.displacements, raw.concentrations, raw.accelerations, raw.rates,
                 raw.domainData.data(), raw.history.data(), input.subIterationTolerance,
                 input.timeIncrement, input.time, input.integrationCode, input.elementID);
    return elem.getGaussPointStretches();
  });

  std::vector<double> history = input.history;
  const int nGP = static_cast<int>(stretches.size()) / 2;

  if (nGP == 0 || history.size() != static_cast<std::size_t>(nGP) * layout.valuesPerGaussPoint) {
    throw std::runtime_error("ace_gen_cmm_smc::history_with_active_stretches: unexpected history or stretch size.");
  }

  for (int g = 0; g < nGP; g++) {
    history[g * layout.valuesPerGaussPoint + layout.activeStretch1] = stretches[2 * g];
    history[g * layout.valuesPerGaussPoint + layout.activeStretch2] = stretches[2 * g + 1];
  }

  return history;
#endif
}

std::vector<double> history_with_growth_orientation(const ElementInput& input) {
#ifndef SV_HAVE_INTERFACE2
  no_interface2();
#else
  RawElementInput raw = raw_element_input(input);

  std::vector<double> history = with_element(input.model, [&](auto* tag) {
    using Element = ElementOf<decltype(tag)>;
    Element elem(raw.positions, raw.displacements, raw.concentrations, raw.accelerations, raw.rates,
                 raw.domainData.data(), raw.history.data(), input.subIterationTolerance,
                 input.timeIncrement, input.time, input.integrationCode, input.elementID);
    return elem.initializeGrowthOrientationVectors();
  });

  if (history.size() != input.history.size()) {
    throw std::runtime_error("ace_gen_cmm_smc::history_with_growth_orientation: unexpected history size.");
  }

  return history;
#endif
}

Array<double> post_process(const ElementInput& input) {
#ifndef SV_HAVE_INTERFACE2
  no_interface2();
#else
  RawElementInput raw = raw_element_input(input);

  return with_element(input.model, [&](auto* tag) {
    using Element = ElementOf<decltype(tag)>;
    Element elem(raw.positions, raw.displacements, raw.concentrations, raw.accelerations, raw.rates,
                 raw.domainData.data(), raw.history.data(), input.subIterationTolerance,
                 input.timeIncrement, input.time, input.integrationCode, input.elementID);

    const int numberOfPostData = elem.getNumberOfPostData();
    double** post = elem.postProcess(raw.displacements, raw.concentrations, raw.history.data(),
                                     raw.rates, raw.accelerations); // 10 x numberOfPostData

    Array<double> out(numberOfPostData, 10);
    for (int a = 0; a < 10; a++) {
      int va = kAceGenNodeToVtkNode[a];
      for (int k = 0; k < numberOfPostData; k++) {
        out(k, va) = post[a][k];
      }
    }

    return out;
  });
#endif
}

} // namespace ace_gen_cmm_smc
