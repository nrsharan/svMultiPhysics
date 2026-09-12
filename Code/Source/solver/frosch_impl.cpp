// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "frosch_impl.h"

#include <algorithm>

// FROSch's Amesos2 solver interface includes the umbrella "Amesos2.hpp", which
// does not compile against an MKL_ILP64 Trilinos build (see the Amesos2
// includes in trilinos_impl.h).
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Xpetra_CrsMatrixWrap.hpp"
#include "Xpetra_TpetraCrsMatrix.hpp"
#include "Xpetra_TpetraMap.hpp"
#include "Xpetra_TpetraMultiVector.hpp"
#include "FROSch_TwoLevelPreconditioner_def.hpp"
#include "FROSch_TpetraPreconditioner_def.hpp"

namespace frosch_impl {

Teuchos::RCP<Teuchos::ParameterList> parameters(const std::string& parameterFile, int nsd, int dof,
    bool haveCoordinates)
{
  if (!parameterFile.empty()) {
    return Teuchos::getParametersFromXmlFile(parameterFile);
  }

  auto setDirectSolver = [](Teuchos::ParameterList& list) {
    list.set("SolverType", "Amesos2");
    list.set("Solver", "klu2");
  };
  bool rotations = (nsd == dof) && haveCoordinates;

  auto params = Teuchos::rcp(new Teuchos::ParameterList("FROSch"));
  params->set("Overlap", 1);
  params->set("TwoLevel", true);
  params->set("Level Combination", "Additive");
  params->set("OverlappingOperator Type", "AlgebraicOverlappingOperator");
  params->set("CoarseOperator Type", "RGDSWCoarseOperator");

  auto& overlapping = params->sublist("AlgebraicOverlappingOperator");
  overlapping.set("Combine Values in Overlap", "Averaging");
  overlapping.set("Adding Layers Strategy", "CrsGraph");
  overlapping.set("Verbosity", "None");
  setDirectSolver(overlapping.sublist("Solver"));

  // RGDSW option 2.2 weights the interface by the inverse distances between
  // the nodes (needs the coordinates), option 1 uses constant weights.
  auto& coarse = params->sublist("RGDSWCoarseOperator");
  auto& block = coarse.sublist("Blocks").sublist("1");
  block.set("Use For Coarse Space", true);
  block.set("Option", rotations ? "2.2" : "1");
  block.set("Rotations", rotations);
  block.set("Verbosity", "None");
  setDirectSolver(coarse.sublist("ExtensionSolver"));
  auto& distribution = coarse.sublist("Distribution");
  distribution.set("Type", "linear");
  distribution.set("GatheringSteps", 1);
  distribution.set("NumProcs", 1);
  setDirectSolver(coarse.sublist("CoarseSolver"));

  return params;
}

Teuchos::RCP<Tpetra::Operator<SC,LO,GO,NO>> create_preconditioner(
    const Teuchos::RCP<Tpetra::CrsMatrix<SC,LO,GO,NO>>& K,
    const Teuchos::RCP<const Tpetra::Map<LO,GO,NO>>& repeatedMap,
    const Teuchos::RCP<const Tpetra::MultiVector<SC,LO,GO,NO>>& nodeCoords,
    int nsd, int dof, std::vector<GO> dirichletDofs, const std::string& parameterFile)
{
  using XCrsMatrix = Xpetra::TpetraCrsMatrix<SC,LO,GO,NO>;
  using XCrsMatrixWrap = Xpetra::CrsMatrixWrap<SC,LO,GO,NO>;
  using XMatrix = Xpetra::Matrix<SC,LO,GO,NO>;
  using XMultiVector = Xpetra::MultiVector<SC,LO,GO,NO>;

  auto params = parameters(parameterFile, nsd, dof, !nodeCoords.is_null());

  Teuchos::RCP<Xpetra::CrsMatrix<SC,LO,GO,NO>> xK = Teuchos::rcp(new XCrsMatrix(K));
  Teuchos::RCP<const XMatrix> A = Teuchos::rcp(new XCrsMatrixWrap(xK));

  // The const overload of Xpetra::toXpetra for multivectors does not compile
  // (Trilinos 16.1); the coordinates are only read.
  Teuchos::RCP<const XMultiVector> nodeList = Teuchos::null;
  if (!nodeCoords.is_null()) {
    nodeList = Xpetra::toXpetra(Teuchos::rcp_const_cast<Tpetra::MultiVector<SC,LO,GO,NO>>(nodeCoords));
  }

  // FROSch looks the Dirichlet dofs up by binary search.
  std::sort(dirichletDofs.begin(), dirichletDofs.end());
  dirichletDofs.erase(std::unique(dirichletDofs.begin(), dirichletDofs.end()), dirichletDofs.end());
  Teuchos::ArrayRCP<GO> dirichletBoundaryDofs;
  if (!dirichletDofs.empty()) {
    dirichletBoundaryDofs = Teuchos::arcp<GO>(dirichletDofs.size());
    std::copy(dirichletDofs.begin(), dirichletDofs.end(), dirichletBoundaryDofs.begin());
  }

  auto prec = Teuchos::rcp(new FROSch::TwoLevelPreconditioner<SC,LO,GO,NO>(A, params));
  prec->initialize(nsd, dof, params->get("Overlap", 1), Teuchos::null, nodeList, FROSch::NodeWise,
      Xpetra::toXpetra(repeatedMap), Teuchos::null, dirichletBoundaryDofs);
  prec->compute();

  return Teuchos::rcp(new FROSch::TpetraPreconditioner<SC,LO,GO,NO>(prec));
}

} // namespace frosch_impl
