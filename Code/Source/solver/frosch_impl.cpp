// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "frosch_impl.h"

#include <algorithm>
#include <stdexcept>

// FROSch's Amesos2 solver interface includes the umbrella "Amesos2.hpp", which
// does not compile against an MKL_ILP64 Trilinos build (see the Amesos2
// includes in trilinos_impl.h).
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Xpetra_CrsMatrixWrap.hpp"
#include "Xpetra_TpetraCrsMatrix.hpp"
#include "Xpetra_TpetraMap.hpp"
#include "Xpetra_TpetraMultiVector.hpp"
#include "FROSch_TwoLevelPreconditioner_def.hpp"
#include "FROSch_TwoLevelBlockPreconditioner_def.hpp"
#include "FROSch_TpetraPreconditioner_def.hpp"

namespace frosch_impl {

namespace {

using Tpetra_Map = Tpetra::Map<LO,GO,NO>;
using Tpetra_CrsMatrix = Tpetra::CrsMatrix<SC,LO,GO,NO>;
using Tpetra_MultiVector = Tpetra::MultiVector<SC,LO,GO,NO>;
using Tpetra_Operator = Tpetra::Operator<SC,LO,GO,NO>;
using XMap = Xpetra::Map<LO,GO,NO>;
using XMatrix = Xpetra::Matrix<SC,LO,GO,NO>;
using XMultiVector = Xpetra::MultiVector<SC,LO,GO,NO>;
using SinglePreconditioner = FROSch::TwoLevelPreconditioner<SC,LO,GO,NO>;
using BlockPreconditioner = FROSch::TwoLevelBlockPreconditioner<SC,LO,GO,NO>;

const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

/// Xpetra matrix wrapping a Tpetra matrix.
Teuchos::RCP<const XMatrix> xpetra_matrix(const Teuchos::RCP<Tpetra_CrsMatrix>& K)
{
  Teuchos::RCP<Xpetra::CrsMatrix<SC,LO,GO,NO>> xK = Teuchos::rcp(new Xpetra::TpetraCrsMatrix<SC,LO,GO,NO>(K));
  return Teuchos::rcp(new Xpetra::CrsMatrixWrap<SC,LO,GO,NO>(xK));
}

/// Xpetra multivector wrapping the coordinates, or null. The const overload of
/// Xpetra::toXpetra for multivectors does not compile (Trilinos 16.1); the
/// coordinates are only read.
Teuchos::RCP<const XMultiVector> xpetra_coordinates(const Teuchos::RCP<const Tpetra_MultiVector>& nodeCoords)
{
  if (nodeCoords.is_null()) {
    return Teuchos::null;
  }
  return Xpetra::toXpetra(Teuchos::rcp_const_cast<Tpetra_MultiVector>(nodeCoords));
}

/// Sorted GIDs as a FROSch dof list. FROSch looks the Dirichlet dofs up by
/// binary search. The list must not be null on any process, even without
/// Dirichlet dofs: for a null list FROSch determines the Dirichlet rows itself,
/// which communicates, so a process whose subdomain has no Dirichlet boundary
/// would enter a communication the others skip (MPI_Waitall errors or a hang,
/// e.g. on 32 processes).
Teuchos::ArrayRCP<GO> dof_list(std::vector<GO> gids)
{
  std::sort(gids.begin(), gids.end());
  gids.erase(std::unique(gids.begin(), gids.end()), gids.end());
  Teuchos::ArrayRCP<GO> list(new GO[std::max<std::size_t>(gids.size(), 1)], 0, gids.size(), true);
  std::copy(gids.begin(), gids.end(), list.begin());
  return list;
}

/// GIDs of the elements of the process in a map, in local order.
std::vector<GO> global_elements(const Tpetra_Map& map)
{
  std::vector<GO> gids(map.getLocalNumElements());
  for (std::size_t i = 0; i < gids.size(); i++) {
    gids[i] = map.getGlobalElement(static_cast<LO>(i));
  }
  return gids;
}

/// Block-contiguous dof numbers for FROSch's block preconditioner: the dof d
/// of node n (GID n*dof+d) becomes n*nsd+d in the first block (d < nsd) and
/// nsd*numNodes + n*(dof-nsd) + (d-nsd) in the second, which is how
/// FROSch::BuildDofMapsVec offsets the second block.
struct BlockNumbering {
  GO numNodes;
  int nsd;
  int dof;

  GO operator()(GO gid) const
  {
    const GO n = gid / dof;
    const int d = static_cast<int>(gid % dof);
    return (d < nsd) ? n * nsd + d : nsd * numNodes + n * (dof - nsd) + (d - nsd);
  }
};

/// The block numbering of the dofs of K, which must be numbered
/// 0 .. (number of nodes)*dof - 1.
BlockNumbering block_numbering(const Tpetra_CrsMatrix& K, int nsd, int dof)
{
  if (dof <= nsd) {
    throw std::runtime_error("the trilinos-frosch-block preconditioner needs more dofs per node (" +
        std::to_string(dof) + ") than space dimensions (" + std::to_string(nsd) + ").");
  }

  const auto rowMap = K.getRowMap();
  const GO numGlobalRows = static_cast<GO>(K.getGlobalNumRows());
  if (numGlobalRows % dof != 0 || rowMap->getMinAllGlobalIndex() != 0 ||
      rowMap->getMaxAllGlobalIndex() != numGlobalRows - 1) {
    throw std::runtime_error("trilinos-frosch-block: the dofs are not numbered 0 .. (number of nodes)*dof - 1.");
  }
  return BlockNumbering{numGlobalRows / dof, nsd, dof};
}

/// The row map of K in the block numbering: the owned dofs in the same local order.
Teuchos::RCP<const Tpetra_Map> block_map(const Tpetra_CrsMatrix& K, const BlockNumbering& number)
{
  const auto rowMap = K.getRowMap();
  Teuchos::Array<GO> blockRows(rowMap->getLocalNumElements());
  for (std::size_t i = 0; i < blockRows.size(); i++) {
    blockRows[i] = number(rowMap->getGlobalElement(static_cast<LO>(i)));
  }
  return Teuchos::rcp(new Tpetra_Map(INVALID, blockRows(), 0, rowMap->getComm()));
}

/// K in the block numbering (row map blockMap): every row stays on its process.
Teuchos::RCP<Tpetra_CrsMatrix> block_matrix(const Tpetra_CrsMatrix& K, const Teuchos::RCP<const Tpetra_Map>& blockMap,
    const BlockNumbering& number)
{
  const auto colMap = K.getColMap();
  auto Kb = Teuchos::rcp(new Tpetra_CrsMatrix(blockMap, K.getLocalMaxNumRowEntries()));
  Teuchos::Array<GO> columns;
  for (std::size_t i = 0; i < blockMap->getLocalNumElements(); i++) {
    typename Tpetra_CrsMatrix::local_inds_host_view_type localColumns;
    typename Tpetra_CrsMatrix::values_host_view_type values;
    K.getLocalRowView(static_cast<LO>(i), localColumns, values);
    columns.resize(localColumns.extent(0));
    for (std::size_t j = 0; j < localColumns.extent(0); j++) {
      columns[j] = number(colMap->getGlobalElement(localColumns(j)));
    }
    Kb->insertGlobalValues(blockMap->getGlobalElement(static_cast<LO>(i)), columns(),
        Teuchos::ArrayView<const SC>(values.data(), values.extent(0)));
  }
  Kb->fillComplete(blockMap, blockMap);
  return Kb;
}

/// Replaces the values of matrix, the matrix FROSch was set up with, by those
/// of K, which has the same rows and column map as the matrix it was set up
/// from. columns maps the local columns of K to those of matrix (empty: the
/// same). Returns false if an entry of K is not in matrix.
bool replace_values(const Tpetra_CrsMatrix& K, Tpetra_CrsMatrix& matrix, const std::vector<LO>& columns)
{
  const auto domainMap = matrix.getDomainMap();
  const auto rangeMap = matrix.getRangeMap();
  matrix.resumeFill();
  bool found = true;
  Teuchos::Array<LO> matrixColumns;
  for (std::size_t i = 0; i < K.getLocalNumRows(); i++) {
    typename Tpetra_CrsMatrix::local_inds_host_view_type localColumns;
    typename Tpetra_CrsMatrix::values_host_view_type values;
    K.getLocalRowView(static_cast<LO>(i), localColumns, values);
    const std::size_t numEntries = localColumns.extent(0);
    matrixColumns.resize(numEntries);
    for (std::size_t j = 0; j < numEntries; j++) {
      matrixColumns[j] = columns.empty() ? localColumns(j) : columns[localColumns(j)];
    }
    const LO numReplaced = matrix.replaceLocalValues(static_cast<LO>(i), matrixColumns(),
        Teuchos::ArrayView<const SC>(values.data(), numEntries));
    found = found && (numReplaced == static_cast<LO>(numEntries));
  }
  matrix.fillComplete(domainMap, rangeMap);
  return found;
}

/// Applies an operator defined on the block-renumbered dofs to vectors in
/// svMultiPhysics's numbering. The renumbered map lists the process's dofs in
/// the same local order, so the vectors are renumbered by relabeling copies.
class BlockRenumberedOperator : public Tpetra_Operator {
  public:
    BlockRenumberedOperator(const Teuchos::RCP<const Tpetra_Map>& map, const Teuchos::RCP<const Tpetra_Map>& blockMap,
                            const Teuchos::RCP<Tpetra_Operator>& blockOperator)
      : map_(map), blockMap_(blockMap), blockOperator_(blockOperator) {}

    Teuchos::RCP<const Tpetra_Map> getDomainMap() const override { return map_; }
    Teuchos::RCP<const Tpetra_Map> getRangeMap() const override { return map_; }

    void apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               SC alpha = Teuchos::ScalarTraits<SC>::one(), SC beta = Teuchos::ScalarTraits<SC>::zero()) const override
    {
      Tpetra_MultiVector Xb(X, Teuchos::Copy);
      Tpetra_MultiVector Yb(Y, Teuchos::Copy);
      Xb.replaceMap(blockMap_);
      Yb.replaceMap(blockMap_);
      blockOperator_->apply(Xb, Yb, mode, alpha, beta);
      Yb.replaceMap(Y.getMap());
      Tpetra::deep_copy(Y, Yb);
    }

  private:
    Teuchos::RCP<const Tpetra_Map> map_;
    Teuchos::RCP<const Tpetra_Map> blockMap_;
    Teuchos::RCP<Tpetra_Operator> blockOperator_;
};

/// Sets up (initializes and computes) the single-block preconditioner.
Teuchos::RCP<SinglePreconditioner> setup_single(const Teuchos::RCP<Tpetra_CrsMatrix>& K,
    const Teuchos::RCP<const Tpetra_Map>& repeatedMap, const Teuchos::RCP<const Tpetra_MultiVector>& nodeCoords,
    int nsd, int dof, const std::vector<GO>& dirichletDofs, const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  auto prec = Teuchos::rcp(new SinglePreconditioner(xpetra_matrix(K), params));
  prec->initialize(nsd, dof, params->get("Overlap", 1), Teuchos::null, xpetra_coordinates(nodeCoords),
      FROSch::NodeWise, Xpetra::toXpetra(repeatedMap), Teuchos::null, dof_list(dirichletDofs));
  prec->compute();
  return prec;
}

/// Sets up (initializes and computes) the two-block preconditioner of Kb, the
/// block-renumbered matrix: the first nsd dofs of every node and the others.
Teuchos::RCP<BlockPreconditioner> setup_block(const Teuchos::RCP<Tpetra_CrsMatrix>& Kb,
    const Teuchos::RCP<const Tpetra_Map>& repeatedMap, const Teuchos::RCP<const Tpetra_MultiVector>& nodeCoords,
    int nsd, int dof, const std::vector<GO>& dirichletDofs, const BlockNumbering& number,
    const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  const auto comm = Kb->getRowMap()->getComm();

  // Repeated maps of the blocks, numbered from 0 within each block (FROSch adds
  // the offset of the second block itself), and the Dirichlet dofs of each
  // block in the renumbered (offset) numbering.
  const std::size_t numRepeated = repeatedMap->getLocalNumElements();
  Teuchos::Array<GO> repeated0, repeated1;
  for (std::size_t i = 0; i < numRepeated; i += dof) {
    const GO n = repeatedMap->getGlobalElement(static_cast<LO>(i)) / dof;
    for (int d = 0; d < nsd; d++) {
      repeated0.push_back(n * nsd + d);
    }
    for (int d = nsd; d < dof; d++) {
      repeated1.push_back(n * (dof - nsd) + (d - nsd));
    }
  }
  std::vector<GO> dirichlet0, dirichlet1;
  for (GO gid : dirichletDofs) {
    ((gid % dof) < nsd ? dirichlet0 : dirichlet1).push_back(number(gid));
  }

  Teuchos::ArrayRCP<Teuchos::RCP<const XMap>> repeatedMaps(2);
  repeatedMaps[0] = Xpetra::toXpetra(Teuchos::RCP<const Tpetra_Map>(new Tpetra_Map(INVALID, repeated0(), 0, comm)));
  repeatedMaps[1] = Xpetra::toXpetra(Teuchos::RCP<const Tpetra_Map>(new Tpetra_Map(INVALID, repeated1(), 0, comm)));

  Teuchos::ArrayRCP<unsigned> dofsPerNode(2);
  dofsPerNode[0] = nsd;
  dofsPerNode[1] = dof - nsd;
  Teuchos::ArrayRCP<FROSch::DofOrdering> dofOrdering(2, FROSch::NodeWise);

  // Both blocks live on the same nodes.
  Teuchos::ArrayRCP<Teuchos::RCP<const XMultiVector>> nodeLists = Teuchos::null;
  if (!nodeCoords.is_null()) {
    nodeLists = Teuchos::ArrayRCP<Teuchos::RCP<const XMultiVector>>(2, xpetra_coordinates(nodeCoords));
  }

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>> dirichletLists(2);
  dirichletLists[0] = dof_list(dirichlet0);
  dirichletLists[1] = dof_list(dirichlet1);

  Teuchos::ArrayRCP<Teuchos::RCP<const XMultiVector>> nullSpaces = Teuchos::null;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<const XMap>>> dofsMaps = Teuchos::null;

  auto prec = Teuchos::rcp(new BlockPreconditioner(xpetra_matrix(Kb), params));
  prec->initialize(nsd, dofsPerNode, dofOrdering, params->get("Overlap", 1), repeatedMaps, nullSpaces, nodeLists,
      dofsMaps, dirichletLists);
  prec->compute();
  return prec;
}

/// The default parameters (see parameters()).
Teuchos::RCP<Teuchos::ParameterList> default_parameters(int nsd, int dof, bool haveCoordinates, bool block)
{
  auto setDirectSolver = [](Teuchos::ParameterList& list) {
    list.set("SolverType", "Amesos2");
    list.set("Solver", "klu2");
  };
  // RGDSW option 2.2 weights the interface by the inverse distances between
  // the nodes (needs the coordinates), option 1 uses constant weights.
  auto setCoarseBlock = [](Teuchos::ParameterList& list, bool rotations) {
    list.set("Use For Coarse Space", true);
    list.set("Option", rotations ? "2.2" : "1");
    list.set("Rotations", rotations);
    list.set("Verbosity", "None");
  };

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

  auto& coarse = params->sublist("RGDSWCoarseOperator");
  auto& blocks = coarse.sublist("Blocks");
  if (block) {
    setCoarseBlock(blocks.sublist("1"), haveCoordinates);
    setCoarseBlock(blocks.sublist("2"), false);
  } else {
    setCoarseBlock(blocks.sublist("1"), (nsd == dof) && haveCoordinates);
  }
  setDirectSolver(coarse.sublist("ExtensionSolver"));
  auto& distribution = coarse.sublist("Distribution");
  distribution.set("Type", "linear");
  distribution.set("GatheringSteps", 1);
  distribution.set("NumProcs", 1);
  setDirectSolver(coarse.sublist("CoarseSolver"));

  return params;
}

} // namespace

Teuchos::RCP<Teuchos::ParameterList> parameters(const std::string& parameterFile, int nsd, int dof,
    bool haveCoordinates, bool block)
{
  Teuchos::RCP<Teuchos::ParameterList> params = parameterFile.empty() ?
      default_parameters(nsd, dof, haveCoordinates, block) : Teuchos::getParametersFromXmlFile(parameterFile);

  // When the preconditioner is recomputed for a new matrix, FROSch keeps the
  // coarse basis of the first matrix by default. Recompute it, and the
  // symbolic factorization of the coarse matrix (the sparsity of the basis
  // after dropping, and so of the coarse matrix, can change).
  if (params->isType<std::string>("CoarseOperator Type")) {
    auto& coarse = params->sublist(params->get<std::string>("CoarseOperator Type"));
    coarse.get("Reuse: Coarse Basis", false);
    coarse.get("Reuse: Coarse Matrix Symbolic Factorization", false);
  }
  return params;
}

/// What the preconditioner was set up for, and the FROSch objects.
struct Preconditioner::State {
  bool block = false;
  int nsd = 0;
  int dof = 0;
  std::string parameterFile;
  std::vector<GO> rows;       ///< GIDs of the owned dofs (rows of K)
  std::vector<GO> columns;    ///< GIDs of the column map of K
  std::vector<GO> repeated;   ///< GIDs of the repeated map
  std::vector<GO> dirichlet;  ///< sorted GIDs of the Dirichlet dofs

  /// The matrix FROSch was set up with (K, or its block-renumbered copy). It is
  /// kept, with the values of each new matrix: svMultiPhysics creates new maps
  /// for every matrix, and FROSch's setup refers to the maps it was set up with
  /// (e.g. a kept coarse basis, "Reuse: Coarse Basis").
  Teuchos::RCP<Tpetra_CrsMatrix> matrix;
  std::vector<LO> blockColumns;                                    ///< block: local column of matrix for each of K
  Teuchos::RCP<const Tpetra_Map> blockMap;                         ///< block: row map of the renumbered K
  Teuchos::RCP<FROSch::OneLevelPreconditioner<SC,LO,GO,NO>> frosch;
  Teuchos::RCP<Tpetra_Operator> op;                                ///< the operator applied by Belos
};

Preconditioner::Preconditioner() = default;
Preconditioner::~Preconditioner() = default;

Teuchos::RCP<Tpetra::Operator<SC,LO,GO,NO>> Preconditioner::update(
    const Teuchos::RCP<Tpetra::CrsMatrix<SC,LO,GO,NO>>& K,
    const Teuchos::RCP<const Tpetra::Map<LO,GO,NO>>& repeatedMap,
    const Teuchos::RCP<const Tpetra::MultiVector<SC,LO,GO,NO>>& nodeCoords,
    int nsd, int dof, std::vector<GO> dirichletDofs, const std::string& parameterFile, bool block)
{
  std::sort(dirichletDofs.begin(), dirichletDofs.end());
  dirichletDofs.erase(std::unique(dirichletDofs.begin(), dirichletDofs.end()), dirichletDofs.end());
  std::vector<GO> rows = global_elements(*K->getRowMap());
  std::vector<GO> columns = global_elements(*K->getColMap());
  std::vector<GO> repeated = global_elements(*repeatedMap);

  // Only recompute if every process has the same dofs and matrix entries as
  // before: FROSch's matrix then gets the values of K.
  int same = (state_ && state_->block == block && state_->nsd == nsd && state_->dof == dof &&
      state_->parameterFile == parameterFile && state_->rows == rows && state_->columns == columns &&
      state_->repeated == repeated && state_->dirichlet == dirichletDofs) ? 1 : 0;
  if (same) {
    same = replace_values(*K, *state_->matrix, state_->blockColumns) ? 1 : 0;
  }
  int allSame = 0;
  Teuchos::reduceAll(*K->getRowMap()->getComm(), Teuchos::REDUCE_MIN, same, Teuchos::outArg(allSame));

  if (allSame) {
    Teuchos::RCP<const XMatrix> xK = xpetra_matrix(state_->matrix);
    state_->frosch->resetMatrix(xK);
    state_->frosch->compute();
    return state_->op;
  }

  state_.reset();
  auto state = std::make_unique<State>();
  state->block = block;
  state->nsd = nsd;
  state->dof = dof;
  state->parameterFile = parameterFile;
  state->rows = std::move(rows);
  state->columns = std::move(columns);
  state->repeated = std::move(repeated);
  state->dirichlet = std::move(dirichletDofs);

  auto params = parameters(parameterFile, nsd, dof, !nodeCoords.is_null(), block);
  if (block) {
    const BlockNumbering number = block_numbering(*K, nsd, dof);
    state->blockMap = block_map(*K, number);
    state->matrix = block_matrix(*K, state->blockMap, number);
    const auto colMap = K->getColMap();
    const auto blockColMap = state->matrix->getColMap();
    state->blockColumns.resize(colMap->getLocalNumElements());
    for (std::size_t j = 0; j < state->blockColumns.size(); j++) {
      state->blockColumns[j] = blockColMap->getLocalElement(number(colMap->getGlobalElement(static_cast<LO>(j))));
    }
    auto prec = setup_block(state->matrix, repeatedMap, nodeCoords, nsd, dof, state->dirichlet, number, params);
    state->frosch = prec;
    Teuchos::RCP<Tpetra_Operator> blockPrec = Teuchos::rcp(new FROSch::TpetraPreconditioner<SC,LO,GO,NO>(prec));
    state->op = Teuchos::rcp(new BlockRenumberedOperator(K->getDomainMap(), state->blockMap, blockPrec));
  } else {
    state->matrix = K;
    auto prec = setup_single(K, repeatedMap, nodeCoords, nsd, dof, state->dirichlet, params);
    state->frosch = prec;
    state->op = Teuchos::rcp(new FROSch::TpetraPreconditioner<SC,LO,GO,NO>(prec));
  }
  state_ = std::move(state);
  return state_->op;
}

} // namespace frosch_impl
