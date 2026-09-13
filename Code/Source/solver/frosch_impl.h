// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FROSCH_IMPL_H
#define FROSCH_IMPL_H
/*!
  \file    frosch_impl.h
  \brief   FROSch (Trilinos ShyLU_DDFROSch) two-level overlapping Schwarz
           preconditioners used by the trilinos-frosch and
           trilinos-frosch-block preconditioners

  FROSch itself is only included in frosch_impl.cpp: its headers use
  'using namespace Xpetra' and 'using namespace Teuchos', so unqualified names
  such as Matrix and Array clash with the svMultiPhysics types in any file
  that includes the svMultiPhysics headers. This header only uses Tpetra.
*/

#include <memory>
#include <string>
#include <vector>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

namespace frosch_impl {

using SC = double;
using LO = int;
using GO = int;
using NO = Tpetra::Map<>::node_type;

/// @brief FROSch parameter list: the file parameterFile (a Teuchos XML
/// parameter list for FROSch::TwoLevelPreconditioner, or for
/// FROSch::TwoLevelBlockPreconditioner if block is true) if it is given,
/// otherwise these settings, with KLU2 as the direct solver:
///  - one layer of algebraic overlap, subdomain problems solved exactly;
///  - an RGDSW coarse space, solved on one process, with a translation per
///    dof, plus the rotations of the displacement dofs if the coordinates
///    are given: for block, those of the first block (the first nsd dofs of
///    every node); otherwise when all dofs are displacements (dof == nsd).
///
/// Unless the file sets them, "Reuse: Coarse Basis" and "Reuse: Coarse Matrix
/// Symbolic Factorization" are false, so a preconditioner recomputed for a new
/// matrix (see Preconditioner) is the same as a new one. FROSch's default
/// (true) keeps the coarse basis of the first matrix. "Reuse: Symbolic
/// Factorization" of the overlapping operator is false as well: FROSch's
/// update of the subdomain matrices for a reused factorization is slower than
/// a new factorization.
Teuchos::RCP<Teuchos::ParameterList> parameters(const std::string& parameterFile, int nsd, int dof,
    bool haveCoordinates, bool block);

/// @brief FROSch two-level overlapping Schwarz preconditioner, kept between
/// linear solves.
///
/// Unlike one-level (block Jacobi, additive Schwarz ILU) preconditioners, the
/// coarse space couples all subdomains, so the number of Krylov iterations
/// grows much less with the number of processes.
///
/// With block, the dofs of every node form two blocks with their own coarse
/// space: the first nsd dofs (displacements, with the rotations in their
/// coarse space) and the others (e.g. a concentration). FROSch's block
/// preconditioner needs block-contiguous dof numbers, so it is applied to a
/// renumbered copy of K and wrapped in an operator that renumbers the vectors
/// (all renumbered dofs stay on their process).
///
/// FROSch sets up in two phases: initialize() builds the overlapping
/// subdomains and the interface of the coarse space from the sparsity
/// pattern, the maps, the coordinates and the Dirichlet dofs; compute()
/// factorizes the subdomain matrices and builds the coarse basis and the
/// coarse matrix. The first update() does both. A later update() for the same
/// dofs, repeated map and Dirichlet dofs on every process (e.g. in the next
/// Newton iteration) only calls compute() for the new matrix, as the
/// "Recycling" option of FROSch's Stratimikos adapter does.
class Preconditioner {
  public:
    Preconditioner();
    ~Preconditioner();

    /// @brief Set up or recompute the preconditioner for K (collective).
    ///
    /// \param K              assembled matrix (row map: owned dofs, numbered node GID * dof + d)
    /// \param repeatedMap    owned and ghost dofs of the process, numbered node-wise (node GID * dof + d)
    /// \param nodeCoords     coordinates of the owned and ghost nodes (nsd columns), or null
    /// \param nsd            spatial dimension
    /// \param dof            dofs per node
    /// \param dirichletDofs  GIDs of the Dirichlet dofs (removed from the interface of the coarse space)
    /// \param parameterFile  FROSch parameter list in Teuchos XML format, or empty for the defaults
    /// \param block          two blocks: the first nsd dofs of every node and the others (needs dof > nsd)
    /// \return the preconditioner as an operator for Belos (the same object when it is recomputed)
    Teuchos::RCP<Tpetra::Operator<SC,LO,GO,NO>> update(
        const Teuchos::RCP<Tpetra::CrsMatrix<SC,LO,GO,NO>>& K,
        const Teuchos::RCP<const Tpetra::Map<LO,GO,NO>>& repeatedMap,
        const Teuchos::RCP<const Tpetra::MultiVector<SC,LO,GO,NO>>& nodeCoords,
        int nsd, int dof, std::vector<GO> dirichletDofs, const std::string& parameterFile,
        bool block = false);

  private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace frosch_impl

#endif
