// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "FsilsLinearAlgebra.h"
#include "fsils_api.hpp"
#include "lhsa.h"
#include <cstdlib>
#include <fstream>
#include <iostream>

/////////////////////////////////////////////////////////////////
//             F s i l s L i n e a r A l g e b r a             //
/////////////////////////////////////////////////////////////////
// The following methods implement the FSILS LinearAlgebra interface.

std::set<consts::LinearAlgebraType> FsilsLinearAlgebra::valid_assemblers = {
  consts::LinearAlgebraType::none,
  consts::LinearAlgebraType::fsils,
};

FsilsLinearAlgebra::FsilsLinearAlgebra()
{
  interface_type = consts::LinearAlgebraType::fsils; 
  assembly_type = consts::LinearAlgebraType::fsils; 
  preconditioner_type = consts::PreconditionerType::PREC_FSILS;
}

/// @brief Allocate data arrays.
void FsilsLinearAlgebra::alloc(ComMod& com_mod, eqType& lEq)
{
  #define n_debug_alloc
  #ifdef debug_alloc
  std::cout << "[FsilsLinearAlgebra::alloc] ---------- alloc ---------- " << std::endl;
  #endif
  int dof = com_mod.dof;

  com_mod.Val.resize(dof*dof, com_mod.lhs.nnz);
}

/// @brief Assemble local element arrays.
void FsilsLinearAlgebra::assemble(ComMod& com_mod, const int num_elem_nodes, const Vector<int>& eqN,
        const Array3<double>& lK, const Array<double>& lR)
{
  #define n_debug_assemble
  #ifdef debug_assemble
  std::cout << "[FsilsLinearAlgebra::assemble] ---------- assemble ---------- " << std::endl;
  std::cout << "[FsilsLinearAlgebra::assemble] num_elem_nodes: " << num_elem_nodes << std::endl;
  std::cout << "[FsilsLinearAlgebra::assemble] eqN.size(): " << eqN.size() << std::endl;
  std::cout << "[FsilsLinearAlgebra::assemble] lK.size(): " << lK.size() << std::endl;
  std::cout << "[FsilsLinearAlgebra::assemble] lR.size(): " << lR.size() << std::endl;
  #endif

  lhsa_ns::do_assem(com_mod, num_elem_nodes, eqN, lK, lR);
}

/// @brief Check the validity of the preconditioner and assembly types options. 
void FsilsLinearAlgebra::check_options(const consts::PreconditionerType prec_cond_type, 
  const consts::LinearAlgebraType assembly_type)
{
  using namespace consts;
  auto prec_cond_type_name = consts::preconditioner_type_to_name.at(prec_cond_type);
  auto assembly_type_name = LinearAlgebra::type_to_name.at(assembly_type);
  std::string error_msg;

  if (valid_assemblers.count(assembly_type) == 0) {
    error_msg = "fsils linear algebra can't use '" + assembly_type_name + "' for assembly.";
  }

  if (fsils_preconditioners.count(prec_cond_type) == 0) { 
    error_msg = "fsils linear algebra can't use '" + prec_cond_type_name + "' for a preconditioner.";
  }

  if (error_msg != "") { 
    throw std::runtime_error("[svMultiPhysics] ERROR: " + error_msg);
  }
}

/// @brief Initialize framework.
void FsilsLinearAlgebra::initialize(ComMod& com_mod, eqType& lEq)
{
  // Nothing is needed to initialize FSILS.
}

/// @brief Finalize framework.
void FsilsLinearAlgebra::finalize()
{
  // Nothing is needed to finalize FSILS.
}

/// @brief Set the linear algebra package for assmbly.
void FsilsLinearAlgebra::set_assembly(consts::LinearAlgebraType atype)
{
  if (atype == consts::LinearAlgebraType::none) {
    return;
  }

  if (valid_assemblers.count(atype) == 0) {
    auto str_type = LinearAlgebra::type_to_name.at(atype);
    throw std::runtime_error("[FsilsLinearAlgebra] ERROR: Can't set fsils linear algebra to use '" +
      str_type + "' for assembly.");
  }

  assembly_type = atype;
}

/// @brief Set the preconditioner.
void FsilsLinearAlgebra::set_preconditioner(consts::PreconditionerType prec_type)
{
  if (consts::fsils_preconditioners.count(prec_type) == 0) {
    auto prec_cond_type_name = consts::preconditioner_type_to_name.at(prec_type);
    throw std::runtime_error("[FsilsLinearAlgebra] ERROR: fsils linear algebra can't use '" + 
        prec_cond_type_name + "' for a preconditioner.");
    return;
  }

  preconditioner_type = prec_type;
}

/// @brief Solve a system of linear equations.
void FsilsLinearAlgebra::solve(ComMod& com_mod, eqType& lEq, const Vector<int>& incL, const Vector<double>& res)
{
  auto& lhs = com_mod.lhs;
  int dof = com_mod.dof;
  auto& R = com_mod.R;
  auto& Val = com_mod.Val;
  auto preconditioner = lEq.linear_algebra_preconditioner;

  // TEMPORARY DIAGNOSTIC (see /tmp or $DUMP_LINEAR_SYSTEM_DIR): dump the
  // raw assembled system (before FSILS's own preconditioning/Krylov
  // solve) for a specific call, so it can be solved exactly (e.g. via
  // scipy) outside svMultiPhysics entirely, to check whether an exact
  // solve gives a much better Newton step than GMRES's inexact one at
  // the same iteration -- i.e. whether the linear solver itself, not the
  // (already FD-verified exact) element tangent, is limiting Newton's
  // convergence rate. Controlled by DUMP_LINEAR_SYSTEM_CALL (0-based call
  // index to dump) and DUMP_LINEAR_SYSTEM_DIR (output directory).
  if (const char* dumpCallEnv = std::getenv("DUMP_LINEAR_SYSTEM_CALL")) {
    static int callCount = 0;
    int dumpCall = std::atoi(dumpCallEnv);
    if (callCount == dumpCall) {
      std::string dir = std::getenv("DUMP_LINEAR_SYSTEM_DIR") ? std::getenv("DUMP_LINEAR_SYSTEM_DIR") : "/tmp";
      std::string base = dir + "/linsys_call" + std::to_string(callCount);
      std::ofstream fmeta(base + "_meta.txt");
      fmeta << "nNo " << lhs.nNo << "\n";
      fmeta << "nnz " << lhs.nnz << "\n";
      fmeta << "dof " << dof << "\n";
      fmeta.close();

      std::ofstream frowptr(base + "_rowptr.txt");
      for (int a = 0; a < lhs.nNo; a++) frowptr << lhs.rowPtr(0,a) << " " << lhs.rowPtr(1,a) << "\n";
      frowptr.close();

      std::ofstream fcolptr(base + "_colptr.txt");
      for (int k = 0; k < lhs.nnz; k++) fcolptr << lhs.colPtr(k) << "\n";
      fcolptr.close();

      std::ofstream fmap(base + "_map.txt");
      for (int a = 0; a < lhs.nNo; a++) fmap << lhs.map(a) << "\n";
      fmap.close();

      std::ofstream fval(base + "_val.txt");
      fval.precision(17);
      for (int idx = 0; idx < dof*dof; idx++) {
        for (int k = 0; k < lhs.nnz; k++) fval << Val(idx,k) << " ";
        fval << "\n";
      }
      fval.close();

      std::ofstream fr(base + "_r.txt");
      fr.precision(17);
      for (int i = 0; i < dof; i++) {
        for (int a = 0; a < lhs.nNo; a++) fr << R(i,a) << " ";
        fr << "\n";
      }
      fr.close();

      std::cerr << "[DUMP_LINEAR_SYSTEM] wrote " << base << "_*.txt (call " << callCount << ")\n";
    }
    callCount++;
  }

  // TEMPORARY DIAGNOSTIC: hot-swap GMRES's solution for this one call with
  // an externally-precomputed EXACT solve of the exact same raw system
  // (dumped above, solved via scipy in a separate script -- see
  // solve_exact.py), then skip the real fsils_solve() call entirely. This
  // lets svMultiPhysics's own (unmodified, correct) corrector/reassembly/
  // convergence-reporting pipeline show the *actual* resulting nonlinear
  // residual on the very next reported iteration, using an exact linear
  // step instead of GMRES's inexact one -- without reimplementing any
  // assembly, BC, or follower-load code. Controlled by
  // OVERRIDE_SOLUTION_CALL (0-based call index to override) and
  // OVERRIDE_SOLUTION_FILE (path to a dof x nNo text file, com_mod node
  // ordering, matching the format DUMP_LINEAR_SYSTEM_CALL's "_r.txt"
  // writes).
  if (const char* overrideCallEnv = std::getenv("OVERRIDE_SOLUTION_CALL")) {
    static int overrideCallCount = 0;
    int overrideCall = std::atoi(overrideCallEnv);
    if (overrideCallCount == overrideCall) {
      const char* file = std::getenv("OVERRIDE_SOLUTION_FILE");
      if (!file) {
        throw std::runtime_error("[OVERRIDE_SOLUTION] OVERRIDE_SOLUTION_FILE not set");
      }
      std::ifstream fin(file);
      if (!fin) {
        throw std::runtime_error(std::string("[OVERRIDE_SOLUTION] cannot open ") + file);
      }
      for (int i = 0; i < dof; i++) {
        for (int a = 0; a < lhs.nNo; a++) {
          fin >> R(i,a);
        }
      }
      std::cerr << "[OVERRIDE_SOLUTION] overrode solve() output at call " << overrideCallCount
                << " from " << file << " (skipped real linear solve)\n";
      overrideCallCount++;
      return;
    }
    overrideCallCount++;
  }

  // TEMPORARY DIAGNOSTIC: see fsils_diag_exact_direct_solve() below.
  bool fsils_diag_exact_direct_solve(ComMod& com_mod, eqType& lEq, const Vector<int>& incL);
  if (fsils_diag_exact_direct_solve(com_mod, lEq, incL)) {
    return;
  }

  fsi_linear_solver::fsils_solve(lhs, lEq.FSILS, dof, R, Val, preconditioner, incL, res);
}

#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

// TEMPORARY DIAGNOSTIC: when EXACT_DIRECT_SOLVE is set, solve the raw
// assembled system exactly (Eigen SparseLU) under the same per-component
// Dirichlet mask FSILS and Trilinos apply (face.val on BC_TYPE_Dir faces),
// in place of FSILS's Krylov solve -- reproduces exact-Newton behavior
// locally without a Trilinos build. EXACT_DIRECT_MAX_STEP, if set, caps
// max|du| per Newton step by uniformly scaling the update (crude step-size
// control, for testing whether globalization alone changes convergence).
bool fsils_diag_exact_direct_solve(ComMod& com_mod, eqType& lEq, const Vector<int>& incL)
{
  if (!std::getenv("EXACT_DIRECT_SOLVE")) {
    return false;
  }
  static int callCount = 0;
  const auto t0 = std::chrono::steady_clock::now();
  auto& lhs = com_mod.lhs;
  auto& R = com_mod.R;
  const auto& Val = com_mod.Val;
  const int dof = com_mod.dof;
  const int nsd = com_mod.nsd;
  const int nNo = lhs.nNo;
  const int N = dof * nNo;

  std::vector<double> mask(N, 1.0);
  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    const auto& face = lhs.face[faIn];
    if ((incL.size() != 0 && incL(faIn) == 0) || face.bGrp != fsi_linear_solver::BcType::BC_TYPE_Dir) {
      continue;
    }
    const int n = std::min(face.dof, dof);
    for (int a = 0; a < face.nNo; a++) {
      for (int i = 0; i < n; i++) {
        mask[face.glob(a)*dof + i] *= face.val(i,a);
      }
    }
  }

  std::vector<int> freeIdx(N, -1);
  int nFree = 0;
  for (int k = 0; k < N; k++) {
    if (mask[k] > 0.5) {
      freeIdx[k] = nFree++;
    }
  }

  std::vector<Eigen::Triplet<double>> trip;
  trip.reserve(static_cast<size_t>(lhs.nnz) * dof * dof);
  for (int a = 0; a < nNo; a++) {
    for (int k = lhs.rowPtr(0,a); k <= lhs.rowPtr(1,a); k++) {
      const int b = lhs.colPtr(k);
      for (int i = 0; i < dof; i++) {
        const int r = freeIdx[a*dof + i];
        if (r < 0) {
          continue;
        }
        for (int j = 0; j < dof; j++) {
          const int c = freeIdx[b*dof + j];
          const double v = Val(i*dof + j, k);
          if (c >= 0 && v != 0.0) {
            trip.emplace_back(r, c, v);
          }
        }
      }
    }
  }
  Eigen::SparseMatrix<double> A(nFree, nFree);
  A.setFromTriplets(trip.begin(), trip.end());

  std::vector<double> bl(N, 0.0);
  for (int a = 0; a < nNo; a++) {
    for (int i = 0; i < dof; i++) {
      bl[lhs.map(a)*dof + i] = R(i,a);
    }
  }
  Eigen::VectorXd rhs(nFree);
  for (int k = 0; k < N; k++) {
    if (freeIdx[k] >= 0) {
      rhs(freeIdx[k]) = bl[k];
    }
  }

  Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
  lu.analyzePattern(A);
  lu.factorize(A);
  if (lu.info() != Eigen::Success) {
    throw std::runtime_error("[EXACT_DIRECT_SOLVE] SparseLU factorization failed (singular system) at call " +
                             std::to_string(callCount));
  }
  const Eigen::VectorXd x = lu.solve(rhs);
  const double iNorm = rhs.norm();
  const double fNorm = (A*x - rhs).norm();

  std::vector<double> xl(N, 0.0);
  for (int k = 0; k < N; k++) {
    if (freeIdx[k] >= 0) {
      xl[k] = x(freeIdx[k]);
    }
  }
  double maxDu = 0.0;
  int maxNode = -1;
  for (int a = 0; a < nNo; a++) {
    for (int i = 0; i < nsd && i < dof; i++) {
      if (std::abs(xl[a*dof + i]) > maxDu) {
        maxDu = std::abs(xl[a*dof + i]);
        maxNode = a;
      }
    }
  }
  double scale = 1.0;
  if (const char* capEnv = std::getenv("EXACT_DIRECT_MAX_STEP")) {
    const double cap = std::atof(capEnv);
    if (cap > 0.0 && maxDu > cap) {
      scale = cap / maxDu;
    }
  }
  for (int a = 0; a < nNo; a++) {
    for (int i = 0; i < dof; i++) {
      R(i,a) = scale * xl[lhs.map(a)*dof + i];
    }
  }

  lEq.FSILS.RI.iNorm = iNorm;
  lEq.FSILS.RI.fNorm = fNorm;
  lEq.FSILS.RI.itr = 1;
  lEq.FSILS.RI.success = true;
  lEq.FSILS.RI.dB = (iNorm > 0.0 && fNorm > 0.0) ? 10.0 * std::log10(fNorm / iNorm) : 0.0;
  lEq.FSILS.RI.callD = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::cerr << "[EXACT_DIRECT_SOLVE] call " << callCount << ": ||R_masked||=" << iNorm
            << " lin.res=" << fNorm / std::max(iNorm, 1e-300) << " max|du|=" << maxDu
            << " (lhs node " << maxNode << ") step scale=" << scale << "\n";
  callCount++;
  return true;
}

