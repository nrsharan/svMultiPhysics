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

  fsi_linear_solver::fsils_solve(lhs, lEq.FSILS, dof, R, Val, preconditioner, incL, res);
}

