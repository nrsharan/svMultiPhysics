#!/usr/bin/env python3
"""Compare the first Newton matrix of svMultiPhysics (artery_dan) with FEDDLib's
(artery_dan_cmm), block by block in FEDDLib's layout.

svMultiPhysics (SVMP_TRILINOS_WRITE_SYSTEM=<svmp>): <svmp>_K.mtx (dof node*4+d,
d = 0..2 displacement, 3 concentration; before the Jacobi scaling, Dirichlet
rows as assembled), <svmp>_dirichlet_<rank>.txt, <svmp>_coords_<rank>.txt.
FEDDLib (FEDD_WRITE_SYSTEM=<fedd>): <fedd>00.mm (displacement, node*3+d),
<fedd>01.mm, <fedd>10.mm, <fedd>11.mm (concentration, node), and
<fedd>_coords_<rank>.txt. Nodes are matched by their coordinates.

usage: compare_systems.py <svmp prefix> <fedd prefix>
"""
import glob
import sys

import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.spatial import cKDTree

svmp, fedd = sys.argv[1], sys.argv[2]
NSD, DOF = 3, 4


def read_coords(prefix):
    data = np.vstack([np.loadtxt(f, ndmin=2) for f in sorted(glob.glob(prefix + "_coords_*.txt"))])
    gid = data[:, 0].astype(np.int64)
    order = np.unique(gid, return_index=True)[1]
    gid, xyz = gid[order], data[order, 1:1 + NSD]
    assert np.all(gid == np.arange(len(gid))), "node GIDs are not 0..n-1"
    return xyz


def read(path):
    A = scipy.io.mmread(path)
    return sp.csr_matrix(A)


def stats(name, A):
    A = A.tocsr()
    nnz_row = np.diff(A.indptr)
    d = A.diagonal()
    fro = sp.linalg.norm(A)
    asym = sp.linalg.norm(A - A.T) / fro if A.shape[0] == A.shape[1] and fro > 0 else float("nan")
    print(f"  {name:22s} {A.shape[0]:7d}x{A.shape[1]:<7d} nnz {A.nnz:10d}  nnz/row mean {nnz_row.mean():6.1f} "
          f"max {nnz_row.max():4d}  |A|_F {fro:10.3e}  max|a| {abs(A).max():10.3e}  "
          f"|diag| median {np.median(abs(d)) if len(d) else 0:10.3e}  asym {asym:8.2e}")


print("== node matching")
xs, xf = read_coords(svmp), read_coords(fedd)
print(f"  svMultiPhysics nodes {len(xs)}, FEDDLib nodes {len(xf)}")
dist, s_of_f = cKDTree(xs).query(xf)
print(f"  max distance of a FEDDLib node to the nearest svMultiPhysics node: {dist.max():.3e}")
assert len(np.unique(s_of_f)) == len(xf), "node matching is not one to one"
N = len(xf)

print("== reading matrices")
K = read(svmp + "_K.mtx")
F = {ij: read(f"{fedd}{ij}.mm") for ij in ("00", "01", "10", "11")}

# svMultiPhysics dofs in FEDDLib's block layout.
disp = (DOF * s_of_f[:, None] + np.arange(NSD)[None, :]).ravel()   # FEDDLib dof 3m+d
conc = DOF * s_of_f + NSD                                          # FEDDLib dof m
rows = {"0": disp, "1": conc}
S = {ij: K[rows[ij[0]]][:, rows[ij[1]]].tocsr() for ij in F}

dirichlet = set()
for f in glob.glob(svmp + "_dirichlet_*.txt"):
    dirichlet.update(np.loadtxt(f, dtype=np.int64, ndmin=1).tolist())
dir_disp = np.array([g in dirichlet for g in disp])
dir_conc = np.array([g in dirichlet for g in conc])
dirs = {"0": dir_disp, "1": dir_conc}
print(f"  svMultiPhysics Dirichlet dofs: displacement {dir_disp.sum()}, concentration {dir_conc.sum()}")

names = {"00": "disp-disp", "01": "disp-conc", "10": "conc-disp", "11": "conc-conc"}
for ij in ("00", "01", "10", "11"):
    print(f"== block {ij} ({names[ij]})")
    stats("svMultiPhysics", S[ij])
    stats("FEDDLib", F[ij])

    # Pattern and values on the rows that are not Dirichlet in svMultiPhysics.
    free = ~dirs[ij[0]]
    Sf, Ff = S[ij][free], F[ij][free]
    Sb, Fb = Sf.copy(), Ff.copy()
    Sb.data[:] = 1
    Fb.data[:] = 1
    both = Sb.multiply(Fb)
    print(f"  free rows {free.sum()}: entries in both {both.nnz}, only svMultiPhysics {Sb.nnz - both.nnz}, "
          f"only FEDDLib {Fb.nnz - both.nnz}")
    common_s = np.asarray(Sf[both.nonzero()]).ravel()
    common_f = np.asarray(Ff[both.nonzero()]).ravel()
    ok = (abs(common_s) > 1e-14 * abs(common_s).max()) & (abs(common_f) > 1e-14 * abs(common_f).max())
    if ok.any():
        r = common_s[ok] / common_f[ok]
        q = np.percentile(r, [1, 25, 50, 75, 99])
        print(f"  ratio svMultiPhysics/FEDDLib on common entries: percentiles 1/25/50/75/99 "
              + " ".join(f"{v:.4g}" for v in q))
        scale = np.median(r)
        diff = sp.linalg.norm(Sf - scale * Ff) / sp.linalg.norm(scale * Ff)
        print(f"  |S - {scale:.4g} F|_F / |{scale:.4g} F|_F on free rows = {diff:.3e}")

    # FEDDLib's Dirichlet rows: rows of the diagonal blocks with a single entry.
    if ij[0] == ij[1]:
        nnz_row = np.diff(F[ij].indptr)
        single = nnz_row == 1
        print(f"  FEDDLib rows with a single entry: {single.sum()} "
              f"(svMultiPhysics Dirichlet rows among them: {(single & dirs[ij[0]]).sum()})")
        cols_of_dir = abs(F[ij][:, dirs[ij[0]]]).sum()
        print(f"  |FEDDLib columns of svMultiPhysics's Dirichlet dofs|_1 = {cols_of_dir:.3e} "
              "(0 if the Dirichlet columns are eliminated)")
