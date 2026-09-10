#!/usr/bin/env python3
"""Generate the geometry/BC override that makes FEDDLib's hollow-cylinder mesh
identical to this test's svMultiPhysics mesh.

FEDDLib only reads P1 medit meshes (feddlib_mesh/hollow_cylinder_p1.mesh, see
generate_feddlib_mesh.py) and builds P2 with straight edges, while this test's
gmsh order-2 mesh puts the mid-edge nodes of the curved inner/outer walls on the
true cylinder. Both meshes share the same vertices and tets, so every FEDDLib P2
node can be keyed by its straight position: a vertex, or the midpoint of its
edge's two vertices (computed from FEDDLib's own vertex coordinates, exactly as
FEDDLib does). Output lines:

  COORD kx ky kz x y z   move the P2 node at key k to svMultiPhysics's position
  FLAG  kx ky kz f       give the node at key k Dirichlet flag f

FLAG lines cover exactly svMultiPhysics's Dirichlet node sets -- every node of
the bottom/top faces and of each 6-node pin face -- with the flags FEDDLib's
ccb_hollow_cylinder main.cpp uses (2/3 = Dirichlet_Z, 13/16 = Dirichlet_Y,
14 = Dirichlet_X). The FEDDLib driver resets any other node carrying one of those
flags, so the constrained node sets end up identical.

Requires: the `meshio` Python package.
Usage: python3 generate_feddlib_p2_override.py
"""

import os
import re

import meshio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOLUME_MESH = os.path.join(HERE, "mesh", "mesh-complete.mesh.vtu")
FACE_DIR = os.path.join(HERE, "mesh", "mesh-surfaces")
SOLVER_XML = os.path.join(HERE, "solver.xml")
P1_MESH = os.path.join(HERE, "feddlib_mesh", "hollow_cylinder_p1.mesh")
OUT = os.path.join(HERE, "feddlib_mesh", "hollow_cylinder_p2_override.txt")

# meshio's "tetra10" order is VTK's: corners 0-3, then the edges below.
TET10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]

# svMultiPhysics Dirichlet face -> (FEDDLib flag, constrained component).
FACE_FLAGS = {
    "bottom": (2, 2),
    "top": (3, 2),
    "pin0": (13, 1),
    "pin1": (14, 0),
    "pin2": (16, 1),
}

MATCH_TOL = 1e-8


def ascii_array(text, name, dtype):
    m = re.search(r'<DataArray[^>]*Name="%s"[^>]*format="ascii"[^>]*>(.*?)</DataArray>' % name, text, re.S)
    if m is None:
        raise RuntimeError(f"array {name} not found")
    return np.array(m.group(1).split(), dtype=dtype)


def read_p1_vertices(path):
    with open(path) as f:
        lines = [line.strip() for line in f]
    i = lines.index("Vertices")
    n = int(lines[i + 1])
    return np.array([[float(v) for v in lines[i + 2 + k].split()[:3]] for k in range(n)])


def check_effective_directions():
    """The pin/face flag mapping above must agree with solver.xml's BCs."""
    xml = open(SOLVER_XML).read()
    for name, direction in re.findall(
            r'<Add_BC name="(\w+)"\s*>\s*<Type>\s*Dir\s*</Type>.*?<Effective_direction>\s*\(([^)]*)\)', xml, re.S):
        comps = [i for i, c in enumerate(direction.split(",")) if float(c) != 0.0]
        if name not in FACE_FLAGS or comps != [FACE_FLAGS[name][1]]:
            raise RuntimeError(f"solver.xml Dir BC '{name}' constrains {comps}, not what FACE_FLAGS assumes")


def main():
    check_effective_directions()

    mesh = meshio.read(VOLUME_MESH)
    X = np.asarray(mesh.points, dtype=float)
    tets = [c for c in mesh.cells if c.type == "tetra10"][0].data
    volume_text = open(VOLUME_MESH).read()
    point_of_gid = {g: i for i, g in enumerate(ascii_array(volume_text, "GlobalNodeID", int))}

    # Key every svMultiPhysics node by FEDDLib's straight P2 position, using FEDDLib's
    # own (rounded) vertex coordinates.
    V = read_p1_vertices(P1_MESH)
    corners = np.unique(tets[:, :4])
    dist = np.linalg.norm(X[corners][:, None, :] - V[None, :, :], axis=2)
    vertex_of = dict(zip(corners, dist.argmin(axis=1)))
    worst = dist.min(axis=1).max()
    if worst > MATCH_TOL or len(set(vertex_of.values())) != len(corners) or len(corners) != len(V):
        raise RuntimeError(f"svMultiPhysics corners do not map one-to-one onto FEDDLib's P1 vertices (worst {worst:.2e})")

    key = {c: V[vertex_of[c]] for c in corners}
    edge_of = {}
    for tet in tets:
        for e, (a, b) in enumerate(TET10_EDGES):
            node = tet[4 + e]
            pair = tuple(sorted((vertex_of[tet[a]], vertex_of[tet[b]])))
            if edge_of.setdefault(node, pair) != pair:
                raise RuntimeError(f"mid-edge node {node} belongs to two different edges")
            key[node] = 0.5 * (V[pair[0]] + V[pair[1]])

    keys = np.array([key[p] for p in range(len(X))])
    if len(np.unique(np.round(keys / MATCH_TOL).astype(np.int64), axis=0)) != len(X):
        raise RuntimeError("two nodes share the same straight-P2 key")

    moved = [p for p in range(len(X)) if np.linalg.norm(X[p] - keys[p]) > 1e-12]

    flag_lines = []
    counts = {}
    for face, (flag, _) in FACE_FLAGS.items():
        face_text = open(os.path.join(FACE_DIR, face + ".vtp")).read()
        nodes = [point_of_gid[g] for g in ascii_array(face_text, "GlobalNodeID", int)]
        counts[face] = len(nodes)
        flag_lines += [(p, flag) for p in nodes]

    with open(OUT, "w") as f:
        f.write("# Geometry/BC override making FEDDLib's P2 hollow cylinder identical to\n")
        f.write("# svMultiPhysics's tests/cases/def_diffu/hollow_cylinder mesh.\n")
        f.write("# Generated by generate_feddlib_p2_override.py -- do not edit by hand.\n")
        f.write(f"# {len(moved)} COORD lines, {len(flag_lines)} FLAG lines ({counts})\n")
        for p in moved:
            k, x = keys[p], X[p]
            f.write(f"COORD {k[0]:.17g} {k[1]:.17g} {k[2]:.17g} {x[0]:.17g} {x[1]:.17g} {x[2]:.17g}\n")
        for p, flag in flag_lines:
            k = keys[p]
            f.write(f"FLAG {k[0]:.17g} {k[1]:.17g} {k[2]:.17g} {flag}\n")

    print(f"Wrote {OUT}")
    print(f"  {len(X)} nodes ({len(corners)} vertices), {len(moved)} mid-edge nodes moved onto the curved walls "
          f"(max offset {max(np.linalg.norm(X[p] - keys[p]) for p in moved):.4e})")
    print(f"  Dirichlet node sets: {counts}")


if __name__ == "__main__":
    main()
