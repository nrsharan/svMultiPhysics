#!/usr/bin/env python3
"""Convert a FEDDLib (medit .mesh, linear tetrahedra) mesh to svMultiPhysics
input files for the deformation-diffusion (def_diffu) equation.

Writes <out>/mesh-complete.mesh.vtu (quadratic TET10 elements, mid-edge nodes
on straight edges, as FEDDLib builds its P2 mesh from the P1 file) with a
DOMAIN_ID cell array equal to FEDDLib's volume (element) flag, so the mesh file
itself can be given as <Domain_file_path> and each FEDDLib material region
becomes the svMultiPhysics <Domain id="flag">. For every requested triangle
flag, <out>/mesh-surfaces/<name>.vtp holds the TRI6 faces with that flag.

FEDDLib also flags single vertices (e.g. points held in x/z or y/z to remove
rigid-body modes). svMultiPhysics prescribes Dirichlet conditions on faces, so
--pin writes a one-element face: the boundary triangle of a given triangle flag
that contains the flagged vertex. That face constrains all six of its nodes,
not only the vertex.

Example (FEDDLib meshes/SPP2311/plaque_short_6_k.mesh):
  python3 feddlib_mesh_to_svmp.py plaque_short_6_k.mesh mesh \\
      --face 2:bottom --face 3:top --face 5:inner --face 6:outer \\
      --pin 13:2:pin_xz --pin 14:2:pin_yz

Needs numpy. Reuses the VTU/VTP writers of ../hollow_cylinder/generate_mesh.py.
"""
import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hollow_cylinder"))
from generate_mesh import vtu_data_array, write_face_vtp  # noqa: E402

# VTK_QUADRATIC_TETRA edge order: (0,1), (1,2), (0,2), (0,3), (1,3), (2,3).
TET_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
# VTK_QUADRATIC_TRIANGLE edge order: (0,1), (1,2), (2,0).
TRI_EDGES = ((0, 1), (1, 2), (2, 0))


def read_medit(path):
    """Sections Vertices, Edges, Triangles, Tetrahedra of a medit .mesh file as
    integer/float arrays; connectivity is converted to 0-based indices and the
    reference (flag) is kept as the last column."""
    with open(path) as f:
        tokens = f.read().split()
    sections = {}
    sizes = {"Vertices": 3, "Edges": 2, "Triangles": 3, "Tetrahedra": 4}
    i = 0
    while i < len(tokens):
        key = tokens[i]
        if key in sizes:
            count = int(tokens[i + 1])
            width = sizes[key] + 1
            data = np.array(tokens[i + 2:i + 2 + count * width], dtype=float).reshape(count, width)
            sections[key] = data
            i += 2 + count * width
        else:
            i += 1
    vertices = sections["Vertices"]
    out = {"points": vertices[:, :3], "point_flags": vertices[:, 3].astype(int)}
    for key in ("Edges", "Triangles", "Tetrahedra"):
        if key in sections:
            data = sections[key]
            out[key] = (data[:, :-1].astype(int) - 1, data[:, -1].astype(int))
    return out


def orient_tets(points, tets):
    """Reorder corners so that every tetrahedron has a positive volume."""
    tets = tets.copy()
    p = points[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    if np.any(np.abs(vol) < 1e-14 * np.abs(vol).max()):
        raise ValueError("degenerate tetrahedra in the mesh")
    flip = vol < 0
    tets[flip, 1], tets[flip, 2] = tets[flip, 2].copy(), tets[flip, 1].copy()
    return tets, int(flip.sum())


def quadratic_mesh(points, tets):
    """Add one node at the middle of every edge. Returns the new point array,
    the TET10 connectivity and the edge -> mid-node map."""
    midnode = {}
    new_points = [points]
    next_id = len(points)
    extra = []
    tet10 = np.empty((len(tets), 10), dtype=np.int64)
    tet10[:, :4] = tets
    for e, tet in enumerate(tets):
        for k, (a, b) in enumerate(TET_EDGES):
            key = (min(tet[a], tet[b]), max(tet[a], tet[b]))
            node = midnode.get(key)
            if node is None:
                node = next_id
                midnode[key] = node
                next_id += 1
                extra.append(0.5 * (points[key[0]] + points[key[1]]))
            tet10[e, 4 + k] = node
    if extra:
        new_points.append(np.array(extra))
    return np.vstack(new_points), tet10, midnode


def write_volume_vtu(points, tet10, domain_id, path):
    n_pts, n_cells = len(points), len(tet10)
    offsets = np.arange(10, 10 * n_cells + 1, 10)
    types = np.full(n_cells, 24, dtype=int)  # VTK_QUADRATIC_TETRA
    with open(path, "w") as f:
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n_pts}" NumberOfCells="{n_cells}">\n')
        f.write("      <PointData>\n")
        f.write(vtu_data_array("GlobalNodeID", np.arange(1, n_pts + 1), dtype="Int32", fmt="%d"))
        f.write("      </PointData>\n")
        f.write('      <CellData Scalars="ModelRegionID">\n')
        f.write(vtu_data_array("ModelRegionID", domain_id, dtype="Int32", fmt="%d"))
        f.write(vtu_data_array("DOMAIN_ID", domain_id, dtype="Int32", fmt="%d"))
        f.write(vtu_data_array("GlobalElementID", np.arange(1, n_cells + 1), dtype="Int32", fmt="%d"))
        f.write("      </CellData>\n")
        f.write("      <Points>\n")
        # Same number format as the face files (write_face_vtp): svMultiPhysics
        # matches face nodes to volume nodes by their coordinates.
        f.write(vtu_data_array("Points", points, components=3, dtype="Float64"))
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write(vtu_data_array("connectivity", tet10, dtype="Int64", fmt="%d"))
        f.write(vtu_data_array("offsets", offsets, dtype="Int64", fmt="%d"))
        f.write(vtu_data_array("types", types, dtype="UInt8", fmt="%d"))
        f.write("      </Cells>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def boundary_faces(tets):
    """Map sorted corner triple -> (tet index, local face corners) for faces
    that belong to exactly one tetrahedron."""
    count = collections.Counter()
    owner = {}
    for e, tet in enumerate(tets):
        for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            key = tuple(sorted(tet[list(face)]))
            count[key] += 1
            owner[key] = e
    return {key: owner[key] for key, n in count.items() if n == 1}


def tri6_faces(points, tets, midnode, triangles, boundary):
    """TRI6 connectivity (0-based) of the given boundary triangles, oriented
    with the normal pointing out of the owning tetrahedron, and the owners."""
    conn = np.empty((len(triangles), 6), dtype=np.int64)
    owners = np.empty(len(triangles), dtype=np.int64)
    for i, tri in enumerate(triangles):
        key = tuple(sorted(tri))
        if key not in boundary:
            raise ValueError(f"triangle {tri + 1} is not a boundary face of the tetrahedral mesh")
        e = boundary[key]
        a, b, c = tri
        normal = np.cross(points[b] - points[a], points[c] - points[a])
        if np.dot(normal, points[a] - points[tets[e]].mean(axis=0)) < 0:
            b, c = c, b
        corners = (a, b, c)
        conn[i, :3] = corners
        for k, (p, q) in enumerate(TRI_EDGES):
            conn[i, 3 + k] = midnode[(min(corners[p], corners[q]), max(corners[p], corners[q]))]
        owners[i] = e
    return conn, owners


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("medit", help="FEDDLib .mesh file (linear tetrahedra)")
    parser.add_argument("out", help="output directory (mesh-complete.mesh.vtu, mesh-surfaces/)")
    parser.add_argument("--face", action="append", default=[], metavar="FLAG:NAME",
                        help="write the triangles with this flag as mesh-surfaces/NAME.vtp")
    parser.add_argument("--pin", action="append", default=[], metavar="VFLAG:TFLAG:NAME",
                        help="one-element face NAME: a TFLAG triangle containing each vertex flagged VFLAG")
    parser.add_argument("--dirichlet-override", metavar="FILE",
                        help="write a FEDDLib geometry override file (FLAG lines, read by FEDDLib's "
                             "artery_dan_cmm) that gives every node of the written faces the FEDDLib "
                             "Dirichlet flag chosen by --override-flag, so that FEDDLib constrains exactly "
                             "svMultiPhysics's node sets")
    parser.add_argument("--override-flag", action="append", default=[], metavar="FACES=FLAG",
                        help="FEDDLib flag of the nodes lying on all the named faces (names joined by '+'); "
                             "rules are tried in the order given and the first match wins, e.g. "
                             "pin_xz+outer=23 pin_xz=13 bottom+inner=7 bottom=2")
    args = parser.parse_args()

    mesh = read_medit(args.medit)
    points = mesh["points"]
    tets, tet_flags = mesh["Tetrahedra"]
    tets, n_flipped = orient_tets(points, tets)
    points10, tet10, midnode = quadratic_mesh(points, tets)

    os.makedirs(os.path.join(args.out, "mesh-surfaces"), exist_ok=True)
    write_volume_vtu(points10, tet10, tet_flags, os.path.join(args.out, "mesh-complete.mesh.vtu"))
    regions = collections.Counter(tet_flags.tolist())
    print(f"{args.medit}: {len(points)} vertices -> {len(points10)} TET10 nodes, {len(tets)} elements "
          f"({n_flipped} reoriented); DOMAIN_ID (element flag) counts: {dict(sorted(regions.items()))}")

    boundary = boundary_faces(tets)
    triangles, tri_flags = mesh.get("Triangles", (np.empty((0, 3), dtype=int), np.empty(0, dtype=int)))
    face_id = 0
    face_nodes = {}  # face name -> set of its TRI6 nodes, for --dirichlet-override
    for spec in args.face:
        flag, name = spec.split(":")
        selected = triangles[tri_flags == int(flag)]
        if len(selected) == 0:
            raise ValueError(f"no triangles with flag {flag}")
        conn, owners = tri6_faces(points10, tets, midnode, selected, boundary)
        face_id += 1
        write_face_vtp(points10, conn + 1, owners + 1, face_id,
                       os.path.join(args.out, "mesh-surfaces", f"{name}.vtp"), nodes_per_face=6)
        face_nodes[name] = set(conn.ravel().tolist())
        print(f"face {name}: flag {flag}, {len(conn)} TRI6 elements")

    point_flags = mesh["point_flags"]
    for spec in args.pin:
        vflag, tflag, name = spec.split(":")
        vertices = np.flatnonzero(point_flags == int(vflag))
        if len(vertices) == 0:
            raise ValueError(f"no vertices with flag {vflag}")
        candidates = triangles[tri_flags == int(tflag)]
        chosen = []
        for v in vertices:
            containing = candidates[np.any(candidates == v, axis=1)]
            if len(containing) == 0:
                raise ValueError(f"vertex {v + 1} (flag {vflag}) is on no triangle with flag {tflag}")
            chosen.append(containing[0])
        conn, owners = tri6_faces(points10, tets, midnode, np.array(chosen), boundary)
        face_id += 1
        write_face_vtp(points10, conn + 1, owners + 1, face_id,
                       os.path.join(args.out, "mesh-surfaces", f"{name}.vtp"), nodes_per_face=6)
        face_nodes[name] = set(conn.ravel().tolist())
        print(f"pin face {name}: vertex flag {vflag} ({len(vertices)} vertices: "
              f"{np.round(points[vertices], 4).tolist()}) on flag-{tflag} triangles")

    if args.dirichlet_override:
        write_dirichlet_override(args.dirichlet_override, points10, face_nodes, args.override_flag)


def write_dirichlet_override(path, points, face_nodes, specs):
    """FEDDLib FLAG lines 'FLAG x y z flag' for every node of the written faces:
    the node's position (FEDDLib builds the same straight-edged P2 mesh, so it
    finds the node by position) and the flag of the first rule whose faces all
    contain the node."""
    rules = []
    for spec in specs:
        faces, flag = spec.split("=")
        names = faces.split("+")
        for name in names:
            if name not in face_nodes:
                raise ValueError(f"--override-flag {spec}: no face '{name}' was written")
        rules.append((names, int(flag), spec))
    if not rules:
        raise ValueError("--dirichlet-override needs at least one --override-flag rule")

    lines = []
    counts = collections.Counter()
    for node in sorted(set().union(*face_nodes.values())):
        for names, flag, spec in rules:
            if all(node in face_nodes[name] for name in names):
                x, y, z = points[node]
                lines.append(f"FLAG {x:.17g} {y:.17g} {z:.17g} {flag}")
                counts[spec] += 1
                break
        else:
            raise ValueError(f"node at {points[node].tolist()} matches no --override-flag rule")

    with open(path, "w") as f:
        f.write("# FEDDLib Dirichlet flags of svMultiPhysics's constrained nodes, written by\n")
        f.write("# tests/cases/def_diffu/tools/feddlib_mesh_to_svmp.py with the rules (first match wins):\n")
        f.write("#   " + " ".join(spec for _, _, spec in rules) + "\n")
        f.write("# FLAG <straight-P2 node position x y z> <flag>\n")
        f.write("\n".join(lines) + "\n")
    print(f"Dirichlet override {path}: {len(lines)} nodes; " +
          ", ".join(f"{spec}: {counts[spec]}" for _, _, spec in rules))


if __name__ == "__main__":
    main()
