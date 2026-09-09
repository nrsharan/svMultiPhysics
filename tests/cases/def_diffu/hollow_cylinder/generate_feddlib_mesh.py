#!/usr/bin/env python3
"""Generate a P1 (linear tet) medit .mesh file for the SAME hollow-cylinder
geometry as generate_mesh.py (Ri=1.0, Ro=1.3, L=2.0), for use with FEDDLib
(which reads P1 medit meshes and builds its own P2 enrichment internally --
see feddlib/core/Mesh/MeshFileReader.hpp and MeshPartitioner).

Flags (chosen to match a materialParameters.xml / main.cpp BC setup built
alongside this):
  Tetrahedra (volume):        15
  Triangles (bottom):          2
  Triangles (top):             3
  Triangles (outer, no pins):  4
  Triangles (inner, pressure): 5
  Vertex (pin0, theta=0):      13
  Vertex (pin1, theta=90):     14
  Vertex (pin2, theta=180):    16
  All other vertices:          0

Usage: python3 generate_feddlib_mesh.py
"""

import os
import subprocess

import meshio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEO_PATH = os.path.join(HERE, "cylinder_p1.geo")
MSH_PATH = os.path.join(HERE, "cylinder_p1.msh")
OUT_MESH = os.path.join(HERE, "feddlib_mesh", "hollow_cylinder_p1.mesh")

RI = 1.0
RO = 1.3
LENGTH = 2.0
CHAR_LENGTH = 0.35

GEO_TEMPLATE = f"""\
SetFactory("OpenCASCADE");
Ri = {RI};
Ro = {RO};
L = {LENGTH};
lc = {CHAR_LENGTH};

Circle(1) = {{0,0,0, Ro}};
Circle(2) = {{0,0,0, Ri}};
Curve Loop(1) = {{1}};
Curve Loop(2) = {{2}};
Plane Surface(1) = {{1, 2}};

out[] = Extrude {{0,0,L}} {{ Surface{{1}}; }};

Physical Surface("bottom") = {{1}};
Physical Surface("top") = {{out[0]}};
Physical Surface("outer") = {{out[2]}};
Physical Surface("inner") = {{out[3]}};
Physical Volume("tube") = {{out[1]}};

Mesh.CharacteristicLengthMax = lc;
Mesh.CharacteristicLengthMin = lc*0.6;
Mesh.ElementOrder = 1;
"""


def run_gmsh():
    with open(GEO_PATH, "w") as f:
        f.write(GEO_TEMPLATE)
    subprocess.run(["gmsh", "-3", GEO_PATH, "-o", MSH_PATH], check=True)


def compute_vertex_flags(points):
    """Assign each vertex a BC flag directly from its geometry (z/r
    thresholds), matching the face_flag_map used for Triangles (2=bottom,
    3=top, 4=outer, 5=inner). This is critical: FEDDLib's MeshFileReader
    reads Dirichlet-BC node flags from the per-vertex flag column in the
    "Vertices" section (readNodes() -> bcFlagRep_), NOT by propagating the
    Triangles' face flags down to their vertices. A vertex flag of 0 for
    every non-pin node (as an earlier version of this script wrote) means
    BCBuilder's addBC(..., 2, ...)/addBC(..., 3, ...) (Dirichlet_Z on
    bottom/top) match ZERO nodes -- the structure ends up constrained only
    at the 3 single-component pins, i.e. is massively under-constrained
    (no Z-fixity anywhere), which is exactly what produced the catastrophic,
    near-singular-stiffness-matrix blow-up (||F|| exploding to inf within a
    few Newton iterations) seen on the cluster run.
    Bottom/top (z-based) take priority over outer/inner (r-based) at rim
    vertices, so the full z=0 and z=L annuli -- including rim nodes also on
    the inner/outer cylindrical surface -- get flag 2/3 and thus Dirichlet_Z,
    matching what main.cpp's BC scheme assumes."""
    tol = 1e-6
    flags = np.zeros(len(points), dtype=int)
    for i, (x, y, z) in enumerate(points):
        r = np.hypot(x, y)
        if abs(z - 0.0) < tol:
            flags[i] = 2
        elif abs(z - LENGTH) < tol:
            flags[i] = 3
        elif abs(r - RO) < tol:
            flags[i] = 4
        elif abs(r - RI) < tol:
            flags[i] = 5
        else:
            flags[i] = 0
    return flags


def pick_pin_vertices(points, outer_tri_conn):
    """Pick 3 vertices on the outer surface, at mid-height, near theta =
    0/90/180 degrees -- same logic/rationale as generate_mesh.py's
    pick_circumferential_pin_faces(), but returning single vertex indices
    (medit's per-vertex flag is a natural fit for point constraints, unlike
    svMultiPhysics's face-based BC system)."""
    outer_nodes = np.unique(outer_tri_conn)
    z_target = LENGTH / 2.0
    pins = []
    for target_deg in (0.0, 90.0, 180.0):
        target_rad = np.radians(target_deg)
        best = None
        best_score = None
        for n in outer_nodes:
            x, y, z = points[n]
            theta = np.arctan2(y, x)
            dtheta = np.abs(np.arctan2(np.sin(theta - target_rad), np.cos(theta - target_rad)))
            dz = np.abs(z - z_target)
            score = dtheta * 5.0 + dz
            if best_score is None or score < best_score:
                best_score = score
                best = n
        pins.append(best)
    return pins


def main():
    run_gmsh()
    m = meshio.read(MSH_PATH)
    points = m.points

    tet_conn = None
    tri_conn_list = []
    tri_tag_list = []
    phys = m.cell_data["gmsh:physical"]
    for cb, tags in zip(m.cells, phys):
        if cb.type == "tetra":
            tet_conn = cb.data
        elif cb.type == "triangle":
            tri_conn_list.append(cb.data)
            tri_tag_list.append(tags)
    tri_conn = np.concatenate(tri_conn_list, axis=0)
    tri_tags = np.concatenate(tri_tag_list, axis=0)

    field_data = m.field_data  # name -> [tag, dim]
    tag_of = {name: int(v[0]) for name, v in field_data.items()}

    outer_gmsh_tag = tag_of["outer"]
    outer_tri_mask = tri_tags == outer_gmsh_tag
    outer_conn = tri_conn[outer_tri_mask]
    pins = pick_pin_vertices(points, outer_conn)
    pin_flags = {pins[0]: 13, pins[1]: 14, pins[2]: 16}

    vertex_flags = compute_vertex_flags(points)
    for idx, flag in pin_flags.items():
        vertex_flags[idx] = flag

    face_flag_map = {"bottom": 2, "top": 3, "outer": 4, "inner": 5}

    os.makedirs(os.path.dirname(OUT_MESH), exist_ok=True)
    with open(OUT_MESH, "w") as f:
        f.write("MeshVersionFormatted 2\n")
        f.write("Dimension 3\n\n")

        f.write("Vertices\n")
        f.write(f"{len(points)}\n")
        for i, p in enumerate(points):
            f.write(f"{p[0]:.10g} {p[1]:.10g} {p[2]:.10g} {vertex_flags[i]}\n")
        f.write("\n")

        f.write("Triangles\n")
        f.write(f"{len(tri_conn)}\n")
        for conn, tag in zip(tri_conn, tri_tags):
            name = next(n for n, t in tag_of.items() if t == tag)
            flag = face_flag_map[name]
            f.write(f"{conn[0]+1} {conn[1]+1} {conn[2]+1} {flag}\n")
        f.write("\n")

        f.write("Tetrahedra\n")
        f.write(f"{len(tet_conn)}\n")
        for conn in tet_conn:
            f.write(f"{conn[0]+1} {conn[1]+1} {conn[2]+1} {conn[3]+1} 15\n")
        f.write("\n")

        f.write("End\n")

    print(f"Wrote {OUT_MESH}: {len(points)} vertices, {len(tri_conn)} triangles, {len(tet_conn)} tets")
    print(f"Pin vertices (0-indexed gmsh ids): {pins}, flags: 13/14/16")
    for idx in pins:
        x, y, z = points[idx]
        theta = np.degrees(np.arctan2(y, x))
        print(f"  vertex {idx}: r={np.hypot(x,y):.4f} theta={theta:.2f} z={z:.4f}")


if __name__ == "__main__":
    main()
