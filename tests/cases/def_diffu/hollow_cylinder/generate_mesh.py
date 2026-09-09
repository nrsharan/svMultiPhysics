#!/usr/bin/env python3
"""Generate a Tet10 hollow-cylinder mesh (volume .vtu + face .vtp files) for
the deformation-diffusion (CCB constrained-mixture) equation test case.

Pipeline: gmsh (OpenCASCADE annulus extrusion, order-2 tets) -> meshio (reads
gmsh's native node ordering and re-expresses it in meshio's canonical
"tetra10"/"triangle6" ordering, which matches VTK's quadratic-tetra /
quadratic-triangle edge convention (0,1),(1,2),(2,0),(0,3),(1,3),(2,3) --
verified empirically by checking that most mid-edge nodes are the exact
arithmetic midpoint of their edge's corners; the AceGen node-order
permutation in ace_gen_cmm_smc_element.cpp handles the *separate* remaining
mismatch between VTK's and AceGen's own local tet10 numbering) -> hand-written
VTK XML (ASCII) matching svMultiPhysics's expected mesh-complete.mesh.vtu /
mesh-surfaces/*.vtp format.

Requires: gmsh (CLI) and the `meshio` Python package.

Usage: python3 generate_mesh.py
"""

import os
import subprocess
import sys

import meshio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEO_PATH = os.path.join(HERE, "cylinder.geo")
MSH_PATH = os.path.join(HERE, "cylinder.msh")
MESH_DIR = os.path.join(HERE, "mesh")
SURF_DIR = os.path.join(MESH_DIR, "mesh-surfaces")

# Geometry: inner radius Ri, outer radius Ro, length L.
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
Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 1;
"""


def run_gmsh():
    with open(GEO_PATH, "w") as f:
        f.write(GEO_TEMPLATE)
    subprocess.run(["gmsh", "-3", GEO_PATH, "-o", MSH_PATH], check=True)


def vtu_data_array(name, values, components=1, dtype="Float64", fmt="%.10g"):
    values = np.asarray(values)
    flat = values.reshape(-1) if components == 1 else values.reshape(-1, components)
    lines = " ".join(fmt % v for v in np.ravel(flat))
    comps_attr = f' NumberOfComponents="{components}"' if components > 1 else ""
    return (
        f'        <DataArray type="{dtype}" Name="{name}"{comps_attr} format="ascii">\n'
        f"          {lines}\n"
        f"        </DataArray>\n"
    )


def write_volume_vtu(points, tet_conn, path):
    n_pts = len(points)
    n_cells = len(tet_conn)
    global_node_id = np.arange(1, n_pts + 1)
    global_elem_id = np.arange(1, n_cells + 1)
    model_region_id = np.ones(n_cells, dtype=int)

    offsets = np.arange(10, 10 * n_cells + 1, 10)
    types = np.full(n_cells, 24, dtype=int)  # VTK_QUADRATIC_TETRA

    with open(path, "w") as f:
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n_pts}" NumberOfCells="{n_cells}">\n')
        f.write("      <PointData>\n")
        f.write(vtu_data_array("GlobalNodeID", global_node_id, dtype="Int32"))
        f.write("      </PointData>\n")
        f.write('      <CellData Scalars="ModelRegionID">\n')
        f.write(vtu_data_array("ModelRegionID", model_region_id, dtype="Int32"))
        f.write(vtu_data_array("GlobalElementID", global_elem_id, dtype="Int32"))
        f.write("      </CellData>\n")
        f.write("      <Points>\n")
        f.write(vtu_data_array("Points", points, components=3, dtype="Float64"))
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write(vtu_data_array("connectivity", tet_conn, dtype="Int64", fmt="%d"))
        f.write(vtu_data_array("offsets", offsets, dtype="Int64", fmt="%d"))
        f.write(vtu_data_array("types", types, dtype="UInt8", fmt="%d"))
        f.write("      </Cells>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def write_face_vtp(points, face_conn_global_node_ids, face_global_elem_ids, model_face_id, path,
                    nodes_per_face):
    """face_conn_global_node_ids: (n_face, nodes_per_face) array of 1-based
    GLOBAL (volume-mesh) node IDs for each face element, already in the
    convention decided after investigating svMultiPhysics's face-reading code
    (either 3 corners only, or all 6 tri6 nodes)."""
    local_ids = np.unique(face_conn_global_node_ids)
    id_map = {gid: i for i, gid in enumerate(local_ids)}
    local_points = points[local_ids - 1]
    local_conn = np.vectorize(id_map.get)(face_conn_global_node_ids)

    n_pts = len(local_ids)
    n_cells = len(face_conn_global_node_ids)
    offsets = np.arange(nodes_per_face, nodes_per_face * n_cells + 1, nodes_per_face)
    model_face_id_arr = np.full(n_cells, model_face_id, dtype=int)

    with open(path, "w") as f:
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n')
        f.write("  <PolyData>\n")
        f.write(
            f'    <Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" NumberOfLines="0" '
            f'NumberOfStrips="0" NumberOfPolys="{n_cells}">\n'
        )
        f.write("      <PointData>\n")
        f.write(vtu_data_array("GlobalNodeID", local_ids, dtype="Int32", fmt="%d"))
        f.write("      </PointData>\n")
        f.write('      <CellData Scalars="ModelFaceID">\n')
        f.write(vtu_data_array("ModelFaceID", model_face_id_arr, dtype="Int32", fmt="%d"))
        f.write(vtu_data_array("GlobalElementID", face_global_elem_ids, dtype="Int32", fmt="%d"))
        f.write("      </CellData>\n")
        f.write("      <Points>\n")
        f.write(vtu_data_array("Points", local_points, components=3, dtype="Float64"))
        f.write("      </Points>\n")
        f.write("      <Verts>\n")
        f.write(vtu_data_array("connectivity", np.array([], dtype=int), dtype="Int64", fmt="%d"))
        f.write(vtu_data_array("offsets", np.array([], dtype=int), dtype="Int64", fmt="%d"))
        f.write("      </Verts>\n")
        f.write("      <Lines>\n")
        f.write(vtu_data_array("connectivity", np.array([], dtype=int), dtype="Int64", fmt="%d"))
        f.write(vtu_data_array("offsets", np.array([], dtype=int), dtype="Int64", fmt="%d"))
        f.write("      </Lines>\n")
        f.write("      <Strips>\n")
        f.write(vtu_data_array("connectivity", np.array([], dtype=int), dtype="Int64", fmt="%d"))
        f.write(vtu_data_array("offsets", np.array([], dtype=int), dtype="Int64", fmt="%d"))
        f.write("      </Strips>\n")
        f.write("      <Polys>\n")
        f.write(vtu_data_array("connectivity", local_conn, dtype="Int64", fmt="%d"))
        f.write(vtu_data_array("offsets", offsets, dtype="Int64", fmt="%d"))
        f.write("      </Polys>\n")
        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")


def pick_circumferential_pin_faces(points, outer_tri_conn):
    """Pick 3 single TRI6 elements from the OUTER surface, near mid-height,
    centered at exactly theta = 0 deg, 90 deg, 180 deg -- the angles where
    the true local circumferential (tangential) direction happens to
    coincide exactly with a global Cartesian axis (tangent at theta=0/180
    is +-y; at theta=90 is +-x). This lets each pin's Dirichlet BC use a
    plain axis-aligned <Effective_direction>, which is all svMultiPhysics's
    BC mechanism actually supports (Effective_direction is an integer 0/1
    axis mask -- see BoundaryConditionParameters::effective_direction, a
    VectorParameter<int> -- NOT a way to specify an arbitrary oblique unit
    vector).

    A single isolated point ("Vert"-only PolyData face) was tried first but
    rejected by svMultiPhysics: nn.cpp's gnnb() requires a real face element
    (with a registered element type/shape functions) to compute the
    boundary normal, even for a face that's only ever used for a strong
    Dirichlet BC -- "could not be matched to a node" is the resulting error.
    So instead we constrain a small existing TRI6 patch (whole element, 6
    nodes) nearest each target angle; a few extra nodes near the intended
    angle also get the same axis-aligned component fixed, which is a minor,
    common, and harmless over-constraint (it does not fix any *other*
    component), not a physically-meaningful error, given the sole purpose is
    removing rigid-body modes.

    Together these 3 axis-aligned patches are equivalent (up to that minor
    over-constraint) to fixing the true tangential direction at each angle:
      theta=0   (R,0,z0):  fix u_y  -> removes dy + R*phi = 0
      theta=180 (-R,0,z0): fix u_y  -> removes dy - R*phi = 0  (combined with
                                        the above: dy=0, phi=0)
      theta=90  (0,R,z0):  fix u_x  -> removes dx (once phi=0 is already
                                        pinned by the other two)
    i.e. together they remove exactly the tube's 3 remaining rigid-body
    modes (x/y translation, rotation about the axis) once both end faces
    are axially (u_z) constrained.

    outer_tri_conn: (n_tri, 6) 0-based local node indices into `points`,
    i.e. the "outer" face's own TRI6 connectivity block.
    """
    outer_r = RO
    z_target = LENGTH / 2.0

    centroids = points[outer_tri_conn[:, :3]].mean(axis=1)  # use corners only

    configs = [
        (0.0, (0, 1, 0)),
        (90.0, (1, 0, 0)),
        (180.0, (0, 1, 0)),
    ]

    results = []
    used = set()
    for angle_deg, direction in configs:
        target_angle = np.radians(angle_deg)
        target_xyz = np.array(
            [outer_r * np.cos(target_angle), outer_r * np.sin(target_angle), z_target]
        )
        dists = np.linalg.norm(centroids - target_xyz, axis=1)
        order = np.argsort(dists)
        best = next(i for i in order if i not in used)
        used.add(best)
        results.append((outer_tri_conn[best], centroids[best], direction))
    return results


def main():
    run_gmsh()
    m = meshio.read(MSH_PATH)
    points = m.points

    tet_block = [c for c in m.cells if c.type == "tetra10"][0]
    tet_conn = tet_block.data  # 0-based local node indices into `points`

    os.makedirs(MESH_DIR, exist_ok=True)
    os.makedirs(SURF_DIR, exist_ok=True)
    write_volume_vtu(points, tet_conn, os.path.join(MESH_DIR, "mesh-complete.mesh.vtu"))

    print(f"Volume mesh: {len(points)} points, {len(tet_conn)} Tet10 elements.")

    # Face .vtp files: svMultiPhysics requires the FULL 6-node TRI6
    # connectivity per face (3 corners + 3 mid-edge nodes), with point
    # coordinates that are an exact match (down to full float precision) of
    # the volume mesh's own node coordinates -- node/element identification
    # in load_msh.cpp's read_sv() is done by hashing coordinates and sorted
    # global-node-ID sets against the volume mesh's own TET10 connectivity,
    # NOT by trusting whatever GlobalNodeID/GlobalElementID arrays the face
    # file itself provides. meshio's canonical "triangle6" node order
    # (corners 0,1,2 then mid-edges in edge order (0,1),(1,2),(2,0)) already
    # matches svMultiPhysics's expected TRI6 convention (verified earlier:
    # it matches VTK's quadratic-triangle edge convention), and since we
    # slice face points directly out of the SAME `points` array the volume
    # mesh was written from, coordinates are trivially exact matches --
    # no independent tessellation or reordering needed here.
    triangle_blocks = [c for c in m.cells if c.type == "triangle6"]

    face_node_ids = {}
    outer_conn = None
    for name, face_id in [("bottom", 1), ("top", 2), ("outer", 3), ("inner", 4)]:
        sets = m.cell_sets[name]
        conns = []
        for blk_idx, idxs in enumerate(sets):
            if len(idxs) == 0:
                continue
            block = m.cells[blk_idx]
            if block.type != "triangle6":
                continue
            conns.append(block.data[idxs])
        conn = np.concatenate(conns, axis=0)
        if name == "outer":
            outer_conn = conn
        # 1-based global node IDs, matching write_volume_vtu's GlobalNodeID.
        global_conn = conn + 1
        elem_ids = np.arange(1, len(conn) + 1)
        face_node_ids[name] = set(np.unique(global_conn).tolist())

        out_path = os.path.join(SURF_DIR, f"{name}.vtp")
        write_face_vtp(points, global_conn, elem_ids, face_id, out_path, nodes_per_face=6)
        print(f"Face '{name}': {len(conn)} TRI6 elements -> {out_path}")

    # Three single-TRI6-element "pin" faces on the outer surface at
    # theta=0/90/180 deg, near mid-height, to remove the tube's remaining
    # rigid-body modes (x/y translation, rotation about the axis) once both
    # end faces are axially (u_z) constrained -- see
    # pick_circumferential_pin_faces()'s docstring for the derivation and
    # for why a single isolated point doesn't work here. Each pin's
    # Dirichlet BC uses a plain axis-aligned <Effective_direction>, printed
    # below.
    pins = pick_circumferential_pin_faces(points, outer_conn)
    print("\nCircumferential pin faces (centroid, Effective_direction):")
    for i, (tri_conn, centroid, direction) in enumerate(pins):
        name = f"pin{i}"
        global_conn = (tri_conn + 1).reshape(1, 6)
        out_path = os.path.join(SURF_DIR, f"{name}.vtp")
        write_face_vtp(points, global_conn, np.array([1]), i + 5, out_path, nodes_per_face=6)
        print(f"  {name}: centroid {centroid} -> Effective_direction {direction} -> {out_path}")


if __name__ == "__main__":
    main()
