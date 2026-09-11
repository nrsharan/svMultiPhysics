#!/usr/bin/env python3
"""Write domains.vtu: the hollow_cylinder mesh with a DOMAIN_ID cell array
splitting the wall at mid-height (domain 1 below, domain 2 above).

Usage: python3 make_domains.py   (from this directory; needs meshio, numpy)
"""
import os

import meshio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.join(HERE, "..", "hollow_cylinder", "mesh", "mesh-complete.mesh.vtu")
OUT = os.path.join(HERE, "domains.vtu")


def main():
    mesh = meshio.read(MESH)
    cells = mesh.cells_dict["tetra10"]
    z_centroid = mesh.points[cells[:, :4], 2].mean(axis=1)
    z_mid = 0.5 * (mesh.points[:, 2].min() + mesh.points[:, 2].max())
    domain_id = np.where(z_centroid < z_mid, 1, 2).astype(np.int32)

    out = meshio.Mesh(mesh.points, [("tetra10", cells)], cell_data={"DOMAIN_ID": [domain_id]})
    meshio.write(OUT, out, binary=False)
    print(f"{OUT}: {len(cells)} elements, domain 1: {np.sum(domain_id == 1)}, domain 2: {np.sum(domain_id == 2)}")


if __name__ == "__main__":
    main()
