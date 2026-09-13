import os

import meshio

from .conftest import run_with_reference, skip_if_no_frosch, skip_if_no_interface2, skip_if_no_trilinos

# Common folder for all tests in this file
base_folder = "def_diffu"

# Post-processing quantities of the Interface2/AceGen element that the cases
# write (see their solver.xml): the principal Cauchy stresses and others, the
# Cauchy stress tensor, the fiber directions, the growth orientation vectors
# and the active stretches.
post_fields = [
    "MisesStress",
    "SCirc",
    "SAxial",
    "SRadial",
    "W",
    "PhiElastin",
    "PhiCollagen",
    "PhiSMC",
    "Stretch1",
    "Stretch2",
    "nC1",
    "nC2",
    "nD1",
    "nD2",
    "DetF",
    "DetFe",
    "DetFg",
    "S",
    "a1",
    "a2",
    "Ag1n",
    "Ag2n",
    "Ag3n",
    "ActiveStretch1",
    "ActiveStretch2",
]

# Fields to test
fields = ["Displacement", "Concentration"] + post_fields

# Every post-processing quantity of the element with its number of components:
# the element's names grouped into scalars, vectors (<P>1, <P>2, <P>3) and 3x3
# tensors (<P>xx, ..., <P>zz). The nodal weight "Volume" is not an output.
all_post_fields = {
    "S": 9, "MisesStress": 1, "SCirc": 1, "SAxial": 1, "SRadial": 1, "E": 9,
    "W": 1, "PhiElastin": 1, "PhiCollagen": 1, "PhiSMC": 1, "Stretch1": 1,
    "Stretch2": 1, "DetF": 1, "Ag1n": 3, "Ag2n": 3, "Ag3n": 3, "a1": 3, "a2": 3,
    "nA1": 1, "nA2": 1, "nB1": 1, "nB2": 1, "nC1": 1, "nC2": 1, "nD1": 1,
    "nD2": 1, "k161": 1, "k162": 1, "k251": 1, "k252": 1, "Ca1": 1, "Ca2": 1,
    "ActiveStretch1": 1, "ActiveStretch2": 1, "ScDir": 3, "SaDir": 3,
    "SrDir": 3, "DetFg": 1, "DetFe": 1,
}


# One element; writes all post-processing quantities (<Element_post_data>).
@skip_if_no_interface2
def test_single_tet10():
    run_with_reference(base_folder, "single_tet10", fields, 1, 1)

    res = meshio.read(os.path.join("cases", base_folder, "single_tet10", "1-procs", "result_001.vtu"))
    components = {name: 1 if a.ndim == 1 else a.shape[1] for name, a in res.point_data.items()}
    assert components == {"Displacement": 3, "Concentration": 1, **all_post_fields}


# Pressure ramp on the hollow-cylinder wall with reorientation, growth (with
# the growth orientation initialized when it switches on), the active response
# and a concentration Dirichlet condition all switched on by time segments
# within ten steps. Exact linear solves (trilinos-amesos2), so the result,
# including the post-processing quantities summed over the processors at
# shared nodes, does not depend on the number of processors.
@skip_if_no_interface2
@skip_if_no_trilinos
def test_hollow_cylinder_short(n_proc):
    run_with_reference(base_folder, "hollow_cylinder_short", fields, n_proc, 10)


# The same case with the wall split into two domains that carry identical
# parameters: must reproduce the single-domain reference, including the
# post-processing quantities at nodes shared by the two domains.
@skip_if_no_interface2
@skip_if_no_trilinos
def test_hollow_cylinder_short_two_domains(n_proc):
    run_with_reference(
        base_folder,
        "hollow_cylinder_short_two_domains",
        fields,
        n_proc,
        10,
        name_ref="../hollow_cylinder_short/result_010.vtu",
    )


# The same case solved iteratively with the FROSch two-level overlapping
# Schwarz preconditioner (trilinos-frosch; needs a Trilinos with
# ShyLU_DDFROSch): with the linear solves converged to 1e-10 it must
# reproduce the direct-solver reference on any number of processors.
@skip_if_no_interface2
@skip_if_no_frosch
def test_hollow_cylinder_short_frosch(n_proc):
    run_with_reference(
        base_folder,
        "hollow_cylinder_short_frosch",
        fields,
        n_proc,
        10,
        name_ref="../hollow_cylinder_short/result_010.vtu",
    )


# The same with the two-block FROSch preconditioner (trilinos-frosch-block:
# displacement with rotations, concentration), which works on a renumbered
# copy of the matrix.
@skip_if_no_interface2
@skip_if_no_frosch
def test_hollow_cylinder_short_frosch_block(n_proc):
    run_with_reference(
        base_folder,
        "hollow_cylinder_short_frosch_block",
        fields,
        n_proc,
        10,
        name_ref="../hollow_cylinder_short/result_010.vtu",
    )


# The smooth-muscle element (<CCBActiveGandR>) on the hollow_cylinder_short
# wall and loads, with its own parameter set: reorientation, growth (growth
# orientation initialized), the active response (active stretches
# initialized) and a rate acceleration within ten steps.
smc_fields = [
    "Displacement",
    "Concentration",
    "MisesStress",
    "SCirc",
    "SAxial",
    "SRadial",
    "W",
    "Growth",
    "Stretch1",
    "Stretch2",
    "nC1",
    "nC2",
    "nD1",
    "nD2",
    "DetF",
    "DetFe",
    "DetFg",
    "S",
    "a1",
    "a2",
    "Ag1n",
    "Ag2n",
    "Ag3n",
]


@skip_if_no_interface2
@skip_if_no_trilinos
def test_hollow_cylinder_short_smc(n_proc):
    run_with_reference(base_folder, "hollow_cylinder_short_smc", smc_fields, n_proc, 10)
