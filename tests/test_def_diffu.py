from .conftest import run_with_reference, skip_if_no_interface2, skip_if_no_trilinos

# Common folder for all tests in this file
base_folder = "def_diffu"

# Fields to test
fields = ["Displacement", "Concentration"]


@skip_if_no_interface2
def test_single_tet10():
    run_with_reference(base_folder, "single_tet10", fields, 1, 1)


# Pressure ramp on the hollow-cylinder wall with reorientation, growth (with
# the growth orientation initialized when it switches on), the active response
# and a concentration Dirichlet condition all switched on by time segments
# within ten steps. Exact linear solves (trilinos-amesos2), so the result does
# not depend on the number of processors.
@skip_if_no_interface2
@skip_if_no_trilinos
def test_hollow_cylinder_short(n_proc):
    run_with_reference(base_folder, "hollow_cylinder_short", fields, n_proc, 10)


# The same case with the wall split into two domains that carry identical
# parameters: must reproduce the single-domain reference.
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
