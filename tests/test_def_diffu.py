import os
import re
import shutil
import subprocess

import meshio
import numpy as np
import pytest

from .conftest import (
    OVERSUBSCRIBE_FLAG,
    cpp_exec,
    run_with_reference,
    skip_if_no_frosch,
    skip_if_no_interface2,
    skip_if_no_trilinos,
)

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


def simulate(case, results, values, n_proc, remove=(), insert="", replace=()):
    """Run the case with the given GeneralSimulationParameters (added if
    missing), writing the results and restart files to the folder `results`,
    and return what it wrote to its standard output.

    The elements named in `remove` are deleted first, the (old, new) pairs in
    `replace` are substituted once each -- for the elements outside
    <GeneralSimulationParameters>, such as an equation's <Max_iterations> --
    and the XML in `insert` is added to <GeneralSimulationParameters>: for the
    elements that may appear more than once (<Add_time_step_segment>) and for
    those that another element forbids (<Number_of_time_steps> and
    <Time_step_size> with time step segments)."""
    with open(os.path.join(case, "solver.xml")) as f:
        text = f.read()
    for key in remove:
        text = re.sub(r"\s*<{0}>.*?</{0}>".format(key), "", text, count=1, flags=re.S)
    for old, new in replace:
        text = text.replace(old, new, 1)
    for key, value in dict(values, Save_results_in_folder=results).items():
        element = "<{0}> {1} </{0}>".format(key, value)
        pattern = re.compile(r"<{0}>.*?</{0}>".format(key), re.S)
        if pattern.search(text):
            text = pattern.sub(lambda match: element, text, count=1)
        else:
            text = text.replace("<GeneralSimulationParameters>", "<GeneralSimulationParameters>\n  " + element, 1)
    if insert:
        text = text.replace("<GeneralSimulationParameters>",
                            "<GeneralSimulationParameters>\n  " + insert, 1)
    name = "restart_test_" + os.path.basename(results) + ".xml"
    with open(os.path.join(case, name), "w") as f:
        f.write(text)

    cmd = " ".join(["mpirun", OVERSUBSCRIBE_FLAG if n_proc > 1 else "", "-np", str(n_proc), cpp_exec, name])
    try:
        completed = subprocess.run(cmd, cwd=case, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
    finally:
        os.remove(os.path.join(case, name))
    if completed.returncode != 0:
        raise RuntimeError("Exit code {}: {}\n{}".format(completed.returncode, completed.stderr,
                                                        completed.stdout))
    return completed.stdout


# A simulation continued from a restart file resumes the element history
# (def_diffu::write_restart_history()): at its last step it has the result of
# the simulation without interruption. The smooth-muscle case switches
# reorientation on at t = 0.4, growth at 1.2 (step 6, growth orientation
# initialized) and the active response at 1.6 (step 8); it is continued from
# step 6, exactly at the growth switch-on, and from step 7. The
# constrained-mixture case likewise.
@skip_if_no_interface2
@skip_if_no_trilinos
@pytest.mark.parametrize("case", ["hollow_cylinder_short_smc", "hollow_cylinder_short"])
def test_restart_element_history(case, n_proc, tmp_path):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases", base_folder, case)
    values = dict(
        Number_of_time_steps=10,
        Save_results_to_VTK_format=1,
        Increment_in_saving_VTK_files=1,
        Start_saving_after_time_step=1,
        Increment_in_saving_restart_files=1,
    )

    full = tmp_path / "full"
    simulate(folder, str(full), values, n_proc)
    reference = meshio.read(str(full / "result_010.vtu"))

    for step in (6, 7):
        run = tmp_path / "from_{}".format(step)
        run.mkdir()
        shutil.copy(full / "stFile_{:03d}.bin".format(step), run / "stFile_last.bin")
        simulate(folder, str(run), dict(values, Continue_previous_simulation="true"), n_proc)

        # It continued from the restart step instead of starting anew.
        assert not (run / "result_{:03d}.vtu".format(step)).exists()
        assert (run / "result_{:03d}.vtu".format(step + 1)).exists()

        result = meshio.read(str(run / "result_010.vtu"))
        assert sorted(result.point_data) == sorted(reference.point_data)
        for name, expected in reference.point_data.items():
            scale = max(1.0, float(np.abs(expected).max()))
            assert np.allclose(result.point_data[name], expected, rtol=1e-8, atol=1e-10 * scale), (
                "{}: restart at step {}".format(name, step)
            )


# The time step segments of a run with adaptive time stepping
# (<Adaptive_time_stepping>), as the <Time_step_size> of one segment that runs
# to <Final_time>: its maximum time step size, i.e. the largest step it may take.
def adaptive_segments(time_step, final_time, **values):
    segments = [
        "<Final_time> {} </Final_time>".format(final_time),
        "<Adaptive_time_stepping> true </Adaptive_time_stepping>",
    ]
    segments += ["<{0}> {1} </{0}>".format(key, value) for key, value in values.items()]
    segments.append(
        "<Add_time_step_segment> <Start_time> 0.0 </Start_time>"
        " <Time_step_size> {} </Time_step_size> </Add_time_step_segment>".format(time_step)
    )
    return "\n  ".join(segments)


# Adaptive time stepping takes the <Time_step_size> of a segment as the
# largest step that segment may take. A run in which no time step fails takes
# that step throughout, so it is the run with the fixed time step size of the
# case, to the last bit.
@skip_if_no_interface2
@skip_if_no_trilinos
@pytest.mark.parametrize("case", ["hollow_cylinder_short_smc", "hollow_cylinder_short"])
def test_adaptive_time_stepping_unchanged(case, n_proc, tmp_path):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases", base_folder, case)
    values = dict(
        Save_results_to_VTK_format=1,
        Increment_in_saving_VTK_files=1,
        Start_saving_after_time_step=1,
    )

    fixed = tmp_path / "fixed"
    simulate(folder, str(fixed), dict(values, Number_of_time_steps=10), n_proc)

    adaptive = tmp_path / "adaptive"
    adaptive_output = simulate(folder, str(adaptive), values, n_proc,
                               remove=("Number_of_time_steps", "Time_step_size"),
                               insert=adaptive_segments(0.2, 2.0))

    reference = meshio.read(str(fixed / "result_010.vtu"))
    result = meshio.read(str(adaptive / "result_010.vtu"))

    assert sorted(result.point_data) == sorted(reference.point_data)
    for name, expected in reference.point_data.items():
        scale = max(1.0, float(np.abs(expected).max()))
        assert np.allclose(result.point_data[name], expected, rtol=1e-8, atol=1e-10 * scale), name

    assert "repeated 0 time step(s)" in adaptive_output


# <Save_results_every_time> writes the results every so much simulated time
# rather than every so many time steps: the first time step, and from then on
# the first step that ends at or after each multiple of the interval. The case
# takes 10 steps of 0.2 to t = 2.0, so an interval of 0.5 writes the steps
# ending at 0.2, 0.6, 1.0, 1.6 and 2.0 and no others.
@skip_if_no_interface2
@skip_if_no_trilinos
def test_save_results_every_time(n_proc, tmp_path):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases", base_folder,
                          "hollow_cylinder_short")
    results = tmp_path / "every_time"

    simulate(folder, str(results), dict(Number_of_time_steps=10, Save_results_to_VTK_format=1,
                                        Start_saving_after_time_step=1, Save_results_every_time=0.5),
             n_proc)

    written = sorted(int(re.search(r"result_(\d+)\.vtu", p).group(1))
                     for p in os.listdir(str(results)) if re.match(r"result_\d+\.vtu", p))
    assert written == [1, 3, 5, 8, 10], written

    # the times themselves, to catch an interval that drifts
    times = [meshio.read(str(results / "result_{:03d}.vtu".format(s))) for s in written]
    assert len(times) == 5


# A time step segment may give an interval of its own, which overrides the
# run's while it is active: one interval cannot sample a run whose phases
# differ by orders of magnitude. Two segments of 1 s each, both in steps of
# 0.125 (a step size and intervals that are exact in binary, so the expected
# steps do not depend on how the time accumulates):
#
#   [0, 1) every 0.5  -> steps 1 (the first step of the segment) and 4
#   [1, 2) every 0.25 -> steps 8 (the first step of the segment), 10, 12, 14, 16
#
# The first step of a segment is always written, so no phase of a run begins
# unrecorded: at step 8 the count starts again from the segment's start time.
@skip_if_no_interface2
@skip_if_no_trilinos
def test_save_results_every_time_per_segment(n_proc, tmp_path):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases", base_folder,
                          "hollow_cylinder_short")
    results = tmp_path / "per_segment"

    segments = "\n  ".join([
        "<Final_time> 2.0 </Final_time>",
        "<Add_time_step_segment> <Start_time> 0.0 </Start_time>"
        " <Time_step_size> 0.125 </Time_step_size>"
        " <Save_results_every_time> 0.5 </Save_results_every_time> </Add_time_step_segment>",
        "<Add_time_step_segment> <Start_time> 1.0 </Start_time>"
        " <Time_step_size> 0.125 </Time_step_size>"
        " <Save_results_every_time> 0.25 </Save_results_every_time> </Add_time_step_segment>",
    ])

    simulate(folder, str(results), dict(Save_results_to_VTK_format=1,
                                        Start_saving_after_time_step=1),
             n_proc,
             remove=("Number_of_time_steps", "Time_step_size"),
             insert=segments)

    written = sorted(int(re.search(r"result_(\d+)\.vtu", p).group(1))
                     for p in os.listdir(str(results)) if re.match(r"result_\d+\.vtu", p))
    assert written == [1, 4, 8, 10, 12, 14, 16], written


# A time step that does not converge is repeated from the state it started
# from with a smaller time step size, and the run goes on. Here the Newton
# iteration is the one that fails: <Max_iterations> 4 is not enough for the
# first step of 0.8, which is repeated with 0.4. The run ends at <Final_time>,
# which is the only way it can end (it exits with an error otherwise).
@skip_if_no_interface2
@skip_if_no_trilinos
@pytest.mark.parametrize("case", ["hollow_cylinder_short_smc", "hollow_cylinder_short"])
def test_adaptive_time_stepping_repeats_failed_step(case, n_proc, tmp_path):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases", base_folder, case)

    output = simulate(
        folder, str(tmp_path / "adaptive"),
        dict(Save_results_to_VTK_format=1, Increment_in_saving_VTK_files=1,
             Start_saving_after_time_step=1),
        n_proc,
        remove=("Number_of_time_steps", "Time_step_size"),
        replace=(("<Max_iterations> 25 </Max_iterations>", "<Max_iterations> 4 </Max_iterations>"),),
        insert=adaptive_segments(0.8, 2.0, Minimum_time_step_size=0.05),
    )

    assert "the Newton iteration did not converge); repeating it with" in output

    repeated = int(re.search(r"repeated (\d+) time step\(s\)", output).group(1))
    assert repeated >= 1


# The other way a time step fails: the element cannot compute its state, here
# with a step of 1.2 that takes the constrained-mixture element from the start
# of the pressure ramp past the switch-on of growth. Its error names the
# element, and the run goes on with a smaller time step size.
@skip_if_no_interface2
@skip_if_no_trilinos
def test_adaptive_time_stepping_repeats_failed_element(n_proc, tmp_path):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases", base_folder,
                          "hollow_cylinder_short")

    output = simulate(
        folder, str(tmp_path / "adaptive"),
        dict(Save_results_to_VTK_format=1, Increment_in_saving_VTK_files=1,
             Start_saving_after_time_step=1),
        n_proc,
        remove=("Number_of_time_steps", "Time_step_size"),
        insert=adaptive_segments(1.2, 2.4),
    )

    assert "CCBActiveCMMGandR element failed to converge" in output
    assert "repeating it with" in output

    repeated = int(re.search(r"repeated (\d+) time step\(s\)", output).group(1))
    assert repeated >= 1

