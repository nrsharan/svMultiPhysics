"""Results saved to one XDMF/HDF5 pair of files (GeneralSimulationParameters
Save_results_in_XDMF_format) hold the same data as the VTU files of the same
simulation, and a continued simulation adds to the files it continues."""

import glob
import math
import os
import re
import shutil
import subprocess

import meshio
import numpy as np
import pytest

from .conftest import OVERSUBSCRIBE_FLAG, cpp_exec, skip_if_no_hdf5

# meshio reads the HDF5 data of an XDMF file with h5py.
h5py = pytest.importorskip("h5py")

cases_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")


def write_input(case, name, values):
    """Write the case's solver.xml as <name>, with the given
    GeneralSimulationParameters elements set (added if missing)."""
    with open(os.path.join(case, "solver.xml")) as f:
        text = f.read()

    for key, value in values.items():
        element = "<{0}> {1} </{0}>".format(key, value)
        pattern = re.compile(r"<{0}>.*?</{0}>".format(key), re.S)
        if pattern.search(text):
            text = pattern.sub(lambda match: element, text, count=1)
        else:
            text = text.replace(
                "<GeneralSimulationParameters>",
                "<GeneralSimulationParameters>\n  " + element,
                1,
            )

    path = os.path.join(case, name)
    with open(path, "w") as f:
        f.write(text)
    return path


def simulate(case, results, values, n_proc=1):
    """Run the case with the given GeneralSimulationParameters, writing the
    results and restart files to the folder `results`."""
    values = dict(values, Save_results_in_folder=results)
    input_file = write_input(case, "xdmf_test_" + os.path.basename(results) + ".xml", values)

    cmd = " ".join(
        [
            "mpirun",
            OVERSUBSCRIBE_FLAG if n_proc > 1 else "",
            "-np",
            str(n_proc),
            cpp_exec,
            os.path.basename(input_file),
        ]
    )

    try:
        completed = subprocess.run(cmd, cwd=case, shell=True, stderr=subprocess.PIPE, text=True)
    finally:
        os.remove(input_file)

    if completed.returncode != 0:
        raise RuntimeError("Exit code {}: {}\n".format(completed.returncode, completed.stderr))


def vtu_results(folder):
    files = glob.glob(os.path.join(folder, "result_*.vtu"))
    files.sort(key=lambda f: int(re.search(r"_(\d+)\.vtu$", f).group(1)))
    return [meshio.read(f) for f in files]


def xdmf_results(folder):
    with meshio.xdmf.TimeSeriesReader(os.path.join(folder, "result.xdmf")) as reader:
        points, cells = reader.read_points_cells()
        steps = [reader.read_data(k) for k in range(reader.num_steps)]
    return points, cells, steps


def saved_steps(folder):
    with h5py.File(os.path.join(folder, "result.h5"), "r") as f:
        return sorted(int(step) for step in f["Step"])


def assert_same_results(vtu_folder, xdmf_folder):
    vtus = vtu_results(vtu_folder)
    points, cells, steps = xdmf_results(xdmf_folder)

    assert len(vtus) > 1
    assert len(steps) == len(vtus)

    first = vtus[0]
    assert np.array_equal(points, first.points)
    assert [block.type for block in cells] == [block.type for block in first.cells]
    for a, b in zip(cells, first.cells):
        assert np.array_equal(a.data, b.data)

    # The cell data that only the first VTU file holds (Domain_ID, Proc_ID)
    # are in every XDMF step.
    cell_data = {}

    for vtu, (time, xdmf_point_data, xdmf_cell_data) in zip(vtus, steps):
        assert time == vtu.field_data["TimeValue"][0]

        assert sorted(xdmf_point_data) == sorted(vtu.point_data)
        for name, values in vtu.point_data.items():
            assert np.array_equal(xdmf_point_data[name], values), name

        cell_data.update(vtu.cell_data)
        assert sorted(xdmf_cell_data) == sorted(cell_data)
        for name, blocks in cell_data.items():
            for a, b in zip(xdmf_cell_data[name], blocks):
                assert np.array_equal(a, b), name


SAVE_EVERY_STEP = {
    "Number_of_time_steps": 3,
    "Increment_in_saving_VTK_files": 1,
    "Start_saving_after_time_step": 1,
    "Name_prefix_of_saved_VTK_files": "result",
}


@skip_if_no_hdf5
@pytest.mark.parametrize(
    "base_folder, test_folder, n_proc",
    [
        ("fluid", "driven_cavity_2d", 1),  # 2D: triangles, 2-component vectors
        ("struct", "block_compression", 2),  # stresses, cell data, two processes
        ("fluid", "quadratic_tet10", 1),  # quadratic tetrahedra
    ],
)
def test_xdmf_same_as_vtu(base_folder, test_folder, n_proc, tmp_path):
    case = os.path.join(cases_dir, base_folder, test_folder)
    vtu = str(tmp_path / "vtu")
    xdmf = str(tmp_path / "xdmf")

    simulate(case, vtu, SAVE_EVERY_STEP, n_proc)
    simulate(case, xdmf, dict(SAVE_EVERY_STEP, Save_results_in_XDMF_format="true"), n_proc)

    assert not glob.glob(os.path.join(xdmf, "*.vtu"))
    assert_same_results(vtu, xdmf)


@skip_if_no_hdf5
def test_xdmf_continued_simulation(tmp_path):
    case = os.path.join(cases_dir, "fluid", "driven_cavity_2d")
    folder = str(tmp_path / "run")
    values = dict(
        SAVE_EVERY_STEP,
        Number_of_time_steps=4,
        Save_results_in_XDMF_format="true",
        Increment_in_saving_restart_files=1,
    )

    simulate(case, folder, values)
    assert saved_steps(folder) == [1, 2, 3, 4]

    # Continuing from the last step (4) up to step 6 adds steps 5 and 6.
    simulate(case, folder, dict(values, Number_of_time_steps=6, Continue_previous_simulation="true"))
    assert saved_steps(folder) == [1, 2, 3, 4, 5, 6]

    # Continuing from step 2 up to step 3 replaces steps 3 to 6 by the new step 3.
    shutil.copy(os.path.join(folder, "stFile_002.bin"), os.path.join(folder, "stFile_last.bin"))
    simulate(case, folder, dict(values, Number_of_time_steps=3, Continue_previous_simulation="true"))
    assert saved_steps(folder) == [1, 2, 3]

    # The XDMF file lists the same steps, and the unchanged mesh is written once.
    _, _, steps = xdmf_results(folder)
    times = [time for time, _, _ in steps]
    assert len(times) == 3
    assert all(math.isclose(t, times[0] * (k + 1)) for k, t in enumerate(times))

    with h5py.File(os.path.join(folder, "result.h5"), "r") as f:
        assert len(f["Mesh"]) == 1
