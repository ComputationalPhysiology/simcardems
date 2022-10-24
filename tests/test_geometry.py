import json
import tempfile
from pathlib import Path

import dolfin
import pytest
from simcardems import geometry


def test_create_slab_geometry_normal():
    parameters = {"lx": 1, "ly": 1, "lz": 1, "dx": 1, "num_refinements": 1}
    geo = geometry.SlabGeometry(parameters=parameters)

    assert geo.mechanics_mesh.num_cells() == 6
    assert geo.ep_mesh.num_cells() == 48
    for k, v in parameters.items():
        assert geo.parameters[k] == v
    assert geo.num_refinements == 1


@pytest.mark.parametrize(
    "include_ffun, include_marker, include_parameter",
    (
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ),
)
def test_create_slab_geometry_from_files(
    include_ffun,
    include_marker,
    include_parameter,
):
    markers = geometry.SlabGeometry.default_markers()
    parameters = {"lx": 1, "ly": 1, "lz": 1, "dx": 1, "num_refinements": 1}
    mesh = geometry.create_boxmesh(lx=1, ly=1, lz=1, dx=1.0, refinements=0)
    ffun = geometry.create_slab_facet_function(mesh=mesh, lx=1.0, markers=markers)

    ffun_path = None
    parameter_path = None
    marker_path = None

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        mesh_path = tmpdir / "mesh.xdmf"
        with dolfin.XDMFFile(mesh_path.as_posix()) as f:
            f.write(mesh)

        if include_ffun:
            ffun_path = tmpdir / "ffun.xdmf"
            with dolfin.XDMFFile(ffun_path.as_posix()) as f:
                f.write(ffun)

        if include_marker:
            marker_path = tmpdir / "markers.json"
            marker_path.write_text(json.dumps(markers))

        if include_parameter:
            parameter_path = tmpdir / "info.json"
            parameter_path.write_text(json.dumps(parameters))

        geo = geometry.SlabGeometry.from_files(
            mesh_path=mesh_path,
            ffun_path=ffun_path,
            marker_path=marker_path,
            parameter_path=parameter_path,
        )

    assert geo.mechanics_mesh.num_cells() == 6
    assert geo.ep_mesh.num_cells() == 48
    if include_parameter:
        for k, v in parameters.items():
            assert geo.parameters[k] == v
    if include_ffun:
        assert (geo.ffun.array() == ffun.array()).all()


def test_create_slab_geometry_with_mechanics_mesh():

    parameters = {"lx": 1, "ly": 1, "lz": 1, "dx": 1, "num_refinements": 1}
    mesh = dolfin.UnitCubeMesh(1, 1, 1)
    geo = geometry.SlabGeometry(
        parameters=parameters,
        mechanics_mesh=mesh,
    )

    assert geo.mechanics_mesh is mesh
    assert geo.ep_mesh.num_cells() == 48
    for k, v in parameters.items():
        assert geo.parameters[k] == v
    assert geo.num_refinements == 1
