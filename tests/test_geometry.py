import json
import tempfile
from pathlib import Path

import dolfin
from simcardems import geometry


def test_create_slab_geometry_normal():
    parameters = {
        "lx": 1,
        "ly": 1,
        "lz": 1,
        "dx": 1,
    }
    geo = geometry.SlabGeometry(parameters=parameters, num_refinements=1)

    assert geo.mechanics_mesh.num_cells() == 6
    assert geo.ep_mesh.num_cells() == 48
    assert geo.parameters == parameters
    assert geo.num_refinements == 1


def test_create_slab_geometry_from_files():
    markers = geometry.SlabGeometry.markers
    parameters = {
        "lx": 1,
        "ly": 1,
        "lz": 1,
        "dx": 1,
    }
    mesh = geometry.create_boxmesh(lx=1, ly=1, lz=1, dx=1.0, refinements=0)
    ffun = geometry.create_slab_facet_function(mesh=mesh, lx=1.0, markers=markers)
    # breakpoint()
    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        mesh_path = tmpdir / "mesh.xdmf"
        ffun_path = tmpdir / "ffun.xdmf"
        marker_path = tmpdir / "markers.json"
        parameter_path = tmpdir / "info.json"
        with dolfin.XDMFFile(mesh_path.as_posix()) as f:
            f.write(mesh)
        with dolfin.XDMFFile(ffun_path.as_posix()) as f:
            f.write(ffun)
        marker_path.write_text(json.dumps(markers))
        parameter_path.write_text(json.dumps(parameters))

        geo = geometry.SlabGeometry.from_files(
            mesh_path=mesh_path,
            ffun_path=ffun_path,
            marker_path=marker_path,
            parameter_path=parameter_path,
            num_refinements=1,
        )

    assert geo.mechanics_mesh.num_cells() == 6
    assert geo.ep_mesh.num_cells() == 48
    assert geo.parameters == parameters
    assert (geo.ffun.array() == ffun.array()).all()


def test_create_slab_geometry_with_mechanics_mesh():
    parameters = {
        "lx": 1,
        "ly": 1,
        "lz": 1,
        "dx": 1,
    }
    mesh = dolfin.UnitCubeMesh(1, 1, 1)
    geo = geometry.SlabGeometry(
        parameters=parameters,
        num_refinements=1,
        mechanics_mesh=mesh,
    )

    assert geo.mechanics_mesh is mesh
    assert geo.ep_mesh.num_cells() == 48
    assert geo.parameters == {
        "lx": 1,
        "ly": 1,
        "lz": 1,
        "dx": 1,
    }
    assert geo.num_refinements == 1
