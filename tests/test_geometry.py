from pathlib import Path

import dolfin
from simcardems import geometry

here = Path(__file__).absolute().parent


def test_create_slab_geometry_normal():
    parameters = {"lx": 1, "ly": 1, "lz": 1, "dx": 1, "num_refinements": 1}
    geo = geometry.SlabGeometry(parameters=parameters)

    assert geo.mechanics_mesh.num_cells() == 6
    assert geo.ep_mesh.num_cells() == 48
    for k, v in parameters.items():
        assert geo.parameters[k] == v
    assert geo.num_refinements == 1


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


def test_load_geometry():
    mesh_folder = here / ".." / "demos" / "geometries"
    mesh_path = mesh_folder / "slab.h5"
    schema_path = mesh_folder / "slab.json"
    geo = geometry.load_geometry(mesh_path=mesh_path, schema_path=schema_path)
    assert geo.mesh.num_cells() == 202
    assert geo.ep_mesh.num_cells() == 1616
    assert isinstance(geo, geometry.SlabGeometry)


def test_dump_geometry(tmp_path):
    mesh_folder = here / ".." / "demos" / "geometries"
    mesh_path = mesh_folder / "slab.h5"
    schema_path = mesh_folder / "slab.json"
    geo = geometry.load_geometry(mesh_path=mesh_path, schema_path=schema_path)
    outpath = tmp_path / "state.h5"
    geo.dump(outpath)

    dumped_geo = geometry.load_geometry(
        mesh_path=outpath,
        schema_path=outpath.with_suffix(".json"),
    )
    assert dumped_geo == geo
