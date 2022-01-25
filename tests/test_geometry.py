import dolfin
from simcardems import geometry


def test_create_slab_geometry():
    geo = geometry.SlabGeometry(1, 1, 1, 1, 1)
    assert geo.mechanics_mesh.num_cells() == 6
    assert geo.ep_mesh.num_cells() == 48
    assert geo.parameters == {
        "lx": 1,
        "ly": 1,
        "lz": 1,
        "dx": 1,
        "num_refinements": 1,
    }


def test_create_slab_geometry_with_mechanics_mesh():
    mesh = dolfin.UnitCubeMesh(1, 1, 1)
    geo = geometry.SlabGeometry(1, 1, 1, 1, 1, mechanics_mesh=mesh)

    assert geo.mechanics_mesh is mesh
    assert geo.ep_mesh.num_cells() == 48
    assert geo.parameters == {
        "lx": 1,
        "ly": 1,
        "lz": 1,
        "dx": 1,
        "num_refinements": 1,
    }
