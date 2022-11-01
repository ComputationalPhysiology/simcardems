import math

import simcardems


def test_SlabBoundaryConditions(geo):
    bcs = simcardems.boundary_conditions.create_slab_boundary_conditions(
        geo=geo,
        pre_stretch=0.1,
        traction=1.0,
        spring=0.2,
        fix_right_plane=False,
    )

    assert len(bcs.neumann) == 1
    assert math.isclose(float(bcs.neumann[0].traction), 1.0)

    assert len(bcs.dirichlet) == 1
    assert callable(bcs.dirichlet[0])

    assert len(bcs.robin) == 1
    assert math.isclose(float(bcs.robin[0].value), 0.2)
