import math

import simcardems


def test_SlabBoundaryConditions():
    geo = simcardems.geometry.SlabGeometry(lx=1, ly=1, lz=1, dx=1)
    bc = simcardems.boundary_conditions.SlabBoundaryConditions(
        geo=geo,
        pre_stretch=0.1,
        traction=1.0,
        spring=0.2,
        fix_right_plane=False,
    )
    bcs = bc.bcs

    assert len(bcs.neumann) == 1
    assert math.isclose(float(bc.bcs.neumann[0].traction), 1.0)

    assert len(bcs.dirichlet) == 1
    assert callable(bcs.dirichlet[0])

    assert len(bcs.robin) == 1
    assert math.isclose(float(bcs.robin[0].value), 0.2)
