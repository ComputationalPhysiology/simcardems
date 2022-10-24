import dolfin
import numpy as np
import pulse
import pytest
import simcardems


@pytest.mark.xfail
@pytest.mark.parametrize(
    "pre_stretch, traction, spring, fix_right_plane, expected_u",
    [
        (None, None, None, True, (0, 0, 0)),
        (None, None, None, False, (0, 0, 0)),
        (None, -10.0, None, True, (0, 0, 0)),
        (None, -10.0, None, False, (0.21933731, -0.04719835, -0.04719835)),
        (0.1, None, None, True, (0.1, -0.02326871, -0.02326871)),
        (0.1, None, None, False, (0.1, -0.02326871, -0.02326871)),
        (None, None, 100, False, (0, 0, 0)),
    ],
)
def test_boundary_conditions(
    pre_stretch,
    traction,
    spring,
    fix_right_plane,
    expected_u,
):
    mesh = dolfin.UnitCubeMesh(2, 2, 2)

    bcs, marker_functions = simcardems.mechanics_model.setup_diriclet_bc(
        mesh,
        pre_stretch=pre_stretch,
        traction=traction,
        spring=spring,
        fix_right_plane=fix_right_plane,
    )
    microstructure = simcardems.mechanics_model.setup_microstructure(mesh)
    geometry = pulse.Geometry(
        mesh=mesh,
        microstructure=microstructure,
        marker_functions=marker_functions,
    )
    material = pulse.NeoHookean()
    problem = pulse.MechanicsProblem(
        geometry,
        material,
        bcs,
        solver_parameters={"linear_solver": "mumps"},
    )
    problem.solve()

    if traction is not None:
        pulse.iterate.iterate(problem, problem.bcs.neumann[0].traction, traction)

    U, P = problem.state.split(deepcopy=True)

    print(U(1.0, 0.5, 0.5))
    assert np.isclose(U(1.0, 0.5, 0.5), expected_u, atol=1e-5).all()


@pytest.mark.xfail
@pytest.mark.parametrize(
    "pre_stretch, traction, spring, fix_right_plane, expected_u",
    [
        (None, None, None, True, (0, 0, 0)),
        (None, None, None, False, (-0.15656733, 0.04443344, 0.04443344)),
        (None, -10.0, None, True, (0.0, 0, 0)),
        (None, -10.0, None, False, (0.0, 0.0, 0.0)),
        (0.1, None, None, True, (0.1, -0.02326871, -0.02326871)),
        (0.1, None, None, False, (0.1, -0.02326871, -0.02326871)),
        (None, None, 100, True, (0, 0, 0)),
        (None, None, 100, False, (-0.0615345, 0.00249483, 0.00249483)),
    ],
)
def test_boundary_conditions_activation(
    pre_stretch,
    traction,
    spring,
    fix_right_plane,
    expected_u,
):
    mesh = dolfin.UnitCubeMesh(2, 2, 2)

    bcs, marker_functions = simcardems.mechanics_model.setup_diriclet_bc(
        mesh,
        pre_stretch=pre_stretch,
        traction=traction,
        spring=spring,
        fix_right_plane=fix_right_plane,
    )
    microstructure = simcardems.mechanics_model.setup_microstructure(mesh)
    geometry = pulse.Geometry(
        mesh=mesh,
        microstructure=microstructure,
        marker_functions=marker_functions,
    )
    Ta = dolfin.Constant(0.0)
    material = pulse.NeoHookean(activation=Ta, active_model="active_stress")
    problem = pulse.MechanicsProblem(
        geometry,
        material,
        bcs,
        solver_parameters={"linear_solver": "mumps"},
    )
    problem.solve()

    if traction is not None:
        pulse.iterate.iterate(problem, problem.bcs.neumann[0].traction, traction)

    pulse.iterate.iterate(problem, Ta, 10.0)

    U, P = problem.state.split(deepcopy=True)
    print(U(1.0, 0.5, 0.5))
    assert np.isclose(U(1.0, 0.5, 0.5), expected_u, rtol=1e-5).all()


if __name__ == "__main__":
    test_boundary_conditions_activation(None, None, 100, True, None)
