import dolfin
import numpy as np
import pytest
from simcardems import cell_model
from simcardems.cell_model import Parameter


def test_parameter_value_no_factor():
    value = 0.4
    p = Parameter("g_CaL", value)
    assert np.isclose(p.factor, 1.0)
    assert np.isclose(p.value, value)


def test_parameter_value_factor():
    value = 0.4
    drug_factor = 0.1
    popu_factor = 0.3
    p = Parameter("g_CaL", value, factors={"drug": drug_factor, "popu": popu_factor})
    assert np.isclose(p.factor, drug_factor * popu_factor)
    assert np.isclose(p.value, value * drug_factor * popu_factor)


def test_parameter_value_add_factor():
    value = 0.4
    drug_factor = 0.1
    popu_factor = 0.3
    p = Parameter("g_CaL", value, factors={"drug": drug_factor, "popu": popu_factor})
    hf_factor = 42.0
    p.add_factor("HF", hf_factor)
    assert np.isclose(p.factor, drug_factor * popu_factor * hf_factor)
    assert np.isclose(p.value, value * drug_factor * popu_factor * hf_factor)


def test_parameter_factors():

    value = 0.4
    drug_factor = 0.1
    popu_factor = 0.3
    p = Parameter("g_CaL", value, factors={"drug": drug_factor, "popu": popu_factor})
    hf_factor = 42.0
    p.add_factor("HF", hf_factor)
    factors = p.factors()
    expected_factors = {"drug": 0.1, "popu": 0.3, "HF": 42.0}
    assert set(factors.keys()) == set(expected_factors.keys())
    for k, v in factors.items():
        assert np.isclose(v, expected_factors[k])


def test_parameter_multiply():

    p1 = Parameter("g_CaL", 2.0, factors={"drug": 2.0, "popu": 2.0}).value
    p2 = Parameter("g_Na", 3.0, factors={"drug": 3.0, "popu": 3.0}).value
    assert np.isclose(p1 * p2, 8 * 27)
    assert np.isclose(p1 * 27, 8 * 27)
    assert np.isclose(27 * p1, 27 * 8)


def test_parameter_add():

    p1 = Parameter("g_CaL", 2.0, factors={"drug": 2.0, "popu": 2.0}).value
    p2 = Parameter("g_Na", 3.0, factors={"drug": 3.0, "popu": 3.0}).value
    assert np.isclose(p1 + p2, 8 + 27)
    assert np.isclose(p1 + 27, 8 + 27)
    assert np.isclose(27 + p1, 27 + 8)


def test_dict():

    p1 = Parameter("g_CaL", 2.0, factors={"drug": 2.0, "popu": 2.0})
    p2 = Parameter("g_Na", 3.0, factors={"drug": 3.0, "popu": 3.0})

    d = dict(map(cell_model.tuplize, [p1, p2]))
    assert len(d) == 2
    assert p1.name in d
    assert p2.name in d
    assert d[p1.name] == p1
    assert d[p2.name] == p2


def test_apply_custom_parameters():
    custom_parameters = {
        "GNa": {
            "value": 24.0,
            "factors": {
                "drug": 0.1,
                "popu": 2.0,
            },
        },
        "Gto": {  # Use default value
            "factors": {
                "drug": 0.2,
                "popu": 1.5,
            },
        },
        "kws": {
            "value": 0.015,
        },
    }

    parameters = cell_model.ORdmm_Land.default_parameters()
    new_parameters = cell_model.apply_custom_parameters(parameters, custom_parameters)
    assert parameters["GNa"] != new_parameters["GNa"]
    assert parameters["Gto"] != new_parameters["Gto"]
    assert parameters["kws"] != new_parameters["kws"]
    # This parameter has not been change
    assert parameters["kuw"] == new_parameters["kuw"]

    assert (
        new_parameters["GNa"]
        == custom_parameters["GNa"]["value"]
        * custom_parameters["GNa"]["factors"]["drug"]
        * custom_parameters["GNa"]["factors"]["popu"]
    )

    assert (
        new_parameters["Gto"]
        == parameters["Gto"].value
        * custom_parameters["Gto"]["factors"]["drug"]
        * custom_parameters["Gto"]["factors"]["popu"]
    )

    assert new_parameters["kws"] == custom_parameters["kws"]["value"]


def test_HF_scaling():
    parameters = cell_model.ORdmm_Land.default_parameters()
    hf_parameters = cell_model.apply_scaling(parameters, "hf")
    # Just assert one changed parameter
    assert np.isclose(hf_parameters["scale_Jup"].value, 0.45)
    assert "HF" in hf_parameters["scale_Jup"].factors()


def test_invalid_scaling_factors():
    parameters = cell_model.ORdmm_Land.default_parameters()
    with pytest.raises(cell_model.InvalidScalingFactorError):
        cell_model.apply_scaling(parameters, "bad_name")


def test_compare_old_model():
    """This test is simply comparing the old model
    implementation with the new one. We should probably
    change this test in the future when we remove the old model"""
    model = cell_model.ORdmm_Land()
    from simcardems.ORdmm_Land import ORdmm_Land as ORdmm_Land_old

    old_model = ORdmm_Land_old()

    assert model.num_states() == old_model.num_states() == 48

    mesh = dolfin.UnitSquareMesh(3, 3)
    V = dolfin.VectorFunctionSpace(mesh, "R", 0, dim=model.num_states() + 1)
    f = dolfin.Function(V)
    f.assign(dolfin.Constant(list(model.default_initial_conditions().values())))

    v, *s_lst = dolfin.split(f)
    s = dolfin.as_vector(s_lst)

    F_new = model.F(v, s)
    F_old = old_model.F(v, s)

    for state_idx in range(model.num_states()):
        assert np.isclose(
            dolfin.assemble(
                (F_new[state_idx] - F_old[state_idx]) * dolfin.dx(domain=mesh),
            ),
            0,
        )

    I_new = model.I(v, s)
    I_old = old_model.I(v, s)
    assert np.isclose(dolfin.assemble((I_new - I_old) * dolfin.dx(domain=mesh)), 0)
