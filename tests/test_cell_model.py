import numpy as np
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

    p1 = Parameter("g_CaL", 2.0, factors={"drug": 2.0, "popu": 2.0})
    p2 = Parameter("g_Na", 3.0, factors={"drug": 3.0, "popu": 3.0})
    assert np.isclose(p1 * p2, 8 * 27)
    assert np.isclose(p1 * 27, 8 * 27)
    assert np.isclose(27 * p1, 27 * 8)


def test_parameter_add():

    p1 = Parameter("g_CaL", 2.0, factors={"drug": 2.0, "popu": 2.0})
    p2 = Parameter("g_Na", 3.0, factors={"drug": 3.0, "popu": 3.0})
    assert np.isclose(p1 + p2, 8 + 27)
    assert np.isclose(p1 + 27, 8 + 27)
    assert np.isclose(27 + p1, 27 + 8)
