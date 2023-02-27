import numpy as np
import pytest
import simcardems
from simcardems import save_load_functions as slf


@pytest.fixture
def dummyfile():
    path = "tmp_h5pyfile.h5"
    simcardems.utils.remove_file(path)
    yield path
    simcardems.utils.remove_file(path)


def tests_h5pyfile(dummyfile):
    h5group = "test"
    data1 = [1, 2, 3, 4]
    data2 = [4, 5, 6]

    with slf.h5pyfile(dummyfile, "w") as h5file:
        group = h5file.create_group(h5group)
        group.create_dataset("data1", data=data1)

    with slf.h5pyfile(dummyfile, "a") as h5file:
        h5file[h5group].create_dataset("data2", data=data2)

    with slf.h5pyfile(dummyfile, "r") as h5file:
        assert h5group in h5file

        assert "data1" in h5file[h5group]
        assert np.isclose(h5file[h5group]["data1"][:], data1).all()

        assert "data2" in h5file[h5group]
        assert np.isclose(h5file[h5group]["data2"][:], data2).all()


@pytest.mark.parametrize(
    "data",
    (
        {},
        {"a": 1, "b": 2},
        {"a": 1.0, "b": 2},
        {"a": 1.0, "b": 2, "c": "three"},
        {"a": 1.0, "b": 2, "c": "three", "d": [1, 2]},
    ),
)
def test_dict_to_h5(data, dummyfile):
    h5group = "testgroup"
    slf.dict_to_h5(data, dummyfile, h5group)

    with slf.h5pyfile(dummyfile, "r") as h5file:
        loaded_data = slf.h5_to_dict(h5file[h5group])

    assert loaded_data == data
