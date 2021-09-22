import pytest
import simcardems.save_load_functions as slf
from simcardems import utils


GROUPNAME = "mygroup/myfunctions"


@pytest.fixture
def filename():
    path = "save_functions.h5"
    utils.remove_file(path)
    yield path
    utils.remove_file(path)


def test_dict_to_h5(filename):
    functions_dict = dict(v1=1, v2=2)

    slf.dict_to_h5(functions_dict, filename, GROUPNAME)

    with slf.h5pyfile(filename) as h5file:
        loaded_dict = slf.h5_to_dict(h5file["mygroup"]["myfunctions"])

    assert loaded_dict == functions_dict
