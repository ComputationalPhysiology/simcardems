import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from simcardems import Runner


@pytest.mark.slow
def test_runner():
    runner = Runner(lx=1, ly=1, lz=1, dx=1)
    runner.solve(0.02)


@pytest.mark.slow
def test_runner_load_state_with_new_parameters():

    outdir = Path("runner_load_state_with_new_parameters")
    if outdir.exists():
        shutil.rmtree(outdir)
    runner = Runner(outdir=outdir, lx=1, ly=1, lz=1, dx=1)
    runner.solve(0.02)

    drug_factors_file = Path("drug_factors_file.json")
    drug_factors = {"scale_drug_INa": 42.43}
    drug_factors_file.write_text(json.dumps(drug_factors))

    popu_factors_file = Path("popu_factors_file.json")
    popu_factors = {"scale_popu_GNa": 13.13}
    popu_factors_file.write_text(json.dumps(popu_factors))

    runner2 = Runner(
        outdir=outdir,
        drug_factors_file=drug_factors_file,
        popu_factors_file=popu_factors_file,
        reset=False,
    )
    assert np.isclose(
        runner2.ep_solver.ode_solver._model.parameters()["scale_drug_INa"],
        drug_factors["scale_drug_INa"],
    )
    assert np.isclose(
        runner2.ep_solver.ode_solver._model.parameters()["scale_popu_GNa"],
        popu_factors["scale_popu_GNa"],
    )

    drug_factors_file.unlink()
    popu_factors_file.unlink()
    shutil.rmtree(outdir)
