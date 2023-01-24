from pathlib import Path

import simcardems
from simcardems import models


def test_explicit_ORdmm_Land(geo, tmpdir):
    coupling = simcardems.models.em_model.setup_EM_model(
        cls_EMCoupling=models.explicit_ORdmm_Land.EMCoupling,
        cls_CellModel=models.explicit_ORdmm_Land.CellModel,
        cls_ActiveModel=models.explicit_ORdmm_Land.ActiveModel,
        geometry=geo,
    )
    config = simcardems.Config(outdir=Path(tmpdir), coupling_type="explicit_ORdmm_Land")
    runner = simcardems.Runner.from_models(coupling=coupling, config=config)
    runner.solve(1.0)
    assert runner.state_path.is_file()
    new_coupling = models.explicit_ORdmm_Land.EMCoupling.from_state(
        path=runner.state_path,
    )
    assert new_coupling == coupling


def test_fully_coupled_ORdmm_Land(geo, tmpdir):
    coupling = simcardems.models.em_model.setup_EM_model(
        cls_EMCoupling=models.fully_coupled_ORdmm_Land.EMCoupling,
        cls_CellModel=models.fully_coupled_ORdmm_Land.CellModel,
        cls_ActiveModel=models.fully_coupled_ORdmm_Land.ActiveModel,
        geometry=geo,
    )
    config = simcardems.Config(
        outdir=Path(tmpdir),
        coupling_type="fully_coupled_ORdmm_Land",
    )
    runner = simcardems.Runner.from_models(coupling=coupling, config=config)
    runner.solve(1.0)
    assert runner.state_path.is_file()
    new_coupling = models.fully_coupled_ORdmm_Land.EMCoupling.from_state(
        path=runner.state_path,
    )
    assert new_coupling == coupling
