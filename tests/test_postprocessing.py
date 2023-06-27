import math
from pathlib import Path

import dolfin
import numpy as np
import simcardems


def test_activation_map():
    mesh = dolfin.UnitCubeMesh(5, 5, 5)
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    v = dolfin.Function(V)
    t = dolfin.Constant(0.0)

    time_stamps = range(5)

    def voltage_gen():
        for ti in time_stamps:
            t.assign(ti)
            v.interpolate(dolfin.Expression("t * (x[0] + 1) - 4.0", t=t, degree=1))
            yield v

    with dolfin.XDMFFile(dolfin.MPI.comm_world, "volt.xdmf") as volt_file:
        for ti, vi in enumerate(voltage_gen()):
            volt_file.write_checkpoint(
                vi,
                "voltage",
                ti,
                dolfin.XDMFFile.Encoding.HDF5,
                True,
            )

    act = simcardems.postprocess.activation_map(
        voltage=voltage_gen(),
        time_stamps=time_stamps,
        V=V,
        threshold=0.0,
    )

    assert math.isclose(act(0.0, 0.0, 0.0), 4)
    assert math.isclose(act(0.0, 0.0, 0.5), 4)
    assert math.isclose(act(0.5, 0.0, 0.0), 3.0)
    assert math.isclose(act(0.5, 0.5, 0.5), 3.0)
    assert math.isclose(act(1.0, 0.0, 0.0), 2.0)
    assert math.isclose(act(1.0, 0.5, 0.5), 2.0)


def test_extract_sub_results(geo, tmp_path):
    results_file = tmp_path / "results.h5"
    collector = simcardems.DataCollector(
        outdir=results_file.parent,
        outfilename=results_file.name,
        geo=geo,
    )

    # Setup a two mech functions and one ep function
    V_mech = dolfin.FunctionSpace(geo.mesh, "Lagrange", 1)
    f1_mech = dolfin.Function(V_mech)
    collector.register("mechanics", "func1", f1_mech)
    f2_mech = dolfin.Function(V_mech)
    collector.register("mechanics", "func2", f2_mech)

    V_ep = dolfin.FunctionSpace(geo.mesh, "Lagrange", 1)
    f3_ep = dolfin.Function(V_ep)
    collector.register("ep", "func3", f3_ep)

    times = np.arange(0, 10, 0.5)
    for t in times:
        f1_mech.vector()[:] = t
        f2_mech.vector()[:] = 10 + t
        f3_ep.vector()[:] = 42 + t
        collector.store(t)

    loader = simcardems.DataLoader(collector.results_file)

    assert loader.time_stamps == [f"{ti:.2f}" for ti in times]
    sub_results_file = tmp_path / "sub_results.h5"
    sub_collector = simcardems.postprocess.extract_sub_results(
        results_file=results_file,
        output_file=sub_results_file,
        t_start=5.0,
        t_end=7.0,
        names={"mechanics": ["func1"]},
    )

    assert str(sub_collector.results_file) == str(sub_results_file)
    assert Path(sub_results_file).is_file()
    sub_loader = simcardems.DataLoader(sub_results_file)
    assert sub_loader.names == {"ep": [], "mechanics": ["func1"]}
    assert sub_loader.time_stamps == ["5.00", "5.50", "6.00", "6.50", "7.00"]
