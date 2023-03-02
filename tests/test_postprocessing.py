import math

import dolfin
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
