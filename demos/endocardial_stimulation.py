# # Endocardial stimulation (multiple cell models)
# In this demo we stimulate a Bi-ventricular geometry at the endocardium and compute a pseudo-ecg
#
# ```{figure} ../docs/_static/torso_electrodes.png
# ---
# name: torso_electrodes
# ---

# Displacement ($u$), active tension ($T_a$), voltage ($V$) and calcium ($Ca$)
# visualized for a specific time point in Paraview.
# ```
#
from collections import defaultdict
from pathlib import Path
import cardiac_geometries
import numpy as np
import matplotlib.pyplot as plt

import dolfin
import pulse
import ldrb
import ufl_legacy as ufl
import simcardems

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

import beat

# import beat.cellmodels.tentusscher_panfilov_2006 as model

# import beat.cellmodels.torord_dyn_chloride as model

# model_name = model.__name__.split(".")[-1]


def get_data(datadir="data_endocardial_stimulation"):
    datadir = Path(datadir)
    msh_file = datadir / "biv_ellipsoid.msh"
    if not msh_file.is_file():
        cardiac_geometries.create_biv_ellipsoid(
            datadir,
            char_length=0.5,  # Reduce this value to get a finer mesh
            center_lv_y=0.2,
            center_lv_z=0.0,
            a_endo_lv=5.0,
            b_endo_lv=2.2,
            c_endo_lv=2.2,
            a_epi_lv=6.0,
            b_epi_lv=3.0,
            c_epi_lv=3.0,
            center_rv_y=1.0,
            center_rv_z=0.0,
            a_endo_rv=6.0,
            b_endo_rv=2.5,
            c_endo_rv=2.7,
            a_epi_rv=8.0,
            b_epi_rv=5.5,
            c_epi_rv=4.0,
            create_fibers=True,
        )

    return cardiac_geometries.geometry.Geometry.from_folder(datadir)


def define_stimulus(mesh, chi, C_m, time, ffun, markers):
    duration = 2.0  # ms
    A = 5  # mu A/cm^3

    factor = 1.0 / (chi * C_m)  # NB: cbcbeat convention
    amplitude = factor * A  # mV/ms

    I_s = dolfin.Expression(
        "time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
        time=time,
        start=0.0,
        duration=duration,
        amplitude=amplitude,
        degree=0,
    )

    subdomain_data = dolfin.MeshFunction("size_t", mesh, 2)
    subdomain_data.set_all(0)
    marker = 1
    subdomain_data.array()[ffun.array() == markers["ENDO_LV"][0]] = 1
    subdomain_data.array()[ffun.array() == markers["ENDO_RV"][0]] = 1

    ds = dolfin.Measure("ds", domain=mesh, subdomain_data=subdomain_data)(marker)
    return beat.base_model.Stimulus(dz=ds, expr=I_s)


def define_conductivity_tensor(chi, C_m, f0, s0, n0):
    # Conductivities as defined by page 4339 of Niederer benchmark
    sigma_il = 0.17  # mS / mm
    sigma_it = 0.019  # mS / mm
    sigma_el = 0.62  # mS / mm
    sigma_et = 0.24  # mS / mm

    # Compute monodomain approximation by taking harmonic mean in each
    # direction of intracellular and extracellular part
    def harmonic_mean(a, b):
        return a * b / (a + b)

    sigma_l = harmonic_mean(sigma_il, sigma_el)
    sigma_t = harmonic_mean(sigma_it, sigma_et)

    # Scale conducitivites by 1/(C_m * chi)
    s_l = sigma_l / (C_m * chi)  # mm^2 / ms
    s_t = sigma_t / (C_m * chi)  # mm^2 / ms

    # Define conductivity tensor
    A = dolfin.as_matrix(
        [
            [f0[0], s0[0], n0[0]],
            [f0[1], s0[1], n0[1]],
            [f0[2], s0[2], n0[2]],
        ],
    )

    M_star = ufl.diag(dolfin.as_vector([s_l, s_t, s_t]))
    M = A * M_star * A.T

    return M


def load_timesteps_from_xdmf(xdmffile):
    import xml.etree.ElementTree as ET

    times = {}
    i = 0
    tree = ET.parse(xdmffile)
    for elem in tree.iter():
        if elem.tag == "Time":
            times[i] = float(elem.get("Value"))
            i += 1

    return times


def load_from_file(heart_mesh, xdmffile, key="v", stop_index=None):
    V = dolfin.FunctionSpace(heart_mesh, "Lagrange", 1)
    v = dolfin.Function(V)

    timesteps = load_timesteps_from_xdmf(xdmffile)
    with dolfin.XDMFFile(Path(xdmffile).as_posix()) as f:
        for i, t in tqdm(timesteps.items()):
            f.read_checkpoint(v, key, i)
            yield v.copy(deepcopy=True), t


def compute_ecg_recovery():
    datadir = Path("data_endocardial_stimulation")
    xdmffile = datadir / "state.xdmf"
    data = get_data(datadir=datadir)

    # https://litfl.com/ecg-lead-positioning/
    vs = load_from_file(data.mesh, xdmffile, key="V")

    leads = dict(
        RA=(-15.0, 0.0, -10.0),
        LA=(4.0, -12.0, -7.0),
        RL=(0.0, 20.0, 3.0),
        LL=(17.0, 11.0, 7.0),
        V1=(-3.0, 4.0, -9.0),
        V2=(0.0, 2.0, -8.0),
        V3=(3.0, 1.0, -8.0),
        V4=(6.0, 1.0, -6.0),
        V5=(10.0, 2.0, 0.0),
        V6=(10.0, -6.0, 2.0),
    )

    fname = datadir / "extracellular_potential.npy"
    if not fname.is_file():
        phie = defaultdict(list)
        time = []
        for v, t in vs:
            time.append(t)
            for name, point in leads.items():
                phie[name].append(
                    beat.ecg.ecg_recovery(
                        v=v,
                        mesh=data.mesh,
                        sigma_b=1.0,
                        point=point,
                    ),
                )
        np.save(fname, {"phie": phie, "time": time})

    phie_time = np.load(fname, allow_pickle=True).item()
    phie = phie_time["phie"]
    time = phie_time["time"]

    fig, ax = plt.subplots(2, 5, sharex=True, figsize=(12, 8))
    for i, (name, values) in enumerate(phie.items()):
        axi = ax.flatten()[i]
        axi.plot(time, values)
        axi.set_title(name)
    fig.savefig(datadir / "extracellular_potential.png")

    ecg = beat.ecg.Leads12(**{k: np.array(v) for k, v in phie.items()})
    fig, ax = plt.subplots(3, 4, sharex=True, figsize=(12, 8))
    for i, name in enumerate(
        [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1_",
            "V2_",
            "V3_",
            "V4_",
            "V5_",
            "V6_",
        ],
    ):
        y = getattr(ecg, name)
        axi = ax.flatten()[i]
        axi.plot(time, y)
        axi.set_title(name)
    fig.savefig(datadir / "ecg_12_leads.png")
    # breakpoint()


def main():
    datadir = Path("data_endocardial_stimulation")
    data = get_data(datadir=datadir)

    dolfin.parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    ep_mesh = dolfin.adapt(data.mesh)
    ffun_ep = dolfin.adapt(data.ffun, ep_mesh)
    ldrb_markers = {
        "base": data.markers["BASE"][0],
        "lv": data.markers["ENDO_LV"][0],
        "rv": data.markers["ENDO_RV"][0],
        "epi": data.markers["EPI"][0],
    }

    f0, s0, n0 = ldrb.dolfin_ldrb(
        mesh=ep_mesh,
        fiber_space="CG_1",
        ffun=ffun_ep,
        markers=ldrb_markers,
        alpha_endo_lv=60,  # Fiber angle on the endocardium
        alpha_epi_lv=-60,  # Fiber angle on the epicardium
    )
    microstructure_ep = pulse.Microstructure(f0=f0, s0=s0, n0=n0)
    microstructure = pulse.Microstructure(f0=data.f0, s0=data.s0, n0=data.n0)
    geo = simcardems.lvgeometry.LeftVentricularGeometry(
        mechanics_mesh=data.mesh,
        ep_mesh=ep_mesh,
        microstructure=microstructure,
        microstructure_ep=microstructure_ep,
        ffun=data.ffun,
        ffun_ep=ffun_ep,
        markers=data.markers,
    )
    coupling = simcardems.models.fully_coupled_ORdmm_Land.EMCoupling(geometry=geo)

    V = dolfin.FunctionSpace(ep_mesh, "Lagrange", 1)

    markers = dolfin.Function(V)
    arr = beat.utils.expand_layer(
        markers=markers,
        mfun=ffun_ep,
        endo_markers=[data.markers["ENDO_LV"][0], data.markers["ENDO_RV"][0]],
        epi_markers=[data.markers["EPI"][0]],
        endo_marker=1,
        epi_marker=2,
        endo_size=0.3,
        epi_size=0.3,
    )

    markers.vector().set_local(arr)

    with dolfin.XDMFFile((datadir / "markers.xdmf").as_posix()) as xdmf:
        xdmf.write(markers)

    model = simcardems.models.fully_coupled_ORdmm_Land.cell_model

    init_states = {
        0: model.init_state_values(),
        1: model.init_state_values(),
        2: model.init_state_values(),
    }
    parameters = {
        0: model.init_parameter_values(amp=0.0, celltype=2),
        1: model.init_parameter_values(amp=0.0, celltype=0),
        2: model.init_parameter_values(amp=0.0, celltype=1),
    }
    fun = {
        0: model.forward_generalized_rush_larsen(coupling=coupling),
        1: model.forward_generalized_rush_larsen(coupling=coupling),
        2: model.forward_generalized_rush_larsen(coupling=coupling),
    }
    v_index = {
        0: model.state_index("v"),
        1: model.state_index("v"),
        2: model.state_index("v"),
    }
    # Surface to volume ratio
    chi = 140.0  # mm^{-1}
    # Membrane capacitance
    C_m = 0.01  # muqq F / mm^2

    time = dolfin.Constant(0.0)
    I_s = define_stimulus(
        mesh=ep_mesh,
        chi=chi,
        C_m=C_m,
        time=time,
        ffun=ffun_ep,
        markers=data.markers,
    )

    M = define_conductivity_tensor(chi, C_m, f0=f0, s0=s0, n0=n0)

    params = {"preconditioner": "sor", "use_custom_preconditioner": False}
    pde = beat.MonodomainModel(time=time, mesh=ep_mesh, M=M, I_s=I_s, params=params)

    ode = beat.odesolver.DolfinMultiODESolver(
        pde.state,
        markers=markers,
        num_states={i: len(s) for i, s in init_states.items()},
        fun=fun,
        init_states=init_states,
        parameters=parameters,
        v_index=v_index,
    )

    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)
    coupling.register_ep_model(solver)
    mech_heart = simcardems.mechanics_model.setup_solver(
        coupling=coupling,
        ActiveModel=simcardems.models.fully_coupled_ORdmm_Land.ActiveModel,
    )
    coupling.register_mech_model(mech_heart)
    coupling.setup_assigners()
    runner = simcardems.Runner.from_models(coupling=coupling)

    runner.solve(T=5.0, save_freq=1)

    # T = 1
    # # Change to 500 to simulate the full cardiac cycle
    # # T = 500
    # t = 0.0
    # dt = 0.05

    # fname = (datadir / f"state.xdmf").as_posix()
    # i = 0
    # while t < T + 1e-12:
    #     if i % 20 == 0:
    #         v = solver.pde.state.vector().get_local()
    #         print(f"Solve for {t=:.2f}, {v.max() =}, {v.min() = }")
    #         with dolfin.XDMFFile(dolfin.MPI.comm_world, fname) as xdmf:
    #             xdmf.write_checkpoint(
    #                 solver.pde.state,
    #                 "V",
    #                 float(t),
    #                 dolfin.XDMFFile.Encoding.HDF5,
    #                 True,
    #             )
    #     solver.step((t, t + dt))
    #     i += 1
    #     t += dt


def postprocess():
    # simcardems.postprocess.plot_state_traces("results/results.h5")
    simcardems.postprocess.make_xdmffiles("results/results.h5", names=["u", "v"])


if __name__ == "__main__":
    main()
    # compute_ecg_recovery()

    postprocess()
