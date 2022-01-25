try:
    import streamlit as st
except ImportError:
    print("Please install streamlit - python3 -m pip install streamlit")
    exit(1)

try:
    import fenics_plotly
except ImportError:
    print("Please install fenics_plotly - python3 -m pip install fenics_plotly")
    exit(1)


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


import simcardems

import pulse
import cbcbeat
import hashlib

simcardems_folder = Path.home().joinpath("simcardems")


def about():

    st.title("About")

    for i in range(8):
        st.sidebar.write("")

    intro_markdown = """

    The SIMula CARDiac ElectroMechanics solver  is ..."""
    st.markdown(intro_markdown)

    return


class Postprocess:
    @staticmethod
    def load_data(outdir):

        loader = simcardems.DataLoader(outdir.joinpath("results.h5"))
        bnd = {
            "ep": simcardems.postprocess.Boundary(loader.ep_mesh),
            "mechanics": simcardems.postprocess.Boundary(loader.mech_mesh),
        }

        all_names = {"mechanics": ["lmbda", "Ta"], "ep": ["V", "Ca"]}

        values = {
            group: {name: np.zeros(len(loader.time_stamps)) for name in names}
            for group, names in all_names.items()
        }

        for i, t in enumerate(loader.time_stamps):
            for group, names in all_names.items():
                for name in names:
                    func = loader.get(group, name, t)
                    dof_coords = func.function_space().tabulate_dof_coordinates()
                    dof = np.argmin(
                        np.linalg.norm(
                            dof_coords - np.array(bnd[group].center),
                            axis=1,
                        ),
                    )
                    if np.isclose(dof_coords[dof], np.array(bnd[group].center)).all():
                        # If we have a dof at the center - evaluation at dof (cheaper)
                        values[group][name][i] = func.vector().get_local()[dof]
                    else:
                        # Otherwise, evaluation at center coordinates
                        values[group][name][i] = func(bnd[group].center)

        times = np.array(loader.time_stamps, dtype=float)
        return times, values

    @staticmethod
    def make_figures(times, values):
        fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        ax[0, 0].plot(times[1:], values["mechanics"]["lmbda"][1:])
        ax[0, 1].plot(times[1:], values["mechanics"]["Ta"][1:])
        ax[1, 0].plot(times, values["ep"]["V"])
        ax[1, 1].plot(times[1:], values["ep"]["Ca"][1:])

        ax[0, 0].set_title(r"$\lambda$")
        ax[0, 1].set_title("Ta")
        ax[1, 0].set_title("V")
        ax[1, 1].set_title("Ca")
        for axi in ax.flatten():
            axi.grid()
            if False:
                axi.set_xlim([0, 5000])
        ax[1, 0].set_xlabel("Time [ms]")
        ax[1, 1].set_xlabel("Time [ms]")
        ax[0, 0].set_ylim(
            [
                min(0.9, min(values["mechanics"]["lmbda"][1:])),
                max(1.1, max(values["mechanics"]["lmbda"][1:])),
            ],
        )
        return fig


def postprocess():
    st.title("Postprocessing")

    for i in range(8):
        st.sidebar.write("")

    intro_markdown = f"""
    Here is the postprocessing page.

    Here you will be able to
    load all the results that are located in the folder
    `{simcardems_folder}`

    """
    st.markdown(intro_markdown)

    outdir = st.selectbox(
        "Select result directory",
        [""] + list(simcardems_folder.iterdir()),
        format_func=lambda x: "Select an option" if x == "" else x,
    )
    if outdir == "":
        st.info("Please select a result diretcory")
        return

    outdir = Path(outdir)

    # TODO: Use multiselect instead
    selected_node = st.selectbox(
        "Select a node",
        [""] + list(simcardems.postprocess.Boundary.nodes()),
        format_func=lambda x: "Select an option" if x == "" else x,
    )
    if selected_node == "":
        st.info("Please select a node")
        return

    times, values = Postprocess.load_data(outdir)
    fig = Postprocess.make_figures(times, values)
    st.pyplot(fig)

    return


class Simulation:
    @staticmethod
    def plot_mesh(mesh, label):
        fig_mech_mesh = fenics_plotly.fenics_plotly.plot(mesh, show=False)
        st.subheader(f"{label} mesh")
        st.text(f"Num cells: {mesh.num_cells()}, Num vertices: {mesh.num_vertices()}")
        st.plotly_chart(fig_mech_mesh.figure)

    @staticmethod
    def handle_mesh():
        st.header("Mesh")
        cols_lxyz = st.columns(3)
        with cols_lxyz[0]:
            lx = st.number_input("lx", value=simcardems.cli._Defaults.lx)
        with cols_lxyz[1]:
            ly = st.number_input("ly", value=simcardems.cli._Defaults.ly)
        with cols_lxyz[2]:
            lz = st.number_input("lz", value=simcardems.cli._Defaults.lz)

        cols_dxref = st.columns(2)
        with cols_dxref[0]:
            dx = st.number_input("dx", value=simcardems.cli._Defaults.dx)
        with cols_dxref[1]:
            num_refinements = st.number_input(
                "num_refinements",
                value=simcardems.cli._Defaults.num_refinements,
            )

        return {
            "lx": lx,
            "ly": ly,
            "lz": lz,
            "dx": dx,
            "num_refinements": num_refinements,
        }

    @staticmethod
    def handle_ep_solver_args():
        st.header("EP solver")

        cols_ep = st.columns(2)
        with cols_ep[0]:
            dt = st.number_input(
                "dt",
                value=simcardems.cli._Defaults.dt,
                help="Time step for EP solver",
            )
        with cols_ep[1]:
            disease_state = st.selectbox(
                "Disease state",
                ["healthy", "hf"],
                help="Healthy or heart failure",
            )

        cols_files = st.columns(3)
        with cols_files[0]:
            cell_init_file = st.file_uploader(
                "Cell initial conditions",
                type=["json", "h5"],
                help="File with initial conditions for the cell",
            )
        with cols_files[1]:
            drug_factors_file = st.file_uploader(
                "Drug factors",
                type="json",
                help="File with drug factors",
            )
        with cols_files[2]:
            popu_factors_file = st.file_uploader(
                "Poulation factors",
                type="json",
                help="File with population factors",
            )
        return {
            "dt": dt,
            "disease_state": disease_state,
            "cell_init_file": cell_init_file,
            "drug_factors_file": drug_factors_file,
            "popu_factors_file": popu_factors_file,
        }

    @staticmethod
    def handle_mechanics_args():
        st.header("Mechanics solver")
        cols_mech = st.columns(4)
        with cols_mech[0]:
            bnd_cond = st.selectbox(
                "Boundary conditions",
                simcardems.mechanics_model.BoundaryConditions._member_names_,
            )
        with cols_mech[1]:
            add_pre_stretch = st.checkbox("Add pre stretch")
            pre_stretch = None
            if add_pre_stretch:
                pre_stretch = st.number_input("Pre stretch value", 0)

        with cols_mech[2]:
            add_traction = st.checkbox("Add pre traction")
            traction = None
            if add_traction:
                traction = st.number_input("Traction value", 0)

        with cols_mech[3]:
            add_spring = st.checkbox("Add spring")
            spring = None
            if add_spring:
                spring = st.number_input("Spring value", 0)

        return {
            "bnd_cond": bnd_cond,
            "pre_stretch": pre_stretch,
            "traction": traction,
            "spring": spring,
        }

    @staticmethod
    @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
    def load_model(geometry_args, ep_solver_args, mechanics_args):

        st.info("Create geometry")
        geometry = simcardems.geometry.SlabGeometry(**geometry_args)
        st.info("Create EM coupling")
        coupling = simcardems.EMCoupling(geometry)
        st.info("Create EP model")
        solver = simcardems.ep_model.setup_solver(coupling=coupling, **ep_solver_args)
        coupling.register_ep_model(solver)
        st.info("Create Mechanics model")
        mech_heart: pulse.MechanicsProblem = (
            simcardems.mechanics_model.setup_mechanics_model(
                coupling=coupling,
                cell_params=solver.ode_solver._model.parameters(),
                linear_solver="superlu_dist",
                **mechanics_args,
            )
        )
        st.success("Done loading model")
        return coupling, solver, mech_heart

    @staticmethod
    def visualize_model(
        coupling,
    ):
        st.header("Visualize model")
        cols_plot_mesh = st.columns(2)
        with cols_plot_mesh[0]:
            plot_mechanics_mesh = st.checkbox("Plot mechanics mesh")
        with cols_plot_mesh[1]:
            plot_ep_mesh = st.checkbox("Plot EP mesh")

        if plot_mechanics_mesh:
            Simulation.plot_mesh(coupling.geometry.mechanics_mesh, "Mechanics")

        if plot_ep_mesh:
            Simulation.plot_mesh(coupling.geometry.ep_mesh, "EP")

    @staticmethod
    def setup_simulation(geometry_args, ep_solver_args, mechanics_args):
        st.header("Run")

        cols_run = st.columns(3)
        with cols_run[0]:
            T = st.number_input("T", value=simcardems.cli._Defaults.T, help="End time")

        with cols_run[1]:
            save_freq = st.number_input(
                "Save frequency",
                value=simcardems.cli._Defaults.save_freq,
                help="How often to save the results",
            )

        all_options = {"T": T, **geometry_args, **ep_solver_args, **mechanics_args}
        simulation_id = hashlib.md5(repr(all_options).encode()).hexdigest()
        with cols_run[2]:
            outdir = st.text_input(
                "Output directory",
                value=simcardems_folder.joinpath(f"results_{simulation_id}"),
            )
        outdir = Path(outdir)
        create_result_dir = st.checkbox(f"Create result directory '{outdir}'")
        if not create_result_dir:
            return T, save_freq, None

        if not outdir.is_dir():
            st.info(f"Directory '{outdir}' does not exist. Creating...")
            outdir.mkdir(parents=True)
        else:
            st.info(f"Directory '{outdir}' allready exist")
        return T, save_freq, outdir

    @staticmethod
    def run(
        coupling,
        mech_heart,
        solver,
        mechanics_args,
        ep_solver_args,
        T,
        save_freq,
        outdir,
    ):
        vs = solver.solution_fields()[1]
        v, v_assigner = simcardems.utils.setup_assigner(vs, 0)
        Ca, Ca_assigner = simcardems.utils.setup_assigner(vs, 45)

        pre_XS, preXS_assigner = simcardems.utils.setup_assigner(vs, 40)
        pre_XW, preXW_assigner = simcardems.utils.setup_assigner(vs, 41)
        u_subspace_index = 1 if mechanics_args["bnd_cond"] == "rigid" else 0
        u, u_assigner = simcardems.utils.setup_assigner(
            mech_heart.state,
            u_subspace_index,
        )
        u_assigner.assign(u, mech_heart.state.sub(u_subspace_index))

        collector = simcardems.DataCollector(
            outdir,
            coupling.mech_mesh,
            coupling.ep_mesh,
            reset_state=True,
        )
        for group, name, f in [
            ("mechanics", "u", u),
            ("ep", "V", v),
            ("ep", "Ca", Ca),
            ("mechanics", "lmbda", coupling.lmbda_mech),
            ("mechanics", "Ta", mech_heart.material.active.Ta_current),
        ]:
            collector.register(group, name, f)

        state_path = outdir.joinpath("state.h5")
        t0 = 0
        dt = ep_solver_args["dt"]
        time_stepper = cbcbeat.utils.TimeStepper((t0, T), dt, annotate=False)
        save_it = int(save_freq / dt)

        mech_heart.solve()

        my_bar = st.progress(0)
        total = round((T - t0) / dt)

        for (i, (t0, t1)) in enumerate(time_stepper):
            my_bar.progress((i + 1) / total)
            # Solve EP model
            solver.step((t0, t1))
            # Update these states that are needed in the Mechanics solver
            coupling.update_mechanics()
            XS_norm = simcardems.utils.compute_norm(coupling.XS_ep, pre_XS)
            XW_norm = simcardems.utils.compute_norm(coupling.XW_ep, pre_XW)

            if XS_norm + XW_norm >= 0.1:

                preXS_assigner.assign(pre_XS, simcardems.utils.sub_function(vs, 40))
                preXW_assigner.assign(pre_XW, simcardems.utils.sub_function(vs, 41))

                coupling.interpolate_mechanics()

                # Solve the Mechanics model
                mech_heart.solve()
                coupling.interpolate_ep()
                # Update previous
                mech_heart.material.active.update_prev()
                coupling.update_ep()

            solver.vs_.assign(solver.vs)
            # # Store every 'save_freq' ms
            if i % save_it == 0:
                # Assign u, v and Ca for postprocessing
                v_assigner.assign(v, simcardems.utils.sub_function(vs, 0))
                Ca_assigner.assign(Ca, simcardems.utils.sub_function(vs, 45))
                u_assigner.assign(u, mech_heart.state.sub(u_subspace_index))
                collector.store(t0)

        simcardems.save_load_functions.save_state(
            state_path,
            solver=solver,
            mech_heart=mech_heart,
            coupling=coupling,
            dt=dt,
            bnd_cond=mechanics_args["bnd_cond"],
            t0=t0,
        )
        st.success("Done!")


def simulation():
    st.title("Simulation")
    geometry_args = Simulation.handle_mesh()
    ep_solver_args = Simulation.handle_ep_solver_args()
    mechanics_args = Simulation.handle_mechanics_args()
    mechanics_args.update({"dt": ep_solver_args["dt"]})

    load_model = st.checkbox("Load model")
    if not load_model:
        return

    coupling, ep_solver, mech_heart = Simulation.load_model(
        geometry_args=geometry_args,
        ep_solver_args=ep_solver_args,
        mechanics_args=mechanics_args,
    )
    Simulation.visualize_model(coupling=coupling)

    T, save_freq, outdir = Simulation.setup_simulation(
        geometry_args,
        ep_solver_args,
        mechanics_args,
    )
    if outdir is None:
        return

    if st.button("Run simulation"):

        Simulation.run(
            coupling=coupling,
            mech_heart=mech_heart,
            solver=ep_solver,
            T=T,
            outdir=outdir,
            save_freq=save_freq,
            mechanics_args=mechanics_args,
            ep_solver_args=ep_solver_args,
        )

    return


# Page settings
st.set_page_config(page_title="simcardems")

# Sidebar settings
pages = {
    "About": about,
    "Simulation": simulation,
    "Postprocess": postprocess,
}

st.sidebar.title("Simcardems")

# Radio buttons to select desired option
page = st.sidebar.radio("", tuple(pages.keys()))

pages[page]()

# About
st.sidebar.markdown(
    """
- [Source code](https://github.com/ComputationalPhysiology/simcardems)
- [Documentation](http://computationalphysiology.github.io/simcardems)
""",
)
