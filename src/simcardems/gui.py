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
import matplotlib.pyplot as plt


import simcardems
import mpi4py

import pulse
import hashlib

simcardems_folder = Path.home().joinpath("simcardems")


def return_none(*args, **kwargs):
    return None


def about():
    st.title("About")

    for i in range(8):
        st.sidebar.write("")

    intro_markdown = """
    The SIMula CARDiac ElectroMechanics solver
    is a FEniCS-based cardiac electro-mechanics
    solver and is developed as a part of the
    [SimCardio Test project](https://www.simcardiotest.eu/wordpress).
    The solver depends on the mechanics solver
    [pulse](https://github.com/ComputationalPhysiology/pulse)
    and electrophysiology solver [cbcbeat](https://github.com/ComputationalPhysiology/cbcbeat).

    Please consult the [documentation](http://computationalphysiology.github.io/simcardems)
    if you want to learn more.
    """
    st.markdown(intro_markdown)

    return


class Postprocess:
    @staticmethod
    def load_data(outdir, reduction="center"):
        loader = simcardems.DataLoader(outdir.joinpath("results.h5"))
        return simcardems.postprocess.extract_traces(loader=loader, reduction=reduction)

    @staticmethod
    def make_figures(values):
        fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        ax[0, 0].plot(values["time"][1:], values["mechanics"]["lmbda"][1:])
        ax[0, 1].plot(values["time"][1:], values["mechanics"]["Ta"][1:])
        ax[1, 0].plot(values["time"], values["ep"]["V"])
        ax[1, 1].plot(values["time"][1:], values["ep"]["Ca"][1:])

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
    if not simcardems_folder.exists():
        st.info("No results found")
        return

    outdir = st.selectbox(
        "Select result directory",
        [""] + list(simcardems_folder.iterdir()),
        format_func=lambda x: "Select an option" if x == "" else x,
    )
    if outdir == "":
        st.info("Please select a result directory")
        return

    outdir = Path(outdir)

    # TODO: Use multiselect instead
    reduction = st.selectbox(
        "Select a reduction",
        [""] + ["center", "average"],
        format_func=lambda x: "Select an option" if x == "" else x,
    )
    if reduction == "":
        st.info("Please select a node")
        return

    values = Postprocess.load_data(outdir, reduction=reduction)
    fig = Postprocess.make_figures(values)
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
        cols_l_xyz = st.columns(3)
        params = simcardems.slabgeometry.SlabGeometry.default_parameters()
        with cols_l_xyz[0]:
            lx = st.number_input("lx", value=params["lx"])
        with cols_l_xyz[1]:
            ly = st.number_input("ly", value=params["ly"])
        with cols_l_xyz[2]:
            lz = st.number_input("lz", value=params["lz"])

        cols_dx_ref = st.columns(2)
        with cols_dx_ref[0]:
            dx = st.number_input("dx", value=params["dx"])
        with cols_dx_ref[1]:
            num_refinements = st.number_input(
                "num_refinements",
                value=params["num_refinements"],
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
                value=simcardems.Config.dt,
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
                "Population factors",
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
            bnd_rigid = st.checkbox("Rigid motion conditions")
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
            "bnd_rigid": bnd_rigid,
            "pre_stretch": pre_stretch,
            "traction": traction,
            "spring": spring,
        }

    @staticmethod
    @st.cache(
        suppress_st_warning=True,
        allow_output_mutation=True,
        show_spinner=True,
        hash_funcs={mpi4py.MPI.Op: return_none},
    )
    def load_model(geometry_args, ep_solver_args, mechanics_args):
        st.info("Create geometry")
        geometry = simcardems.slabgeometry.SlabGeometry(parameters=geometry_args)
        st.info("Create EM coupling")
        coupling = simcardems.EMCoupling(geometry)
        st.info("Create EP model")
        ep_solver = simcardems.setup_models.setup_ep_solver(
            coupling=coupling, **ep_solver_args
        )
        coupling.register_ep_model(ep_solver)
        st.info("Create Mechanics model")
        mech_heart: pulse.MechanicsProblem = (
            simcardems.setup_models.setup_mechanics_solver(
                coupling=coupling,
                cell_params=ep_solver.ode_solver._model.parameters(),
                linear_solver="superlu_dist",
                geo=geometry,
                **mechanics_args,
            )
        )
        st.success("Done loading model")
        return simcardems.Runner.from_models(
            coupling=coupling,
            ep_solver=ep_solver,
            mech_heart=mech_heart,
            geo=geometry,
        )

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
            T = st.number_input(
                "T",
                value=simcardems.Config.T,
                help="End time",
            )

        with cols_run[1]:
            save_freq = st.number_input(
                "Save frequency",
                value=simcardems.Config.save_freq,
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
            st.info(f"Directory '{outdir}' already exist")
        return T, save_freq, outdir


def simulation():
    st.title("Simulation")
    geometry_args = Simulation.handle_mesh()
    ep_solver_args = Simulation.handle_ep_solver_args()
    mechanics_args = Simulation.handle_mechanics_args()

    load_model = st.checkbox("Load model")
    if not load_model:
        return

    runner: simcardems.Runner = Simulation.load_model(
        geometry_args=geometry_args,
        ep_solver_args=ep_solver_args,
        mechanics_args=mechanics_args,
    )
    Simulation.visualize_model(coupling=runner.coupling)

    T, save_freq, outdir = Simulation.setup_simulation(
        geometry_args,
        ep_solver_args,
        mechanics_args,
    )
    if outdir is None:
        return

    if st.button("Run simulation"):
        runner.outdir = outdir
        st_progress = st.progress(0)
        runner.solve(T=T, save_freq=save_freq, st_progress=st_progress)
        st.success("Done!")

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
page = st.sidebar.radio("Pages", tuple(pages.keys()))

pages[page]()

# About
st.sidebar.markdown(
    """
- [Source code](https://github.com/ComputationalPhysiology/simcardems)
- [Documentation](http://computationalphysiology.github.io/simcardems)
""",
)
