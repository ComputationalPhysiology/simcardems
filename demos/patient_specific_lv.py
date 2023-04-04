# # Patient specific geometry
#
# In this demo we show to use `simcardems` on a patient specific geometry coming from `gmsh`. We have uploaded a [gmsh file](https://github.com/ComputationalPhysiology/simcardems/blob/main/demos/geometries/patient.msh) of an LV geometry where the basal, endocardial and epicardial surfaces are marked. In order to run a simulation we need to first convert this geometry into FEniCS format and then generate some fiber orientations. To convert the mesh we will use a library called [`cardiac-geometries`](https://computationalphysiology.github.io/cardiac_geometries/) and we will generate rule-based fiber orientations using the [`ldrb algorithm`](https://finsberg.github.io/ldrb/README.html).
#
# First we make the necessary imports.

from pathlib import Path
import math
import cardiac_geometries
import ldrb
import simcardems
import pulse

# Next we use `cardiac-geometries` to convert the gmsh file into FEniCS format.
#

msh_file = "geometries/patient.msh"
geo = cardiac_geometries.gmsh2dolfin(msh_file)

# Mesh is currently in centimeters. Let us make sure it in a unit so that we can use kPa directly without scaling it.
#

geo.mesh.scale(1 / math.sqrt(10))

# The `geo` object now contains the markers, but the `ldrb` algorithm expects the markers in a dictionary with keys `base`, `lv` and `epi` so we create a translation for this.
#

ldrb_markers = {
    "base": geo.markers["BASE"][0],
    "lv": geo.markers["ENDO"][0],
    "epi": geo.markers["EPI"][0],
}

# We also need to specify a function space for the fibers. In this example we will will first order discontinuous lagrange elements.

fiber_space = "DG_1"

# We also need so specify the fiber fiber orientations on the endo- and epicardium.

angles = dict(
    alpha_endo_lv=60,  # Fiber angle on the endocardium
    alpha_epi_lv=-60,  # Fiber angle on the epicardium
    beta_endo_lv=0,  # Sheet angle on the endocardium
    beta_epi_lv=0,
)

# Now we can run the ldrb algorithm in order to get the fibers (`f0`), sheets (`s0`) and sheet-normal (`n0`) directions. For some reason we also need to allow these functions to be extrapolated.

f0, s0, n0 = ldrb.dolfin_ldrb(
    mesh=geo.mesh,
    fiber_space=fiber_space,
    ffun=geo.marker_functions.ffun,
    markers=ldrb_markers,
    **angles
)
f0.set_allow_extrapolation(True)
s0.set_allow_extrapolation(True)
n0.set_allow_extrapolation(True)


# Next we define a directory where to store the output

here = Path(__file__).absolute().parent
outdir = here / "results_patient_specific_lv"
outdir.mkdir(exist_ok=True)


# and then we can load the geometry into simcardems.
#

geometry = simcardems.lvgeometry.LeftVentricularGeometry(
    mechanics_mesh=geo.mesh,
    microstructure=pulse.Microstructure(f0=f0, s0=s0, n0=n0),
    ffun=geo.marker_functions.ffun,
    markers=geo.markers,
    parameters={"num_refinements": 1, "fiber_space": fiber_space},
)

# For this example we will use the Tor-Land model with some non-default initial conditions specified in the following file
#

# Specify path to the initial conditions for the cell model
initial_conditions_path = (
    here / "initial_conditions/fully_coupled_Tor_Land/init_5000beats.json"
)

# We will run the simulation for 1000 milliseconds. We apply a spring on the epicardium to mimic the pericardium and we set the traction on the endocardium to zero.

config = simcardems.Config(
    outdir=outdir,
    coupling_type="fully_coupled_Tor_Land",
    T=1000,
    traction=0.0,  # Pressure on the endocardium
    cell_init_file=initial_conditions_path,
    mechanics_use_continuation=True,
)

# Now we can create the coupling.

coupling = simcardems.models.em_model.setup_EM_model_from_config(
    config=config,
    geometry=geometry,
)

# And to make things move a little bit more we will scale the reference active tension to 60 kPa.

coupling.mech_solver.material.active.T_ref.assign(10.0)

# Now we create the runner

runner = simcardems.Runner.from_models(config=config, coupling=coupling)

# but before we run the EM simulations, we will inflate the LV to a cavity pressure of 3 kPa.
#
# xdmf = dolfin.XDMFFile(dolfin.MPI.comm_world, "u.xdmf")
# u = dolfin.Function(coupling.mech_solver.state_space.sub(0).collapse())
# xdmf.write_checkpoint(u, "u", 0.0, dolfin.XDMFFile.Encoding.HDF5, True)

pulse.iterate.iterate(
    problem=runner.coupling.mech_solver,
    control=runner.coupling.mech_solver.bcs.neumann[0].traction,
    target=3,  # Set it to 3 kPa
    initial_number_of_steps=50,
)

# u.assign(coupling.mech_solver.state.split(deepcopy=True)[0])
# xdmf.write_checkpoint(u, "u", 1.0, dolfin.XDMFFile.Encoding.HDF5, True)

# V = dolfin.FunctionSpace(coupling.geometry.mechanics_mesh, "CG", 1)
# Ta = dolfin.Function(V)


# class _Land(coupling.mech_solver.material.active.__class__):
#     Ta = Ta


# coupling.mech_solver.material.active.__class__ = _Land
# coupling.mech_solver._init_forms()

# pulse.iterate.iterate(
#     problem=runner.coupling.mech_solver,
#     control=Ta,
#     target=60.0,
#     initial_number_of_steps=100,
# )

# u.assign(coupling.mech_solver.state.split(deepcopy=True)[0])
# xdmf.write_checkpoint(u, "u", 2.0, dolfin.XDMFFile.Encoding.HDF5, True)
# Now we run the EM simulation.

runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)


# And save the output to xdmf-files that can be viewed in Paraview
#

simcardems.postprocess.make_xdmffiles(outdir.joinpath("results.h5"), names=["u"])
