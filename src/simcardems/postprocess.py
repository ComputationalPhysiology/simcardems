import json
import warnings
from pathlib import Path

import ap_features as apf
import dolfin
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from . import utils
from .datacollector import DataLoader

logger = utils.getLogger(__name__)


def center_func(fmin, fmax):
    return fmin + (fmax - fmin) / 2


class Boundary:
    def __init__(self, mesh):
        self.mesh = mesh

    @staticmethod
    def nodes():
        return (
            "center",
            "node_A",
            "node_B",
            "node_C",
            "node_D",
            "node_E",
            "node_F",
            "node_G",
            "node_H",
            "xmax",
            "xmin",
            "ymax",
            "ymin",
            "zmax",
            "zmin",
        )

    @property
    def boundaries(self):
        coords = self.mesh.coordinates()
        return dict(
            max_x=coords.T[0].max(),
            min_x=coords.T[0].min(),
            max_y=coords.T[1].max(),
            min_y=coords.T[1].min(),
            max_z=coords.T[2].max(),
            min_z=coords.T[2].min(),
        )

    @property
    def xmin(self):
        return [
            self.boundaries["min_x"],
            center_func(self.boundaries["min_y"], self.boundaries["max_y"]),
            center_func(self.boundaries["min_z"], self.boundaries["max_z"]),
        ]

    @property
    def xmax(self):
        return [
            self.boundaries["max_x"],
            center_func(self.boundaries["min_y"], self.boundaries["max_y"]),
            center_func(self.boundaries["min_z"], self.boundaries["max_z"]),
        ]

    @property
    def ymin(self):
        return [
            center_func(self.boundaries["min_x"], self.boundaries["max_x"]),
            self.boundaries["min_y"],
            center_func(self.boundaries["min_z"], self.boundaries["max_z"]),
        ]

    @property
    def ymax(self):
        return [
            center_func(self.boundaries["min_x"], self.boundaries["max_x"]),
            self.boundaries["max_y"],
            center_func(self.boundaries["min_z"], self.boundaries["max_z"]),
        ]

    @property
    def zmin(self):
        return [
            center_func(self.boundaries["min_x"], self.boundaries["max_x"]),
            center_func(self.boundaries["min_y"], self.boundaries["max_y"]),
            self.boundaries["min_z"],
        ]

    @property
    def zmax(self):
        return [
            center_func(self.boundaries["min_x"], self.boundaries["max_x"]),
            center_func(self.boundaries["min_y"], self.boundaries["max_y"]),
            self.boundaries["max_z"],
        ]

    @property
    def center(self):
        return [
            center_func(self.boundaries["min_x"], self.boundaries["max_x"]),
            center_func(self.boundaries["min_y"], self.boundaries["max_y"]),
            center_func(self.boundaries["min_z"], self.boundaries["max_z"]),
        ]

    @property
    def node_A(self):
        return [
            self.boundaries["min_x"],
            self.boundaries["min_y"],
            self.boundaries["min_z"],
        ]

    @property
    def node_B(self):
        return [
            self.boundaries["max_x"],
            self.boundaries["min_y"],
            self.boundaries["min_z"],
        ]

    @property
    def node_C(self):
        return [
            self.boundaries["max_x"],
            self.boundaries["min_y"],
            self.boundaries["max_z"],
        ]

    @property
    def node_D(self):
        return [
            self.boundaries["min_x"],
            self.boundaries["min_y"],
            self.boundaries["max_z"],
        ]

    @property
    def node_E(self):
        return [
            self.boundaries["min_x"],
            self.boundaries["max_y"],
            self.boundaries["min_z"],
        ]

    @property
    def node_F(self):
        return [
            self.boundaries["max_x"],
            self.boundaries["max_y"],
            self.boundaries["min_z"],
        ]

    @property
    def node_G(self):
        return [
            self.boundaries["max_x"],
            self.boundaries["max_y"],
            self.boundaries["max_z"],
        ]

    @property
    def node_H(self):
        return [
            self.boundaries["min_x"],
            self.boundaries["max_y"],
            self.boundaries["max_z"],
        ]


def load_mesh(file):
    # load the mesh from the results file
    mesh = dolfin.Mesh()

    with dolfin.HDF5File(mesh.mpi_comm(), file, "r") as h5file:
        h5file.read(mesh, "/mesh", False)

    bnd = Boundary(mesh)

    return mesh, bnd


def load_times(filename):
    from mpi4py import MPI

    # Find time points
    time_points = None

    if h5py.h5.get_config().mpi and dolfin.MPI.size(dolfin.MPI.comm_world) > 1:
        h5file = h5py.File(filename, "r", driver="mpio", comm=MPI.COMM_WORLD)
    else:
        if dolfin.MPI.size(dolfin.MPI.comm_world) > 1:
            warnings.warn("h5py is not installed with MPI support")
        h5file = h5py.File(filename, "r")

    time_points = list(h5file["V"].keys())
    # Make sure they are sorted
    time_points = sorted(time_points, key=lambda x: float(x))

    if time_points is None:
        raise IOError("No results found")

    h5file.close()

    return time_points


def load_data(file, mesh, bnd, time_points):
    V = dolfin.FunctionSpace(mesh, "CG", 1)

    v_space = dolfin.Function(V)

    # Create a dictionary to assign all values to
    data = {
        "node_A": None,
        "node_B": None,
        "node_C": None,
        "node_D": None,
        "node_E": None,
        "node_F": None,
        "node_G": None,
        "node_H": None,
    }

    with dolfin.HDF5File(mesh.mpi_comm(), file, "r") as h5file:
        for data_node in data.keys():
            logger.info("analyzing: ", data_node)

            # Assign the variables to be stored in the dictionary
            data[data_node] = {
                "V": np.zeros(len(time_points)),
                "Cai": np.zeros(len(time_points)),
                "Ta": np.zeros(len(time_points)),
                "stretch": np.zeros(len(time_points)),
            }

            # Loop over all variables to be stored
            for nestedkey in data[data_node]:
                logger.info("analyzing: ", nestedkey)
                for i, t in enumerate(time_points):
                    h5file.read(
                        v_space,
                        "/" + nestedkey + "/{0:.2f}/".format(float(t)),
                    )
                    data_temp = v_space(eval("bnd." + data_node))
                    data[data_node][nestedkey][i] = data_temp
    return data


def plot_peaks(fname, data, threshold):
    # Find peaks for assessment steady state
    from scipy.signal import find_peaks

    peak_indices = find_peaks(data, height=threshold)

    for i, idx in enumerate(peak_indices[0]):
        if i == 0:
            x = [idx]
            y = [data[idx]]
        else:
            x.append(idx)
            y.append(data[idx])

    # Calculate difference between consecutive list elements
    change_y = [(s - q) / q * 100 for q, s in zip(y, y[1:])]

    fig, ax = plt.subplots()
    ax.plot(change_y)
    ax.set_title("Compare peak values")
    ax.grid()
    ax.set_xlabel("Number of beats")
    ax.set_ylabel("% change from previous beat")
    fig.savefig(fname, dpi=300)


def plot_state_traces(results_file):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    fig2, ax2 = plt.subplots(2, 4, figsize=(10, 8), sharex=True)
    results_file = Path(results_file)
    if not results_file.is_file():
        raise FileNotFoundError(f"File {results_file} does not exist")

    outdir = results_file.parent

    loader = DataLoader(results_file)
    bnd = {"ep": Boundary(loader.ep_mesh), "mechanics": Boundary(loader.mech_mesh)}

    all_names = {
        "mechanics": ["lmbda", "Ta", "Zetas_mech", "Zetaw_mech", "XS_mech", "XW_mech"],
        "ep": ["V", "Ca", "XS", "XW", "CaTrpn", "TmB", "Cd", "Zetas", "Zetaw"],
    }

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
                    np.linalg.norm(dof_coords - np.array(bnd[group].center), axis=1),
                )
                if np.isclose(dof_coords[dof], np.array(bnd[group].center)).all():
                    # If we have a dof at the center - evaluation at dof (cheaper)
                    values[group][name][i] = func.vector().get_local()[dof]
                else:
                    # Otherwise, evaluation at center coordinates
                    try:
                        values[group][name][i] = func(bnd[group].center)
                    except RuntimeError:
                        values[group][name][i] = func.vector().get_local()[dof]

    times = np.array(loader.time_stamps, dtype=float)

    if times[-1] > 4000 and False:
        plot_peaks(
            outdir.joinpath("compare-peak-values.png"),
            values["ep"]["Ca"],
            0.0002,
        )

    ax[0, 0].plot(times, values["mechanics"]["lmbda"])
    ax[0, 1].plot(times, values["mechanics"]["Ta"])
    ax[1, 0].plot(times, values["ep"]["V"])
    ax[1, 1].plot(times, values["ep"]["Ca"])

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

    ax2[0, 0].plot(times, values["ep"]["XS"], linestyle="solid", color="blue")
    ax2[0, 0].plot(
        times,
        values["mechanics"]["XS_mech"],
        linestyle="dotted",
        color="orange",
    )
    ax2[0, 1].plot(times, values["ep"]["CaTrpn"], linestyle="solid", color="blue")
    ax2[0, 2].plot(times, values["ep"]["TmB"], linestyle="solid", color="blue")
    ax2[0, 3].plot(times, values["ep"]["Zetas"], linestyle="solid", color="blue")
    ax2[0, 3].plot(
        times,
        values["mechanics"]["Zetas_mech"],
        linestyle="dotted",
        color="orange",
    )
    ax2[1, 0].plot(times, values["ep"]["XW"], linestyle="solid", color="blue")
    ax2[1, 0].plot(
        times,
        values["mechanics"]["XW_mech"],
        linestyle="dotted",
        color="orange",
    )
    ax2[1, 1].plot(times, values["ep"]["Cd"], linestyle="solid", color="blue")
    ax2[1, 3].plot(times, values["ep"]["Zetaw"], linestyle="solid", color="blue")
    ax2[1, 3].plot(
        times,
        values["mechanics"]["Zetaw_mech"],
        linestyle="dotted",
        color="orange",
    )

    ax2[0, 0].set_title("XS")
    ax2[0, 1].set_title("CaTrpn")
    ax2[0, 2].set_title("TmB")
    ax2[0, 3].set_title("Zetas")
    ax2[1, 0].set_title("XW")
    ax2[1, 1].set_title("Cd")
    # ax2[1, 2].set_title("")
    ax2[1, 3].set_title("Zetaw")
    for axi in ax2.flatten():
        axi.grid()
        if False:
            axi.set_xlim([0, 5000])
    ax2[1, 0].set_xlabel("Time [ms]")
    ax2[1, 1].set_xlabel("Time [ms]")
    ax2[1, 2].set_xlabel("Time [ms]")
    ax2[1, 3].set_xlabel("Time [ms]")

    # If there is a residual.txt file: load and plot these results
    if outdir.joinpath("residual.txt").is_file():
        residual0 = np.loadtxt(outdir.joinpath("residual.txt"), usecols=0)

        residual_file = open(outdir.joinpath("residual.txt"), "r")
        residual_data = residual_file.readlines()
        residualN = []
        for line in residual_data:
            if line.strip().split("\t")[-1] != "":
                residualN.append(float(line.strip().split("\t")[-1]))
        residual_file.close()

        # Back to initial dt and time points
        dt = 0.05
        times_dt = np.arange(times[0], times[0] + residual0.size * dt, dt)

        ax00r = ax[0, 0].twinx()
        ax00r.plot(
            times_dt,
            residualN,
            "--",
            color="lightcoral",
            label="Newton residualN",
        )
        ax01r = ax[0, 1].twinx()
        ax01r.plot(
            times_dt,
            residualN,
            "--",
            color="lightcoral",
            label="Newton residualN",
        )
        ax01r.set_ylabel("Newton residual N")
        ax10r = ax[1, 0].twinx()
        ax10r.plot(
            times_dt,
            residualN,
            "--",
            color="lightcoral",
            label="Newton residualN",
        )
        ax11r = ax[1, 1].twinx()
        ax11r.plot(
            times_dt,
            residualN,
            "--",
            color="lightcoral",
            label="Newton residualN",
        )
        ax11r.set_ylabel("Newton residual N")
        fig.tight_layout()

        ax200r = ax2[0, 0].twinx()
        ax200r.plot(times_dt, residual0, "--", color="grey", label="Newton residual")
        ax201r = ax2[0, 1].twinx()
        ax201r.plot(times_dt, residual0, "--", color="grey", label="Newton residual")
        ax202r = ax2[0, 2].twinx()
        ax202r.plot(times_dt, residual0, "--", color="grey", label="Newton residual")
        ax203r = ax2[0, 3].twinx()
        ax203r.plot(times_dt, residual0, "--", color="grey", label="Newton residual")
        ax203r.set_ylabel("Newton residual 0")
        ax210r = ax2[1, 0].twinx()
        ax210r.plot(times_dt, residual0, "--", color="grey", label="Newton residual")
        ax211r = ax2[1, 1].twinx()
        ax211r.plot(times_dt, residual0, "--", color="grey", label="Newton residual")
        ax213r = ax2[1, 3].twinx()
        ax213r.plot(times_dt, residual0, "--", color="grey", label="Newton residual")
        ax213r.set_ylabel("Newton residual 0")
        fig2.tight_layout()

        # Create and save figure with Newton residual of first and last iteration
        figNR, axNR = plt.subplots(1, 1)
        axNRr = axNR.twinx()
        axNR.plot(times_dt, residual0, color="grey", label="First res")
        axNRr.plot(times_dt, residualN, color="lightcoral", label="Last residual")
        axNR.set_xlabel("Time (ms)")
        axNR.set_ylabel("Residual 1st Newton iter.")
        axNRr.set_ylabel("Residual last Newton iter.")
        axNR.legend(loc="upper left")
        axNRr.legend(loc="upper right")
        # figNR.tight_layout()
        figNR.savefig(outdir.joinpath("NewtonResidual.png"), dpi=300)

    fig.savefig(outdir.joinpath("state_traces_center.png"), dpi=300)
    fig2.savefig(outdir.joinpath("state_mech_traces_center.png"), dpi=300)


def make_xdmffiles(results_file):

    loader = DataLoader(results_file)
    outdir = Path(results_file).parent

    for group, names in loader.names.items():
        logger.info(f"Save xdmffile for group {group}")
        for name in names:
            xdmf = dolfin.XDMFFile(
                dolfin.MPI.comm_world,
                outdir.joinpath(f"{group}_{name}.xdmf").as_posix(),
            )
            logger.info(f"Save {name}")
            try:
                for t in tqdm.tqdm(loader.time_stamps):
                    f = loader.get(group, name, t)
                    xdmf.write(f, float(t))
            except RuntimeError:
                logger.info(f"Could not save {name}")


def plot_population(dict, outdir, num_models, reset_time=True):
    plt.rcParams["svg.fonttype"] = "none"
    plt.rc("axes", labelsize=13)

    times = np.array(dict["m1"]["time"], dtype=float)
    if reset_time:
        times = times - times[0]
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for PoMm in range(1, num_models + 1):
        ax[0, 0].plot(times, np.array(dict[f"m{PoMm}"]["lmbda"], dtype=float))
        ax[0, 1].plot(times, np.array(dict[f"m{PoMm}"]["Ta"], dtype=float))
        ax[1, 0].plot(times, np.array(dict[f"m{PoMm}"]["V"], dtype=float))
        ax[1, 1].plot(times, np.array(dict[f"m{PoMm}"]["Ca"], dtype=float) * 1000)

    for axi in ax.flatten():
        axi.grid()
        # axi.set_xlim([-20, 1000])
    ax[0, 0].set_ylabel("Active stretch")
    ax[0, 1].set_ylabel("Active tension (kPa)")
    ax[1, 0].set_ylabel("Voltage (mV)")
    ax[1, 0].set_xlabel("Time (ms)")
    ax[1, 1].set_ylabel(r"Intracellular calcium concentration ($\mu$M)")
    ax[1, 1].set_xlabel("Time (ms)")
    fig.savefig(outdir.joinpath("traces_center.png"), dpi=300)
    fig.savefig(outdir.joinpath("traces_center.svg"), format="svg")


def find_duration(y, t, repolarisation):
    s = apf.Beat(y=y, t=t)
    return s.apd(repolarisation)


def find_ttp(y, x):
    s = apf.Beat(y=y, t=x)
    return s.ttp()


def find_decaytime(y, x, a):
    s = apf.Beat(y=y, t=x)
    return s.tau(a / 100)


def stats(y):
    ave = sum(y) / len(y)
    SD = (sum([((x - ave) ** 2) for x in y]) / len(y)) ** 0.5
    return ave, SD


def extract_last_beat(y, time, pacing):
    allbeats = apf.Beats(y=y, t=time, pacing=pacing)
    lastbeat = allbeats.beats[-1]
    return lastbeat.y, lastbeat.t


def get_biomarkers(dict, outdir, num_models):
    biomarker_dict = {}
    fig, ax = plt.subplots()
    for PoMm in range(1, num_models + 1):
        biomarker_dict[f"m{PoMm}"] = {}
        # Create numpy arrays for analysis
        time = np.array(dict[f"m{PoMm}"]["time"], dtype=float)
        V = np.array(dict[f"m{PoMm}"]["V"], dtype=float)
        Ca = np.array(dict[f"m{PoMm}"]["Ca"], dtype=float)
        Ta = np.array(dict[f"m{PoMm}"]["Ta"], dtype=float)
        lmbda = np.array(dict[f"m{PoMm}"]["lmbda"], dtype=float)
        u = np.array(dict[f"m{PoMm}"]["u"], dtype=float)
        # Create contraction-array as 1-lambda
        inv_lmbda = np.zeros_like(lmbda)
        for i in range(0, len(lmbda)):
            inv_lmbda[i] = 1.0 - lmbda[i]

        # Select only last beat for further analysis
        onlylastbeat = True
        if onlylastbeat:
            logger.info(
                "Only extracting biomarkers for the last beat of the simulation",
            )
            # Create a list with pacing indicators
            pacing = np.zeros_like(time)
            for i in range(0, len(pacing)):
                if time[i] % 1000 <= 0.09 or i == 0:
                    pacing[i] = 1

            # Overwrite the variables with data from only the last beat
            V, timelb = extract_last_beat(V, time, pacing)
            Ca, timelb = extract_last_beat(Ca, time, pacing)
            Ta, timelb = extract_last_beat(Ta, time, pacing)
            lmbda, timelb = extract_last_beat(lmbda, time, pacing)
            u, timelb = extract_last_beat(u, time, pacing)
            inv_lmbda, timelb = extract_last_beat(inv_lmbda, time, pacing)
            time = timelb

            figlast, axlast = plt.subplots()
            axlast.plot(time, V)
            axlast.set_xlabel("Time (ms)")
            axlast.set_ylabel("Voltage (mV)")
            figlast.savefig(outdir.joinpath("AP_lastbeat.png"), dpi=300)

        biomarker_dict[f"m{PoMm}"]["maxTa"] = np.max(Ta)
        biomarker_dict[f"m{PoMm}"]["ampTa"] = np.max(Ta) - np.min(Ta)
        biomarker_dict[f"m{PoMm}"]["APD40"] = find_duration(V, time, 40)
        biomarker_dict[f"m{PoMm}"]["APD50"] = find_duration(V, time, 50)
        biomarker_dict[f"m{PoMm}"]["APD90"] = find_duration(V, time, 90)
        biomarker_dict[f"m{PoMm}"]["triangulation"] = (
            biomarker_dict[f"m{PoMm}"]["APD90"] - biomarker_dict[f"m{PoMm}"]["APD40"]
        )
        biomarker_dict[f"m{PoMm}"]["Vpeak"] = np.max(V)
        biomarker_dict[f"m{PoMm}"]["Vmin"] = np.min(V)
        biomarker_dict[f"m{PoMm}"]["dvdt"] = (V[1] - V[0]) / 2.0
        biomarker_dict[f"m{PoMm}"]["maxCa"] = np.max(Ca)
        biomarker_dict[f"m{PoMm}"]["ampCa"] = np.max(Ca) - np.min(Ca)
        biomarker_dict[f"m{PoMm}"]["CaTD50"] = find_duration(Ca, time, 50)
        biomarker_dict[f"m{PoMm}"]["CaTD80"] = find_duration(Ca, time, 80)
        biomarker_dict[f"m{PoMm}"]["CaTD90"] = find_duration(Ca, time, 90)
        biomarker_dict[f"m{PoMm}"]["ttp_Ta"] = find_ttp(Ta, time)
        biomarker_dict[f"m{PoMm}"]["rt50_Ta"] = find_decaytime(Ta, time, 50)
        biomarker_dict[f"m{PoMm}"]["rt95_Ta"] = find_decaytime(Ta, time, 5)
        biomarker_dict[f"m{PoMm}"]["maxlmbda"] = np.max(lmbda)
        biomarker_dict[f"m{PoMm}"]["minlmbda"] = np.min(lmbda)
        biomarker_dict[f"m{PoMm}"]["ttplmbda"] = find_ttp(inv_lmbda, time)
        biomarker_dict[f"m{PoMm}"]["lmbdaD50"] = find_duration(inv_lmbda, time, 50)
        biomarker_dict[f"m{PoMm}"]["lmbdaD80"] = find_duration(inv_lmbda, time, 80)
        biomarker_dict[f"m{PoMm}"]["lmbdaD90"] = find_duration(inv_lmbda, time, 90)
        biomarker_dict[f"m{PoMm}"]["rt50_lmbda"] = find_decaytime(inv_lmbda, time, 50)
        biomarker_dict[f"m{PoMm}"]["rt95_lmbda"] = find_decaytime(inv_lmbda, time, 5)

        biomarker_dict[f"m{PoMm}"]["max_displacement"] = np.abs(np.min(u))
        biomarker_dict[f"m{PoMm}"]["rel_max_displacement"] = np.abs(
            np.min(u) - np.max(u),
        )
        biomarker_dict[f"m{PoMm}"]["max_displacement_perc"] = (
            biomarker_dict[f"m{PoMm}"]["max_displacement"] * 100 / 20.0
        )
        biomarker_dict[f"m{PoMm}"]["rel_max_displacement_perc"] = (
            biomarker_dict[f"m{PoMm}"]["rel_max_displacement"]
            * 100
            / (20.0 - np.abs(np.max(u)))
        )
        biomarker_dict[f"m{PoMm}"]["time_to_max_displacement"] = (
            time[np.where(u == np.min(u))[0][0]] % 1000
        )

        ax.plot(PoMm, biomarker_dict[f"m{PoMm}"]["APD90"], "*")

    fig.savefig(outdir.joinpath("APD90_permodel.png"), dpi=300)

    with open(outdir.joinpath("biomarkers_PoMcontrol.json"), "w") as f:
        json.dump(biomarker_dict, f)

    # Get average and standard deviation of each biomarker
    biomarker_stats = {}
    for biomarker in biomarker_dict["m1"].keys():
        biomarker_stats[biomarker] = []
        for PoMm in biomarker_dict.keys():
            biomarker_stats[biomarker].append(biomarker_dict[PoMm][biomarker])

        ave, SD = stats(biomarker_stats[biomarker])
        print("Average {} ± SD = {} ± {} ".format(biomarker, ave, SD))

    return biomarker_dict


def save_popu_json(population_folder, num_models):
    population_folder = Path(population_folder)
    if population_folder.joinpath("output_dict_center.json").is_file():
        print("Load json file")
        f = open(population_folder.joinpath("output_dict_center.json"))
        dict = json.load(f)
        f.close()
        if len(dict.keys()) != num_models:
            raise Exception(
                "Number of models in json-file is not equal to number of models in cli",
            )
    else:
        dict = {}
        for PoMm in range(1, num_models + 1):
            print(f"Analyzing model {PoMm}")
            results_file = population_folder.joinpath("results.h5")
            if not results_file.is_file():
                results_file = population_folder.joinpath(f"m{PoMm}/results.h5")
                if not results_file.is_file():
                    raise FileNotFoundError(f"File {results_file} does not exist")

            loader = DataLoader(results_file)
            dict[f"m{PoMm}"] = {
                "time": [],
                "V": [],
                "Ca": [],
                "Ta": [],
                "lmbda": [],
                "u": [],
                "XS": [],
                "XW": [],
                "CaTrpn": [],
                "TmB": [],
                "Cd": [],
                "Zetas": [],
                "Zetaw": [],
            }

            # Save times to dictionary
            dict[f"m{PoMm}"]["time"] = loader.time_stamps

            bnd = {
                "ep": Boundary(loader.ep_mesh),
                "mechanics": Boundary(loader.mech_mesh),
            }

            all_names = {
                "mechanics": ["lmbda", "Ta", "u"],
                "ep": ["V", "Ca", "XS", "XW", "CaTrpn", "TmB", "Cd", "Zetas", "Zetaw"],
            }

            # Fill arrays with data from file
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
                        if np.isclose(
                            dof_coords[dof],
                            np.array(bnd[group].center),
                        ).all():
                            # If we have a dof at the center - evaluation at dof (cheaper)
                            dict[f"m{PoMm}"][name].append(
                                func.vector().get_local()[dof],
                            )

                        else:
                            # Otherwise, evaluation at center coordinates
                            try:
                                dict[f"m{PoMm}"][name].append(
                                    float(func(bnd[group].center)),
                                )
                            except RuntimeError:
                                dict[f"m{PoMm}"][name].append(
                                    func.vector().get_local()[dof],
                                )

        # Save entire dict to json file in outdir(=population_folder)
        with open(population_folder.joinpath("output_dict_center.json"), "w") as f:
            json.dump(dict, f)

    print("Start plotting")
    plot_population(dict, population_folder, num_models)

    print("Start analysis of single node results")
    get_biomarkers(dict, population_folder, num_models)
