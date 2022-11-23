import json
from pathlib import Path

import ap_features as apf
import dolfin
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from . import utils
from .datacollector import DataGroups
from .datacollector import DataLoader

logger = utils.getLogger(__name__)


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


def extract_traces(loader: DataLoader, reduction: str = "average"):

    all_names = {
        "mechanics": [
            "lmbda",
            "Ta",
            "Zetas_mech",
            "Zetaw_mech",
            "XS_mech",
            "XW_mech",
            "u",
        ],
        "ep": ["V", "Ca", "XS", "XW", "CaTrpn", "TmB", "Cd", "Zetas", "Zetaw"],
    }

    values = {
        group: {
            name: np.zeros(len(loader.time_stamps)) for name in names if name != "u"
        }
        for group, names in all_names.items()
    }

    values["mechanics"]["u"] = np.zeros((len(loader.time_stamps), 3))
    values["time"] = np.array(loader.time_stamps, dtype=float)

    logger.info("Extract traces...")

    for group, names in all_names.items():
        logger.info(f"Group: {group}")
        # value_point = getattr(bnd[group], utils.enum2str(point, BoundaryNodes))
        datagroup = getattr(DataGroups, utils.enum2str(group, DataGroups))
        for name in names:
            logger.info(f"Name: {name}")
            for i, t in enumerate(loader.time_stamps):
                values[group][name][i] = loader.extract_value(
                    datagroup,
                    name,
                    t,
                    reduction=reduction,
                )

    values["mechanics"]["inv_lmbda"] = 1 - values["mechanics"]["lmbda"]
    return values


def plot_state_traces(results_file: utils.PathLike, reduction: str = "average"):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    fig2, ax2 = plt.subplots(2, 4, figsize=(10, 8), sharex=True)
    results_file = Path(results_file)
    if not results_file.is_file():
        raise FileNotFoundError(f"File {results_file} does not exist")

    outdir = results_file.parent

    loader = DataLoader(results_file)
    values = extract_traces(loader, reduction=reduction)

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
                    xdmf.write_checkpoint(
                        f,
                        name,
                        float(t),
                        dolfin.XDMFFile.Encoding.HDF5,
                        True,
                    )

            except RuntimeError:
                logger.info(f"Could not save {name}")


def plot_population(results, outdir, num_models, reset_time=True):
    plt.rcParams["svg.fonttype"] = "none"
    plt.rc("axes", labelsize=13)

    times = np.array(results["m1"]["time"], dtype=float)
    if reset_time:
        times = times - times[0]
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for PoMm in range(1, num_models + 1):
        ax[0, 0].plot(
            times,
            np.array(results[f"m{PoMm}"]["mechanics"]["lmbda"], dtype=float),
        )
        ax[0, 1].plot(
            times,
            np.array(results[f"m{PoMm}"]["mechanics"]["Ta"], dtype=float),
        )
        ax[1, 0].plot(times, np.array(results[f"m{PoMm}"]["ep"]["V"], dtype=float))
        ax[1, 1].plot(
            times,
            np.array(results[f"m{PoMm}"]["ep"]["Ca"], dtype=float) * 1000,
        )

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


def extract_last_beat(y, time, pacing, return_interval=False):

    allbeats = apf.Beats(y=y, t=time, pacing=pacing)

    try:
        lastbeat = allbeats.beats[-1]

        interval = allbeats.intervals[-1]
    except apf.chopping.EmptyChoppingError:
        lastbeat = allbeats
        interval = (time[0], time[-1])

    start = next(i for i, t in enumerate(time) if i >= interval[0])
    try:
        end = next(i for i, t in enumerate(time) if i >= interval[1])
    except StopIteration:
        end = len(time)

    if return_interval:
        return lastbeat.y, lastbeat.t, (start, end)
    return lastbeat.y, lastbeat.t


def extract_biomarkers(V, Ta, time, Ca, lmbda, inv_lmbda, u):
    d = {}
    d["maxTa"] = np.max(Ta)
    d["ampTa"] = np.max(Ta) - np.min(Ta)
    d["APD40"] = find_duration(V, time, 40)
    d["APD50"] = find_duration(V, time, 50)
    d["APD90"] = find_duration(V, time, 90)
    d["triangulation"] = d["APD90"] - d["APD40"]
    d["Vpeak"] = np.max(V)
    d["Vmin"] = np.min(V)
    d["dvdt"] = (V[1] - V[0]) / 2.0
    d["maxCa"] = np.max(Ca)
    d["ampCa"] = np.max(Ca) - np.min(Ca)
    d["CaTD50"] = find_duration(Ca, time, 50)
    d["CaTD80"] = find_duration(Ca, time, 80)
    d["CaTD90"] = find_duration(Ca, time, 90)
    d["ttp_Ta"] = find_ttp(Ta, time)
    d["rt50_Ta"] = find_decaytime(Ta, time, 50)
    d["rt95_Ta"] = find_decaytime(Ta, time, 5)
    d["maxlmbda"] = np.max(lmbda)
    d["minlmbda"] = np.min(lmbda)
    d["ttplmbda"] = find_ttp(inv_lmbda, time)
    d["lmbdaD50"] = find_duration(inv_lmbda, time, 50)
    d["lmbdaD80"] = find_duration(inv_lmbda, time, 80)
    d["lmbdaD90"] = find_duration(inv_lmbda, time, 90)
    d["rt50_lmbda"] = find_decaytime(inv_lmbda, time, 50)
    d["rt95_lmbda"] = find_decaytime(inv_lmbda, time, 5)

    u_norm = np.linalg.norm(u, axis=1)
    ux, uy, uz = u.T
    for name, arr in zip(["norm", "x", "y", "z"], [u_norm, ux, uy, uz]):
        d[f"max_displacement_{name}"] = np.abs(np.max(arr))
        d[f"rel_max_displacement_{name}"] = np.abs(
            np.min(arr) - np.max(arr),
        )
        d[f"max_displacement_perc_{name}"] = d[f"max_displacement_{name}"] * 100 / 20.0
        d[f"rel_max_displacement_perc_{name}"] = (
            d[f"rel_max_displacement_{name}"] * 100 / (20.0 - np.abs(np.max(arr)))
        )
        d[f"time_to_max_displacement_{name}"] = (
            time[np.where(arr == np.min(arr))[0][0]] % 1000
        )
    return d


def get_biomarkers(results, outdir, num_models):
    biomarker_dict = {}
    fig, ax = plt.subplots()
    for PoMm in range(1, num_models + 1):
        biomarker_dict[f"m{PoMm}"] = {}
        # Create numpy arrays for analysis
        time = results[f"m{PoMm}"]["time"]
        V = results[f"m{PoMm}"]["ep"]["V"]
        Ca = results[f"m{PoMm}"]["ep"]["Ca"]
        Ta = results[f"m{PoMm}"]["mechanics"]["Ta"]
        lmbda = results[f"m{PoMm}"]["mechanics"]["lmbda"]
        u = results[f"m{PoMm}"]["mechanics"]["u"]
        inv_lmbda = results[f"m{PoMm}"]["mechanics"]["inv_lmbda"]

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
            V, timelb, interval = extract_last_beat(
                V,
                time,
                pacing,
                return_interval=True,
            )
            Ca = Ca[interval[0] : interval[1]]
            Ta = Ta[interval[0] : interval[1]]
            lmbda = lmbda[interval[0] : interval[1]]

            u = u[interval[0] : interval[1], :]
            inv_lmbda = inv_lmbda[interval[0] : interval[1]]
            time = timelb[interval[0] : interval[1]]

            figlast, axlast = plt.subplots()
            axlast.plot(time, V)
            axlast.set_xlabel("Time (ms)")
            axlast.set_ylabel("Voltage (mV)")
            figlast.savefig(outdir.joinpath("AP_lastbeat.png"), dpi=300)

        biomarker_dict[f"m{PoMm}"] = extract_biomarkers(
            V=V,
            Ta=Ta,
            time=time,
            Ca=Ca,
            lmbda=lmbda,
            inv_lmbda=inv_lmbda,
            u=u,
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


def json_serial(obj):
    if isinstance(obj, (np.ndarray)):
        return obj.tolist()
    else:
        try:
            return str(obj)
        except Exception:
            raise TypeError("Type %s not serializable" % type(obj))


def numpyfy(d):

    if isinstance(d, (list, tuple)):
        return np.array(d)
    if np.isscalar(d):
        return d

    new_d = {}
    for k, v in d.items():
        new_d[k] = numpyfy(v)
    return new_d


def save_popu_json(population_folder, num_models):
    population_folder = Path(population_folder)
    outfile = population_folder.joinpath("output_dict_center.json")
    if outfile.is_file():
        print("Load json file")
        with open(outfile) as f:
            results = numpyfy(json.load(f))

        if len(results.keys()) != num_models:
            raise Exception(
                "Number of models in json-file is not equal to number of models in cli",
            )
    else:
        results = {}
        for PoMm in range(1, num_models + 1):
            print(f"Analyzing model {PoMm}")
            results_file = population_folder.joinpath("results.h5")
            if not results_file.is_file():
                results_file = population_folder.joinpath(f"m{PoMm}/results.h5")
                if not results_file.is_file():
                    raise FileNotFoundError(f"File {results_file} does not exist")

            loader = DataLoader(results_file)
            results[f"m{PoMm}"] = extract_traces(loader=loader)

        # Save entire dict to json file in outdir(=population_folder)
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2, default=json_serial)

    print("Start plotting")
    plot_population(results, population_folder, num_models)

    print("Start analysis of single node results")
    get_biomarkers(results, population_folder, num_models)
