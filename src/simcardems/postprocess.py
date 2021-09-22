import warnings

import dolfin
import h5py
import matplotlib.pyplot as plt
import numpy as np


def center_func(fmin, fmax):
    return fmin + (fmax - fmin) / 2


class Boundary:
    def __init__(self, mesh):
        self.mesh = mesh

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

    @property
    def node_A1B(self):
        return [
            self.boundaries["max_x"] * 1.0 / 4.0,
            self.boundaries["min_y"],
            self.boundaries["min_z"],
        ]

    @property
    def node_A2B(self):
        return [
            self.boundaries["max_x"] * 2.0 / 4.0,
            self.boundaries["min_y"],
            self.boundaries["min_z"],
        ]

    @property
    def node_A3B(self):
        return [
            self.boundaries["max_x"] * 3.0 / 4.0,
            self.boundaries["min_y"],
            self.boundaries["min_z"],
        ]

    @property
    def node_D1C(self):
        return [
            self.boundaries["max_x"] * 1.0 / 4.0,
            self.boundaries["min_y"],
            self.boundaries["max_z"],
        ]

    @property
    def node_D2C(self):
        return [
            self.boundaries["max_x"] * 2.0 / 4.0,
            self.boundaries["min_y"],
            self.boundaries["max_z"],
        ]

    @property
    def node_D3C(self):
        return [
            self.boundaries["max_x"] * 3.0 / 4.0,
            self.boundaries["min_y"],
            self.boundaries["max_z"],
        ]

    @property
    def node_E1F(self):
        return [
            self.boundaries["max_x"] * 1.0 / 4.0,
            self.boundaries["max_y"],
            self.boundaries["min_z"],
        ]

    @property
    def node_E2F(self):
        return [
            self.boundaries["max_x"] * 2.0 / 4.0,
            self.boundaries["max_y"],
            self.boundaries["min_z"],
        ]

    @property
    def node_E3F(self):
        return [
            self.boundaries["max_x"] * 3.0 / 4.0,
            self.boundaries["max_y"],
            self.boundaries["min_z"],
        ]

    @property
    def node_H1G(self):
        return [
            self.boundaries["max_x"] * 1.0 / 4.0,
            self.boundaries["max_y"],
            self.boundaries["max_z"],
        ]

    @property
    def node_H2G(self):
        return [
            self.boundaries["max_x"] * 2.0 / 4.0,
            self.boundaries["max_y"],
            self.boundaries["max_z"],
        ]

    @property
    def node_H3G(self):
        return [
            self.boundaries["max_x"] * 3.0 / 4.0,
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

    if h5py.h5.get_config().mpi:
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
        "node_A1B": None,
        "node_A2B": None,
        "node_A3B": None,
        "node_D1C": None,
        "node_D2C": None,
        "node_D3C": None,
        "node_E1F": None,
        "node_E2F": None,
        "node_E3F": None,
        "node_H1G": None,
        "node_H2G": None,
        "node_H3G": None,
    }

    with dolfin.HDF5File(mesh.mpi_comm(), file, "r") as h5file:
        for data_node in data.keys():
            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                print("analyzing: ", data_node)

            # Assign the variables to be stored in the dictionary
            data[data_node] = {
                "V": np.zeros(len(time_points)),
                "Cai": np.zeros(len(time_points)),
                "Ta": np.zeros(len(time_points)),
                "stretch": np.zeros(len(time_points)),
            }

            # Loop over all variables to be stored
            for nestedkey in data[data_node]:
                if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                    print("analyzing: ", nestedkey)
                for i, t in enumerate(time_points):
                    h5file.read(
                        v_space,
                        "/" + nestedkey + "/{0:.2f}/".format(float(t)),
                    )
                    data_temp = v_space(eval("bnd." + data_node))
                    data[data_node][nestedkey][i] = data_temp
    return data


class Analysis:
    def plot_peaks(outdir, data, threshold):
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
        ax.set_xlabel("Numer of beats")
        ax.set_ylabel("% change from previous beat")
        fig.savefig(outdir + "/compare-peak-values.png", dpi=300)
