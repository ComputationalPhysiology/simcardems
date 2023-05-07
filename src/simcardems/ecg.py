from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

import dolfin
import ufl

from . import ep_model
from .config import Config


class ECG:
    def __init__(self, torso_mesh: dolfin.Mesh) -> None:
        self.torso_mesh = torso_mesh

    @staticmethod
    def default_markers() -> Dict[str, List[int]]:
        return {
            "Heart": [7],
            "Torso": [8],
        }

    ### --- ECG recovery --- ###
    # \phi_e = 1/(4*pi*\sigma_t) \int_{\heart} div( \sigma_i* grad(vm) ) / ||r||
    # where r = distance between source (center of the tissue ?) and field points
    def ecg_recovery(
        self,
        config: Config,
        em_coupling,
    ) -> Dict[str, dolfin.Function]:
        phi_e_dict: Dict[str, dolfin.Function] = {}

        # Intracellular conductivity
        self.sigma_i = em_coupling.ep_solver._model.intracellular_conductivity()
        # Torso conductivity (assumed isotropic)
        # FIXME : Should be changeable from config
        self.sigma_t = ep_model.default_conductivities()["sigma_t"]
        # transmembrane potential = \phi_i - \phi_e
        self.vm = em_coupling.ep_solver.vs[0]

        sigma = ufl.as_matrix(self.sigma_i)
        grad_vm = ufl.as_vector(ufl.grad(self.vm))

        electrodes_markers = self.electrodes_marking(config.ecg_electrodes_file)
        for tag, e in enumerate(electrodes_markers):
            distance = self.EikonalDistance(
                em_coupling,
                boundary_parts=electrodes_markers[e],
                BoundaryIDs=[tag + 1],
                volume_parts=em_coupling.geometry.cfun,
                id_myo=em_coupling.geometry.markers["Myocardium"][0],
            )
            # with dolfin.XDMFFile("distance_to_" + e + ".xdmf") as xdmf:
            #     xdmf.write(distance)

            int_heart_expr = (ufl.div(sigma * grad_vm) / distance) * ufl.dx
            int_heart = dolfin.assemble(int_heart_expr)

            phi_e = 1 / (4 * ufl.pi * self.sigma_t) * int_heart
            phi_e_dict[e] = phi_e

        return phi_e_dict

    ### --- Augmented-monodomain --- ##
    # This function is used to computed a "corrected" conductivity
    # combining sigma_i (intracellular) with the torso conductivity
    # tensor - to be used in the monodomain solve and ecg recovery
    def augmented_bidomain(self):
        print("Not implemented yet")
        return None

    ### --- Pseudo-bidomain --- ##
    # This function is called inside the ep_model to solve
    # additional problem to approximate the extracellular potential \phi_e
    # without solving the full bidomain model
    # Question : Should this be implemented into cbcbeat instead ?
    def pseudo_bidomain(self):
        print("Not implemented yet")
        return None

    ### --- Utils --- ###
    # Take the coordinates of the electrode, mark the region (facets point if it is a mech vertex, or if not the element containing the point)
    def electrodes_marking(
        self,
        electrodes_path: Union[Path, str],
    ) -> Dict[str, dolfin.MeshFunction]:
        electrodes_markers = {}
        if electrodes_path is not None:
            electrodes_pts = json.loads(Path(electrodes_path).read_text())

        marker = dolfin.MeshFunction("size_t", self.torso_mesh, 2, 0)
        for tag, (name, coords) in enumerate(electrodes_pts.items()):
            is_point = False
            for idx, val in enumerate(list(self.torso_mesh.coordinates())):
                if (val == coords).all():
                    is_point = True
                    pt_entity = dolfin.MeshEntity(self.torso_mesh, 0, idx)
                    facets = dolfin.facets(pt_entity)
                    for f in facets:
                        marker[f.index()] = tag + 1
                    continue
            if not is_point:
                bbt = self.torso_mesh.bounding_box_tree()
                collision_idx = bbt.compute_first_entity_collision(dolfin.Point(coords))
                cell_entity = dolfin.MeshEntity(self.torso_mesh, 3, collision_idx)
                facets = dolfin.facets(cell_entity)
                # FIXME : Only mark the facets which are on the boundary (facet.exterior() ?)?
                for f in facets:
                    marker[f.index()] = tag + 1

            electrodes_markers[name] = marker
            with dolfin.XDMFFile("electrode_" + name + ".xdmf") as xdmf:
                xdmf.write(electrodes_markers[name])

        return electrodes_markers

    # Compute distance to the boundary as a function of the tissue
    # From : https://bitbucket.org/Epoxid/femorph/src/c7317791c8f00d70fe16d593344cb164a53cad9b/femorph/Legacy/SAD_MeshDeform.py?at=dokken%2Frestructuring
    def EikonalDistance(
        self,
        em_coupling,
        boundary_parts=None,
        BoundaryIDs=[],
        volume_parts=None,
        id_myo=7,
        stab=25,
        deg=2,
        UseLU=True,
    ):
        mesh = em_coupling.geometry._mesh
        heart_mesh = em_coupling.geometry.ep_mesh
        V = dolfin.FunctionSpace(mesh, "CG", deg)
        heart_V = dolfin.FunctionSpace(heart_mesh, "CG", deg)
        v = dolfin.TestFunction(V)
        u = dolfin.TrialFunction(V)
        f = dolfin.Constant(1.0)
        dist = dolfin.Function(V)
        bc = []

        if BoundaryIDs == []:
            bc = dolfin.DirichletBC(V, dolfin.Constant(0.0), "on_boundary")
        else:
            for i in BoundaryIDs:
                bc.append(
                    dolfin.DirichletBC(V, dolfin.Constant(0.0), boundary_parts, i),
                )
        dc = ufl.dx(domain=mesh)
        F1 = (ufl.inner(ufl.grad(u), ufl.grad(v)) - f * v) * dc
        # FIXME : Could we fine a way to integrate / compute the distance only
        # in the myocardium ?
        # dc = ufl.dx(domain=mesh, subdomain_data=volume_parts)
        # F1 = (ufl.inner(ufl.grad(u), ufl.grad(v)) - f * v) * dc(id_myo)

        Problem1 = dolfin.LinearVariationalProblem(
            dolfin.lhs(F1),
            dolfin.rhs(F1),
            dist,
            bc,
        )
        Solver1 = dolfin.LinearVariationalSolver(Problem1)

        # FIXME
        if UseLU:
            Solver1.parameters["linear_solver"] = "lu"
        else:
            Solver1.parameters["linear_solver"] = "cg"
            Solver1.parameters["preconditioner"] = "default"
        Solver1.solve()

        # Stabilized Eikonal equation
        eps = dolfin.Constant(mesh.hmax() / stab)
        F = (
            ufl.sqrt(ufl.inner(ufl.grad(dist), ufl.grad(dist))) * v
            - f * v
            + eps * ufl.inner(ufl.grad(abs(dist)), ufl.grad(v))
        ) * ufl.dx

        J = ufl.derivative(F, dist)
        Problem2 = dolfin.NonlinearVariationalProblem(F, dist, bc, J)
        Solver2 = dolfin.NonlinearVariationalSolver(Problem2)
        if UseLU:
            Solver2.parameters["newton_solver"]["linear_solver"] = "lu"
        else:
            Solver2.parameters["newton_solver"]["linear_solver"] = "gmres"
        Solver2.parameters["newton_solver"]["preconditioner"] = "default"
        Solver2.solve()
        dist.rename("b", "dist")

        # Interpolating from parent (heart + torso) mesh to heart mesh
        dist_heart = dolfin.interpolate(dist, heart_V)
        return dist_heart

    def write_to_file(
        self,
        phi_e_dict: Dict[float, Dict[str, dolfin.Function]],
        path: Union[str, Path],
        file_format: str = "json",
    ) -> None:
        import pandas as pd

        df = pd.DataFrame(data=phi_e_dict)
        if file_format == "json":
            df.to_json(str(path) + ".json", orient="columns")
        elif file_format == "csv":
            df.to_csv(str(path) + ".csv")
        else:
            print("ECG output format needs to be json or csv")
