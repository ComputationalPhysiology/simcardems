from __future__ import annotations

from typing import Dict
from typing import List
from typing import TYPE_CHECKING

import dolfin
import ufl

# from .config import Config

if TYPE_CHECKING:
    from .models.em_model import EMCoupling

# logger = utils.getLogger(__name__)


class ECG:
    def __init__(self, em_model: EMCoupling, torso_mesh: dolfin.Mesh, sigma_t) -> None:
        self.em_coupling = em_model
        self.torso_mesh = torso_mesh
        self.sigma_t = sigma_t
        self.sigma_i = self.em_coupling.ep_solver.intracellular_conductivity()

    @staticmethod
    def default_markers() -> Dict[str, List[int]]:
        return {
            "Heart": [7],
            "Torso": [8],
        }

    @property
    def em_coupling(self) -> EMCoupling:
        return self.em_coupling

    ### --- ECG recovery --- ###
    # Recover \phi_e (extrac-cellular potential), as :
    # \phi_e = 1/(4*pi*\sigma_b) \int_{\heart} \beta*I_m / ||r||, where :

    # \sigma_b = bath conductivity tensor
    # (bath = unbounded volume conductor the tissue is immersed in)
    # \beta = bidomain membrane surface to volume ratio
    # I_m = transmembrane currents
    # \beta * I_m = div(\sigma_i) * grad(vm)
    # with \sigma_i (intracellular / tissue conductivity tensor)
    # and vm (transmembrane potential)
    # r : distance between source (center of the tissue ?) and field points
    def ecg_recovery(
        electrodes_pts: Dict[str, List[float]],
    ) -> Dict[str, dolfin.Function]:
        print("Not implemented yet")
        # self.phi_e = ...
        return None

    ### --- Augmented-monodomain --- ##
    # This function is used to computed a "corrected" conductivity
    # combining sigma_i (intracellular) with the torso conductivity
    # tensor - to be used in the monodomain solve and ecg recovery
    def augmented_bidomain(self):
        print("Not implemented yet")
        # self.phi_e = ...
        return None

    ### --- Pseudo-bidomain --- ##
    # This function is called inside the ep_model to solve (unfrequently)
    # additional problem to approximate the extracellular potential \phi_e
    # without solving the full bidomain model
    # Question : Should this be implemented into cbcbeat instead ?
    def pseudo_bidomain(self):
        print("Not implemented yet")
        # self.phi_e = ...
        return None

    def save_ecg(self, path):
        # Saving phi_e (for each electrode location) to hdf5 file
        return None

    ### --- Utils --- ###
    # Take the coordinates of the electrode, mark the region (point if it is a mech vertex, or if not the element containing the point)
    def electrodes_marking(
        electrodes_pts: Dict[str, List[float]],
    ) -> Dict[str, dolfin.MeshFunction]:
        electrodes_markers = {}
        return electrodes_markers

    # Compute distance to the boundary as a function of the tissue
    # From : https://bitbucket.org/Epoxid/femorph/src/c7317791c8f00d70fe16d593344cb164a53cad9b/femorph/Legacy/SAD_MeshDeform.py?at=dokken%2Frestructuring
    # But this would require embedding the tissue mesh into the torso mesh
    def EikonalDistance(
        em_coupling,
        boundary_parts=None,
        BoundaryIDs=[],
        stab=25,
        deg=2,
        UseLU=True,
    ):
        V = dolfin.FunctionSpace(em_coupling.ep_solver.domain(), "CG", deg)
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

        # dc = dx(domain=mesh)
        F1 = (ufl.inner(ufl.grad(u), ufl.grad(v)) - f * v) * ufl.dx

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
        eps = dolfin.Constant(em_coupling.ep_solver.domain().hmax() / stab)
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

        return dist
