import typing
from enum import Enum

import dolfin
import pulse
import ufl
from mpi4py import MPI

from . import utils

logger = utils.getLogger(__name__)


def Max(a, b):
    return (a + b + abs(a - b)) / dolfin.Constant(2)


class BoundaryConditions(str, Enum):
    dirichlet = "dirichlet"
    rigid = "rigid"


class LandModel(pulse.ActiveModel):
    def __init__(
        self,
        XS,
        XW,
        dt,
        lmbda,
        Zetas,
        Zetaw,
        parameters,
        function_space,
        f0=None,
        s0=None,
        n0=None,
        eta=0,
        **kwargs,
    ):
        super().__init__(f0=f0, s0=s0, n0=n0)
        self._eta = eta
        self.function_space = function_space

        self.XS = XS
        self.XS_prev = dolfin.Function(function_space)
        self.XW = XW
        self.XW_prev = dolfin.Function(function_space)
        self._parameters = parameters
        self.dt = dt

        self.Zetas = Zetas
        self.Zetaw = Zetaw

        self.Zetas_prev = dolfin.Function(self.Zetas.function_space())
        self.Zetas_prev_prev = dolfin.Function(self.Zetas.function_space())

        self.Zetaw_prev = dolfin.Function(self.Zetaw.function_space())
        self.Zetaw_prev_prev = dolfin.Function(self.Zetaw.function_space())

        self.Ta_current = dolfin.Function(function_space, name="Ta")
        self.lmbda_prev = dolfin.Function(function_space)
        self.lmbda_current = dolfin.Function(function_space)
        self.lmbda_current = lmbda
        self.update_prev()

    def update_prev(self):
        self.XS_prev.vector()[:] = self.XS.vector()
        self.XW_prev.vector()[:] = self.XW.vector()

        self.Zetas_prev_prev.vector()[:] = self.Zetas_prev.vector()
        self.Zetaw_prev_prev.vector()[:] = self.Zetaw_prev.vector()
        self.Zetas_prev.vector()[:] = self.Zetas.vector()
        self.Zetaw_prev.vector()[:] = self.Zetaw.vector()
        self.lmbda_prev.vector()[:] = self.lmbda_current.vector()

    def lmbda(self, F):
        f = F * self.f0
        return dolfin.sqrt(f**2)

    def dLambda(self, F):
        dLambda = (self.lmbda(F) - self.lmbda_prev) / self.dt
        return dLambda

    def _solve_ode(self, F):

        phi = self._parameters["phi"]
        Tot_A = self._parameters["Tot_A"]
        F = self._parameters["F"]
        L = self._parameters["L"]
        rad = self._parameters["rad"]
        cmdnmax = self._parameters["cmdnmax"]
        kmcmdn = self._parameters["kmcmdn"]
        trpnmax = self._parameters["trpnmax"]
        Beta1 = self._parameters["Beta1"]

        Trpn50 = self._parameters["Trpn50"]
        cat50_ref = self._parameters["cat50_ref"]

        etal = self._parameters["etal"]
        etas = self._parameters["etas"]
        gammas = self._parameters["gammas"]
        gammaw = self._parameters["gammaw"]
        ktrpn = self._parameters["ktrpn"]
        ku = self._parameters["ku"]
        kuw = self._parameters["kuw"]
        kws = self._parameters["kws"]
        # lmbda = self._parameters["lmbda"]
        ntm = self._parameters["ntm"]
        ntrpn = self._parameters["ntrpn"]
        p_k = self._parameters["p_k"]

        rs = self._parameters["rs"]
        rw = self._parameters["rw"]

        # Population factors
        scale_popu_nTm = self._parameters["scale_popu_nTm"]
        scale_popu_CaT50ref = self._parameters["scale_popu_CaT50ref"]
        scale_popu_kuw = self._parameters["scale_popu_kuw"]
        scale_popu_kws = self._parameters["scale_popu_kws"]
        scale_popu_kTRPN = self._parameters["scale_popu_kTRPN"]
        scale_popu_nTRPN = self._parameters["scale_popu_nTRPN"]
        scale_popu_ku = self._parameters["scale_popu_ku"]
        scale_popu_TRPN50 = self._parameters["scale_popu_TRPN50"]
        scale_popu_rw = self._parameters["scale_popu_rw"]
        scale_popu_rs = self._parameters["scale_popu_rs"]

        # Systolic Heart Failure (HF with preserved ejection fraction)
        HF_scaling_cat50_ref = self._parameters["HF_scaling_cat50_ref"]

        vcell = 3140.0 * L * (rad * rad)
        Ageo = 6.28 * (rad * rad) + 6.28 * L * rad
        Acap = 2 * Ageo
        vmyo = 0.68 * vcell
        vnsr = 0.0552 * vcell
        vss = 0.02 * vcell

        Aw = (
            Tot_A
            * rs
            * scale_popu_rs
            / (rs * scale_popu_rs + rw * scale_popu_rw * (1 - (rs * scale_popu_rs)))
        )
        As = Aw

        cw = (
            kuw
            * scale_popu_kuw
            * phi
            * (1 - (rw * scale_popu_rw))
            / (rw * scale_popu_rw)
        )
        cs = (
            kws
            * scale_popu_kws
            * phi
            * rw
            * scale_popu_rw
            * (1 - (rs * scale_popu_rs))
            / (rs * scale_popu_rs)
        )

        kwu = -kws * scale_popu_kws + (kuw * scale_popu_kuw) * (
            -1 + 1.0 / (rw * scale_popu_rw)
        )

        lambda_min12 = ufl.conditional(ufl.lt(self.lmbda(F), 1.2), self.lmbda(F), 1.2)

        # dZetas = self.dLambda * As - self.Zetas * cs
        # dZetaw = self.dLambda * Aw - self.Zetaw * cw

        # Lets use Backward Euler scheme with a few fixed point iterations
        Zetas = self.Zetas
        Zetaw = self.Zetaw
        XS = self.XS
        XW = self.XW

        XS = ufl.conditional(ufl.lt(XS, 0), 0, XS)
        XW = ufl.conditional(ufl.lt(XW, 0), 0, XW)
        XU = 1 - TmB - XS - XW
        gammawu = gammaw * abs(Zetaw)

        zetas1 = Zetas * ufl.conditional(ufl.gt(Zetas, 0), 1, 0)
        zetas2 = (-1 - Zetas) * ufl.conditional(ufl.lt(Zetas, -1), 1, 0)
        gammasu = gammas * Max(zetas1, zetas2)

        dXS_dt = kws * scale_popu_kws * XW - XS * gammasu - XS * ksu
        dXW_dt = (
            kuw * scale_popu_kuw * XU
            - kws * scale_popu_kws * XW
            - XW * gammawu
            - XW * kwu
        )
        cat50 = cat50_ref * scale_popu_CaT50ref * HF_scaling_cat50_ref + Beta1 * (
            -1 + lambda_min12
        )
        CaTrpn = ufl.conditional(ufl.lt(CaTrpn, 0), 0, CaTrpn)
        dCaTrpn_dt = (
            ktrpn
            * scale_popu_kTRPN
            * (
                -CaTrpn
                + ufl.elem_pow(1000 * cai / cat50, ntrpn * scale_popu_nTRPN)
                * (1 - CaTrpn)
            )
        )
        kb = (
            ku
            * scale_popu_ku
            * ufl.elem_pow(Trpn50 * scale_popu_TRPN50, (ntm * scale_popu_nTm))
            / (
                1
                - (rs * scale_popu_rs)
                - rw * scale_popu_rw * (1 - (rs * scale_popu_rs))
            )
        )
        dTmB_dt = (
            ufl.conditional(
                ufl.lt(ufl.elem_pow(CaTrpn, -(ntm * scale_popu_nTm) / 2), 100),
                ufl.elem_pow(CaTrpn, -(ntm * scale_popu_nTm) / 2),
                100,
            )
            * XU
            * kb
            - ku
            * scale_popu_ku
            * ufl.elem_pow(CaTrpn, (ntm * scale_popu_nTm) / 2)
            * TmB
        )

        C = -1 + lambda_min12
        dCd = -Cd + C
        eta = ufl.conditional(ufl.lt(dCd, 0), etas, etal)
        dCd_dt = p_k * (-Cd + C) / eta
        Bcai = 1.0 / (1.0 + cmdnmax * kmcmdn * ufl.elem_pow(kmcmdn + cai, -2.0))
        J_TRPN = trpnmax * dCaTrpn_dt
        dcai = (
            -J_TRPN
            + Jdiff * vss / vmyo
            - Jup * vnsr / vmyo
            + 0.5 * (-ICab - IpCa - Isac_P_ns / 3 + 2.0 * INaCa_i) * Acap / (F * vmyo)
        ) * Bcai

        for _ in range(10):
            Zetas = self.Zetas_prev + self.dt * (self.dLambda(F) * As - Zetas * cs)
            Zetaw = self.Zetaw_prev + self.dt * (self.dLambda(F) * Aw - Zetaw * cw)

        return Zetas, Zetaw

    def Ta(self, F):

        Zetas, Zetaw = self._solve_ode(F)
        Tref = self._parameters["Tref"]
        rs = self._parameters["rs"]
        scale_popu_Tref = self._parameters["scale_popu_Tref"]
        scale_popu_rs = self._parameters["scale_popu_rs"]
        Beta0 = self._parameters["Beta0"]
        lmbda = self.lmbda(F)

        _min = ufl.min_value
        _max = ufl.max_value
        if isinstance(lmbda, (int, float)):
            _min = min
            _max = max
        lmbda = _min(1.2, lmbda)
        h_lambda_prima = 1 + Beta0 * (lmbda + _min(lmbda, 0.87) - 1.87)
        h_lambda = _max(0, h_lambda_prima)

        Ta = (
            h_lambda
            * (Tref * scale_popu_Tref / (rs * scale_popu_rs))
            * (self.XS * (Zetas + 1) + self.XW * Zetaw)
        )

        # Assign the current value of Ta so that we can retrive them for postprocessing
        self.Ta_current.assign(dolfin.project(Ta, self.function_space))
        # Assign these in order to update the EM coupling
        self.lmbda_current.assign(dolfin.project(lmbda, self.function_space))
        self.Zetas.assign(dolfin.project(Zetas, self.function_space))
        self.Zetaw.assign(dolfin.project(Zetaw, self.function_space))

        return Ta

    def Wactive(self, F, diff=0):
        """Active stress energy"""
        C = F.T * F

        if diff == 0:
            return pulse.material.active_model.Wactive_transversally(
                Ta=self.Ta(F),
                C=C,
                f0=self.f0,
                eta=self.eta,
            )
        return self.Ta(F)


class MechanicsProblem(pulse.MechanicsProblem):
    boundary_condition = BoundaryConditions.dirichlet

    def solve(self):
        self._init_forms()
        return super().solve()


class RigidMotionProblem(MechanicsProblem):
    boundary_condition = BoundaryConditions.rigid

    def _init_spaces(self):

        mesh = self.geometry.mesh

        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P3 = dolfin.VectorElement("Real", mesh.ufl_cell(), 0, 6)

        self.state_space = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P1, P2, P3]))

        self.state = dolfin.Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

    def _handle_bcs(self, bcs, bcs_parameters):
        self.bcs = pulse.BoundaryConditions()

    def _init_forms(self):
        # Displacement and hydrostatic_pressure; 3rd space for rigid motion component
        p, u, r = dolfin.split(self.state)
        q, v, w = dolfin.split(self.state_test)

        # Some mechanical quantities
        F = dolfin.variable(pulse.DeformationGradient(u))
        J = pulse.Jacobian(F)
        dx = self.geometry.dx

        internal_energy = self.material.strain_energy(
            F,
        ) + self.material.compressibility(p, J)

        self._virtual_work = dolfin.derivative(
            internal_energy * dx,
            self.state,
            self.state_test,
        )

        self._virtual_work += dolfin.derivative(
            RigidMotionProblem.rigid_motion_term(
                mesh=self.geometry.mesh,
                u=u,
                r=r,
            ),
            self.state,
            self.state_test,
        )

        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.state,
            dolfin.TrialFunction(self.state_space),
        )
        self._init_solver()

    def _init_solver(self):
        self._problem = pulse.NonlinearProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=[],
        )
        self.solver = pulse.NonlinearSolver(
            self._problem,
            self.state,
            parameters=self.solver_parameters,
        )

    def rigid_motion_term(mesh, u, r):
        position = dolfin.SpatialCoordinate(mesh)
        RM = [
            dolfin.Constant((1, 0, 0)),
            dolfin.Constant((0, 1, 0)),
            dolfin.Constant((0, 0, 1)),
            dolfin.cross(position, dolfin.Constant((1, 0, 0))),
            dolfin.cross(position, dolfin.Constant((0, 1, 0))),
            dolfin.cross(position, dolfin.Constant((0, 0, 1))),
        ]

        return sum(dolfin.dot(u, zi) * r[i] * dolfin.dx for i, zi in enumerate(RM))


def setup_microstructure(mesh):
    logger.debug("Set up microstructure")
    V_f = dolfin.VectorFunctionSpace(mesh, "DG", 1)
    f0 = dolfin.interpolate(
        dolfin.Expression(("1.0", "0.0", "0.0"), degree=1, cell=mesh.ufl_cell()),
        V_f,
    )
    s0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "1.0", "0.0"), degree=1, cell=mesh.ufl_cell()),
        V_f,
    )
    n0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "0.0", "1.0"), degree=1, cell=mesh.ufl_cell()),
        V_f,
    )
    # Collect the microstructure
    return pulse.Microstructure(f0=f0, s0=s0, n0=n0)


def float_to_constant(x: typing.Union[dolfin.Constant, float]) -> dolfin.Constant:
    """Convert float to a dolfin constant.
    If value is allready a constant, do nothing.

    Parameters
    ----------
    x : typing.Union[dolfin.Constant, float]
        The value to be converted

    Returns
    -------
    dolfin.Constant
        The same value, wrapped in a constant
    """
    if isinstance(x, float):
        return dolfin.Constant(x)
    return x


def setup_diriclet_bc(
    mesh: dolfin.Mesh,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = True,
) -> typing.Tuple[pulse.BoundaryConditions, pulse.MarkerFunctions]:
    """Completely fix the left side of the mesh (i.e the side with the
    lowest x-values) and apply some boundary condition to the right side.


    Parameters
    ----------
    mesh : dolfin.Mesh
        A cube or box-shaped mesh
    pre_stretch : typing.Union[dolfin.Constant, float], optional
        Value representing the amount of pre stretch, by default None
    traction : typing.Union[dolfin.Constant, float], optional
        Value representing the amount of traction, by default None
    spring : typing.Union[dolfin.Constant, float], optional
        Value representing the stiffness of the string, by default None
    fix_right_plane : bool, optional
        Fix the right plane so that it is not able to move in any direction
        except the x-direction, by default True

    Returns
    -------
    typing.Tuple[pulse.BoundaryConditions, pulse.MarkerFunctions]
        The boundary conditions and markers for the mesh

    Notes
    -----
    If `pre_stretch` if different from None, a pre stretch will be applied
    to the right side, through a Dirichlet boundary condition.

    If `traction` is different from None then an external force is applied
    to the right size, through a Neumann boundary condition.
    A positive value means that the force is compressing while a
    negative value means that it is stretching

    If `spring` is different from None then the amount of force that needs
    to be applied to displace the right boundary increases with the amount
    of displacement. The value of `spring` represents the stiffness of the
    spring.

    """
    logger.debug("Setup diriclet bc")
    # Get the value of the greatest x-coordinate
    Lx = mesh.mpi_comm().allreduce(mesh.coordinates().max(0)[0], op=MPI.MAX)

    # Define domain to apply dirichlet boundary conditions
    left = dolfin.CompiledSubDomain("near(x[0], 0) && on_boundary")
    right = dolfin.CompiledSubDomain("near(x[0], Lx) && on_boundary", Lx=Lx)

    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)

    left_marker = 1
    left.mark(boundary_markers, left_marker)
    right_marker = 2
    right.mark(boundary_markers, right_marker)

    marker_functions = pulse.MarkerFunctions(ffun=boundary_markers)

    def dirichlet_bc(W):
        # W here refers to the state space

        # BC with fixing left size
        bcs = [
            dolfin.DirichletBC(
                W.sub(0),
                dolfin.Constant((0.0, 0.0, 0.0)),
                left,
            ),
        ]

        if fix_right_plane:
            bcs.extend(
                [
                    dolfin.DirichletBC(
                        W.sub(0).sub(1),  # Second component of u, i.e u_y
                        dolfin.Constant(0.0),  # should be kept fixed
                        right,  # in this region
                    ),
                    dolfin.DirichletBC(
                        W.sub(0).sub(2),  # Third component of u, i.e u_z
                        dolfin.Constant(0.0),  # should be kept fixed
                        right,  # in this region
                    ),
                ],
            )

        if pre_stretch is not None:
            bcs.append(
                dolfin.DirichletBC(
                    W.sub(0).sub(0),
                    float_to_constant(pre_stretch),
                    right,
                ),
            )
        return bcs

    neumann_bc = []
    if traction is not None:
        neumann_bc.append(
            pulse.NeumannBC(
                traction=float_to_constant(traction),
                marker=right_marker,
            ),
        )

    robin_bc = []
    if spring is not None:
        robin_bc.append(
            pulse.RobinBC(value=float_to_constant(spring), marker=right_marker),
        )
    bcs = pulse.BoundaryConditions(
        dirichlet=(dirichlet_bc,),
        neumann=neumann_bc,
        robin=robin_bc,
    )

    return bcs, marker_functions
