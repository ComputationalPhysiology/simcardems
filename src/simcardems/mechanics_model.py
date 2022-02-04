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


def mechanics_ode_rhs(s, CaTrpn, dLambda, parameters):
    phi = parameters["phi"]
    Tot_A = parameters["Tot_A"]
    Trpn50 = parameters["Trpn50"]
    gammas = parameters["gammas"]
    gammaw = parameters["gammaw"]
    ku = parameters["ku"]
    kuw = parameters["kuw"]
    kws = parameters["kws"]
    ntm = parameters["ntm"]

    rs = parameters["rs"]
    rw = parameters["rw"]

    # Population factors
    scale_popu_nTm = parameters["scale_popu_nTm"]
    scale_popu_kuw = parameters["scale_popu_kuw"]
    scale_popu_kws = parameters["scale_popu_kws"]
    scale_popu_ku = parameters["scale_popu_ku"]
    scale_popu_TRPN50 = parameters["scale_popu_TRPN50"]
    scale_popu_rw = parameters["scale_popu_rw"]
    scale_popu_rs = parameters["scale_popu_rs"]

    XS, XW, TmB, Zetas, Zetaw = s

    Aw = (
        Tot_A
        * rs
        * scale_popu_rs
        / (rs * scale_popu_rs + rw * scale_popu_rw * (1 - (rs * scale_popu_rs)))
    )
    As = Aw

    cw = kuw * scale_popu_kuw * phi * (1 - (rw * scale_popu_rw)) / (rw * scale_popu_rw)
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
    ksu = kws * scale_popu_kws * rw * scale_popu_rw * (-1 + 1.0 / (rs * scale_popu_rs))

    XS = ufl.conditional(ufl.lt(XS, 0), 0, XS)
    XW = ufl.conditional(ufl.lt(XW, 0), 0, XW)
    XU = 1 - TmB - XS - XW
    gammawu = gammaw * abs(Zetaw)

    zetas1 = Zetas * ufl.conditional(ufl.gt(Zetas, 0), 1, 0)
    zetas2 = (-1 - Zetas) * ufl.conditional(ufl.lt(Zetas, -1), 1, 0)
    gammasu = gammas * Max(zetas1, zetas2)

    dXS_dt = kws * scale_popu_kws * XW - XS * gammasu - XS * ksu
    dXW_dt = (
        kuw * scale_popu_kuw * XU - kws * scale_popu_kws * XW - XW * gammawu - XW * kwu
    )

    kb = (
        ku
        * scale_popu_ku
        * ufl.elem_pow(Trpn50 * scale_popu_TRPN50, (ntm * scale_popu_nTm))
        / (1 - (rs * scale_popu_rs) - rw * scale_popu_rw * (1 - (rs * scale_popu_rs)))
    )
    dTmB_dt = (
        ufl.conditional(
            ufl.lt(ufl.elem_pow(CaTrpn, -(ntm * scale_popu_nTm) / 2), 100),
            ufl.elem_pow(CaTrpn, -(ntm * scale_popu_nTm) / 2),
            100,
        )
        * XU
        * kb
        - ku * scale_popu_ku * ufl.elem_pow(CaTrpn, (ntm * scale_popu_nTm) / 2) * TmB
    )

    dZetas_dt = dLambda * As - Zetas * cs
    dZetaw_dt = dLambda * Aw - Zetaw * cw

    return dolfin.as_vector([dXS_dt, dXW_dt, dTmB_dt, dZetas_dt, dZetaw_dt])


class BoundaryConditions(str, Enum):
    dirichlet = "dirichlet"
    rigid = "rigid"


class LandModel(pulse.ActiveModel):
    def __init__(
        self,
        CaTrpn,
        lmbda,
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
        self._mesh = function_space.mesh()

        self.state_space = dolfin.VectorFunctionSpace(self._mesh, "CG", 1, dim=5)
        self.state = dolfin.Function(self.state_space)
        self.state_ = dolfin.Function(self.state_space)
        self.state_test = dolfin.TestFunction(self.state_space)

        self.CaTrpn = CaTrpn
        self.CaTrpn_prev = dolfin.Function(function_space)
        self._parameters = parameters
        self._t = 0
        self._t_prev = 0

        self.Ta_current = dolfin.Function(function_space, name="Ta")
        self.lmbda_prev = dolfin.Function(function_space)
        self.lmbda_current = dolfin.Function(function_space)
        self.lmbda_current = lmbda
        self.update_prev()

    def update_time(self, t):
        self._t_prev = self._t
        self._t = t

    @property
    def dt(self):
        return self._t - self._t_prev

    def update_prev(self):
        self.state_.vector()[:] = self.state.vector()
        self.CaTrpn_prev.vector()[:] = self.CaTrpn.vector()
        self.lmbda_prev.vector()[:] = self.lmbda_current.vector()

    def lmbda(self, F):
        f = F * self.f0
        return dolfin.sqrt(f**2)

    def dLambda(self, F):
        dLambda = (self.lmbda(F) - self.lmbda_prev) / self.dt
        return dLambda

    def _solve_ode(self, F, s):

        if abs(self.dt) < 1e-10:
            return dolfin.Constant([0.0] * 5)

        s_ = dolfin.as_vector(dolfin.split(self.state_))
        Dt_s = (s - s_) / self.dt
        dLambda = self.dLambda(F)

        theta = 0.5
        s_mid = theta * s + (1 - theta) * s_
        CaTrpn_mid = theta * self.CaTrpn + (1 - theta) * self.CaTrpn_prev
        F_theta = mechanics_ode_rhs(
            s_mid,
            CaTrpn=CaTrpn_mid,
            dLambda=dLambda,
            parameters=self._parameters,
        )

        return Dt_s - F_theta

    def Ta(self, F, s):
        s_split = dolfin.as_vector(dolfin.split(s))
        G = self._solve_ode(F, s_split)
        (XS, XW, TmB, Zetas, Zetaw) = s_split

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
            * (XS * (Zetas + 1) + XW * Zetaw)
        )

        # Assign the current value of Ta so that we can retrive them for postprocessing
        self.Ta_current.assign(dolfin.project(Ta, self.function_space))
        # Assign these in order to update the EM coupling
        self.lmbda_current.assign(dolfin.project(lmbda, self.function_space))

        return Ta, G

    def Wactive(self, F, s):
        """Active stress energy"""
        C = F.T * F

        Ta, G = self.Ta(F, s)
        return (
            pulse.material.active_model.Wactive_transversally(
                Ta=Ta,
                C=C,
                f0=self.f0,
                eta=self.eta,
            ),
            G,
        )


class MechanicsProblem(pulse.MechanicsProblem):
    boundary_condition = BoundaryConditions.dirichlet

    def __init__(self, *args, **kwargs):
        self.active_model = kwargs.pop("active_model")
        super().__init__(*args, **kwargs)

    def _init_spaces(self):

        mesh = self.geometry.mesh

        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P_ode = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 1, dim=5)

        self.state_space = dolfin.FunctionSpace(
            mesh,
            dolfin.MixedElement([P2, P1, P_ode]),
        )

        self.state = dolfin.Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

        self.s, self.s_assigner = utils.setup_assigner(self.state, 2)

    def _init_forms(self):
        u, p, s = dolfin.split(self.state)
        v, q, w = dolfin.split(self.state_test)
        self.s_assigner.assign(self.s, utils.sub_function(self.state, 2))

        # Some mechanical quantities
        F = dolfin.variable(pulse.DeformationGradient(u))
        J = pulse.Jacobian(F)
        dx = self.geometry.dx

        Wactive, G_ode = self.active_model.Wactive(F, self.s)

        internal_energy = (
            self.material.strain_energy(
                F,
            )
            + self.material.compressibility(p, J)
            + Wactive
        )

        self._virtual_work = dolfin.derivative(
            internal_energy * dx,
            self.state,
            self.state_test,
        )

        self._virtual_work += dolfin.derivative(
            dolfin.inner(G_ode, w) * dx,
            self.state,
            self.state_test,
        )

        self._set_dirichlet_bc()
        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.state,
            dolfin.TrialFunction(self.state_space),
        )
        self._init_solver()

    def solve(self):
        self._init_forms()
        return super().solve()


class RigidMotionProblem(MechanicsProblem):
    boundary_condition = BoundaryConditions.rigid

    def __init__(self, *args, **kwargs):
        self.active_model = kwargs.pop("active_model")
        super().__init__(*args, **kwargs)

    def _init_spaces(self):

        mesh = self.geometry.mesh

        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P3 = dolfin.VectorElement("Real", mesh.ufl_cell(), 0, 6)

        self.state_space = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P1, P2, P3]))

        self.state = dolfin.Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

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

    def _handle_bcs(self, bcs, bcs_parameters):
        self.bcs = pulse.BoundaryConditions()

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
