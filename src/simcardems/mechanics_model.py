import typing
from enum import Enum

import dolfin
import pulse
import ufl
from mpi4py import MPI

from . import utils

logger = utils.getLogger(__name__)


class BoundaryConditions(str, Enum):
    dirichlet = "dirichlet"
    rigid = "rigid"


class Scheme(str, Enum):
    fd = "fd"
    bd = "bd"
    analytic = "analytic"


class LandModel(pulse.ActiveModel):
    def __init__(
        self,
        XS,
        XW,
        parameters,
        mesh,
        Zetas=None,
        Zetaw=None,
        f0=None,
        s0=None,
        n0=None,
        eta=0,
        scheme: Scheme = Scheme.fd,
        **kwargs,
    ):
        super().__init__(f0=f0, s0=s0, n0=n0)
        self._eta = eta
        self.function_space = dolfin.FunctionSpace(mesh, "CG", 1)

        self.XS = XS
        self.XW = XW
        self._parameters = parameters
        self.t = dolfin.Constant(0.0)
        self._t_prev = dolfin.Constant(0.0)
        self._scheme = scheme

        self._Ta = dolfin.Constant(0.0)

        self.dLambda = dolfin.Constant(0.0)

        self.Zetas_prev = dolfin.Function(self.function_space)
        if Zetas is not None:
            self.Zetas_prev.assign(Zetas)

        self.Zetaw_prev = dolfin.Function(self.function_space)
        if Zetaw is not None:
            self.Zetaw_prev.assign(Zetaw)

        self.Ta_current = dolfin.Function(self.function_space, name="Ta")

    @property
    def Aw(self):
        Tot_A = self._parameters["Tot_A"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        scale_popu_rw = self._parameters["scale_popu_rw"]
        scale_popu_rs = self._parameters["scale_popu_rs"]
        return (
            Tot_A
            * rs
            * scale_popu_rs
            / (rs * scale_popu_rs + rw * scale_popu_rw * (1 - (rs * scale_popu_rs)))
        )

    @property
    def As(self):
        return self.Aw

    @property
    def cw(self):
        phi = self._parameters["phi"]
        kuw = self._parameters["kuw"]
        rw = self._parameters["rw"]

        scale_popu_kuw = self._parameters["scale_popu_kuw"]
        scale_popu_rw = self._parameters["scale_popu_rw"]
        return (
            kuw
            * scale_popu_kuw
            * phi
            * (1 - (rw * scale_popu_rw))
            / (rw * scale_popu_rw)
        )

    @property
    def cs(self):
        phi = self._parameters["phi"]
        kws = self._parameters["kws"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        scale_popu_kws = self._parameters["scale_popu_kws"]
        scale_popu_rw = self._parameters["scale_popu_rw"]
        scale_popu_rs = self._parameters["scale_popu_rs"]
        return (
            kws
            * scale_popu_kws
            * phi
            * rw
            * scale_popu_rw
            * (1 - (rs * scale_popu_rs))
            / (rs * scale_popu_rs)
        )

    @property
    def Zetas(self):
        if self._scheme == Scheme.analytic:
            return self.Zetas_prev * dolfin.exp(-self.cs * self.dt) + (
                self.As * self.dLambda / self.cs * self.dt
            ) * (1 - dolfin.exp(-self.cs * self.dt))

        elif self._scheme == Scheme.bd:
            return self.Zetas_prev + self.As * self.dLambda / (1 + self.cs * self.dt)
        else:
            return self.Zetas_prev * (1 - self.cs * self.dt) + self.As * self.dLambda

    @property
    def Zetaw(self):
        if self._scheme == Scheme.analytic:
            return self.Zetaw_prev * dolfin.exp(-self.cw * self.dt) + (
                self.Aw * self.dLambda / self.cw * self.dt
            ) * (1 - dolfin.exp(-self.cw * self.dt))
        elif self._scheme == Scheme.bd:
            return self.Zetaw_prev + self.Aw * self.dLambda / (1 + self.cw * self.dt)
        else:
            return self.Zetaw_prev * (1 - self.cw * self.dt) + self.Aw * self.dLambda

    @property
    def dt(self):
        return self.t - self._t_prev

    @property
    def t(self) -> dolfin.Constant:
        return self._t

    @t.setter
    def t(self, t: typing.Union[float, dolfin.Constant]) -> None:
        if isinstance(t, (int, float)):
            t = dolfin.Constant(t)
        self._t = t

    def update_time(self, t):
        self._t_prev.assign(self.t)
        self._t.assign(dolfin.Constant(t))

    def update_prev(self):

        self.Zetas_prev.assign(dolfin.project(self.Zetas, self.function_space))
        self.Zetaw_prev.assign(dolfin.project(self.Zetaw, self.function_space))
        self.Ta_current.assign(dolfin.project(self._Ta, self.function_space))

    def Ta(self, lmbda, dLambda):

        self.dLambda = dLambda
        Tref = self._parameters["Tref"]
        rs = self._parameters["rs"]
        scale_popu_Tref = self._parameters["scale_popu_Tref"]
        scale_popu_rs = self._parameters["scale_popu_rs"]
        Beta0 = self._parameters["Beta0"]

        _min = ufl.min_value
        _max = ufl.max_value
        if isinstance(lmbda, (int, float)):
            _min = min
            _max = max
        lmbda = _min(1.2, lmbda)
        h_lambda_prima = 1 + Beta0 * (lmbda + _min(lmbda, 0.87) - 1.87)
        h_lambda = _max(0, h_lambda_prima)

        self._Ta = (
            h_lambda
            * (Tref * scale_popu_Tref / (rs * scale_popu_rs))
            * (self.XS * (self.Zetas + 1) + self.XW * self.Zetaw)
        )

        return self._Ta

    def Wactive(self, F, lmbda=None, dLambda=None, **kwargs):
        """Active stress energy"""

        C = F.T * F
        Ta = self.Ta(lmbda=lmbda, dLambda=dLambda)

        return pulse.material.active_model.Wactive_transversally(
            Ta=Ta,
            C=C,
            f0=self.f0,
            eta=self.eta,
        )


class HolzapfelOgden(pulse.HolzapfelOgden):
    def strain_energy(self, F):
        # Invariants
        I1 = pulse.kinematics.I1(F, isochoric=self.isochoric)
        I4f = pulse.kinematics.I4(F, self.f0, isochoric=self.isochoric)
        I4s = pulse.kinematics.I4(F, self.s0, isochoric=self.isochoric)
        I8fs = pulse.kinematics.I8(F, self.f0, self.s0)

        if self.active_model == "active_strain":
            inv_gamma = 1 - self.activation_field
            I1e = inv_gamma * I1 + (1 / inv_gamma**2 - inv_gamma) * I4f
            I4fe = 1 / inv_gamma**2 * I4f
            I4se = inv_gamma * I4s
            I8fse = 1 / dolfin.sqrt(inv_gamma) * I8fs
        else:
            I1e = I1
            I4fe = I4f
            I4se = I4s
            I8fse = I8fs

        W1 = self.W_1(I1e)
        W4f = self.W_4(I4fe, "f")
        W4s = self.W_4(I4se, "s")
        W8fs = self.W_8(I8fse)

        W = W1 + W4f + W4s + W8fs

        return W


class MechanicsProblem(pulse.MechanicsProblem):
    boundary_condition = BoundaryConditions.dirichlet

    def _init_spaces(self):

        mesh = self.geometry.mesh

        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)

        self.state_space = dolfin.FunctionSpace(
            mesh,
            dolfin.MixedElement([P2, P1]),
        )
        self._init_functions()

    def _init_functions(self):
        self.state = dolfin.Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)
        self.lmbda_space = dolfin.FunctionSpace(self.geometry.mesh, "CG", 1)
        self.lmbda_prev = dolfin.Function(self.lmbda_space)
        self.lmbda = dolfin.Function(self.lmbda_space)

    def _init_forms(self):
        u, p = dolfin.split(self.state)

        # Some mechanical quantities
        F = dolfin.variable(pulse.DeformationGradient(u))
        J = pulse.Jacobian(F)
        dx = self.geometry.dx

        f = F * self.material.f0
        self.lmbda = dolfin.sqrt(f**2)
        dLambda = self.lmbda - self.lmbda_prev

        Wactive = self.material.active.Wactive(
            F=F,
            lmbda=self.lmbda,
            dLambda=dLambda,
        )

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

        self._set_dirichlet_bc()
        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.state,
            dolfin.TrialFunction(self.state_space),
        )
        self._init_solver()

    @property
    def F(self):
        return dolfin.variable(pulse.DeformationGradient(dolfin.split(self.state)[0]))

    def solve(self):
        # self._init_forms()
        return super().solve()

    def update_lmbda_prev(self):
        self.lmbda_prev.assign(dolfin.project(self.lmbda, self.lmbda_space))


class RigidMotionProblem(MechanicsProblem):
    boundary_condition = BoundaryConditions.rigid

    def _init_spaces(self):

        mesh = self.geometry.mesh

        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P3 = dolfin.VectorElement("Real", mesh.ufl_cell(), 0, 6)

        self.state_space = dolfin.FunctionSpace(
            mesh,
            dolfin.MixedElement([P1, P2, P3]),
        )

        self._init_functions()

    def _handle_bcs(self, bcs, bcs_parameters):
        self.bcs = pulse.BoundaryConditions()

    def _init_forms(self):
        # Displacement and hydrostatic_pressure; 3rd space for rigid motion component
        p, u, r = dolfin.split(self.state)
        q, v, z = dolfin.split(self.state_test)

        # Some mechanical quantities
        F = dolfin.variable(pulse.DeformationGradient(u))
        J = pulse.Jacobian(F)
        dx = self.geometry.dx

        Wactive = self.material.active.Wactive(
            F=F,
            lmbda=self.lmbda,
            dLambda=self.dLambda,
        )

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

    @property
    def F(self):
        return dolfin.variable(pulse.DeformationGradient(dolfin.split(self.state)[1]))

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
