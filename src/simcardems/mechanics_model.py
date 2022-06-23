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


def _Zeta(Zeta_prev, A, c, dLambda, dt, scheme: Scheme):

    if scheme == Scheme.analytic:
        return Zeta_prev * dolfin.exp(-c * dt) + (A * dLambda / c * dt) * (
            1 - dolfin.exp(-c * dt)
        )

    elif scheme == Scheme.bd:
        return Zeta_prev + A * dLambda / (1 + c * dt)
    else:
        return Zeta_prev * (1 - c * dt) + A * dLambda


class LandModel(pulse.ActiveModel):
    def __init__(
        self,
        XS,
        XW,
        parameters,
        mesh,
        Zetas=None,
        Zetaw=None,
        lmbda=None,
        f0=None,
        s0=None,
        n0=None,
        eta=0,
        scheme: Scheme = Scheme.analytic,
        **kwargs,
    ):
        super().__init__(f0=f0, s0=s0, n0=n0)
        self._eta = eta
        self.function_space = pulse.QuadratureSpace(mesh, degree=3, dim=1)

        self.XS = XS
        self.XW = XW
        self._parameters = parameters
        self._t = 0.0
        self._t_prev = 0.0
        self._scheme = scheme

        self._dLambda = dolfin.Function(self.function_space)
        self.lmbda_prev = dolfin.Function(self.function_space)
        if lmbda is not None:
            self.lmbda_prev = lmbda
        self.lmbda = dolfin.Function(self.function_space)

        self._Zetas = dolfin.Function(self.function_space)
        self.Zetas_prev = dolfin.Function(self.function_space)
        if Zetas is not None:
            self.Zetas_prev.assign(Zetas)

        self._Zetaw = dolfin.Function(self.function_space)
        self.Zetaw_prev = dolfin.Function(self.function_space)
        if Zetaw is not None:
            self.Zetaw_prev.assign(Zetaw)

        self.V_cg1 = dolfin.FunctionSpace(mesh, "CG", 1)
        self.Ta_current = dolfin.Function(self.function_space, name="Ta")
        self.Ta_current_cg1 = dolfin.Function(self.V_cg1, name="Ta")

    @property
    def dLambda(self):
        self._dLambda.vector()[:] = self.lmbda.vector() - self.lmbda_prev.vector()
        return self._dLambda

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

    def update_Zetas(self):
        self._Zetas.vector()[:] = _Zeta(
            self.Zetas_prev.vector(),
            self.As,
            self.cs,
            self.dLambda.vector(),
            self.dt,
            self._scheme,
        )

    @property
    def Zetas(self):
        return self._Zetas

    def update_Zetaw(self):
        self._Zetaw.vector()[:] = _Zeta(
            self.Zetaw_prev.vector(),
            self.Aw,
            self.cw,
            self.dLambda.vector(),
            self.dt,
            self._scheme,
        )

    @property
    def Zetaw(self):
        return self._Zetaw

    @property
    def dt(self) -> float:
        from .setup_models import TimeStepper

        return TimeStepper.ns2ms(self.t - self._t_prev)

    @property
    def t(self) -> float:
        return self._t

    def start_time(self, t):
        self._t_prev = t
        self._t = t

    def update_time(self, t):
        self._t_prev = self.t
        self._t = t

    def update_prev(self):
        self.Zetas_prev.vector()[:] = self.Zetas.vector()
        self.Zetaw_prev.vector()[:] = self.Zetaw.vector()
        self.lmbda_prev.vector()[:] = self.lmbda.vector()
        self.Ta_current.assign(
            dolfin.project(
                self.Ta,
                self.function_space,
                form_compiler_parameters={"representation": "quadrature"},
            ),
        )
        utils.local_project(self.Ta_current, self.V_cg1, self.Ta_current_cg1)

    @property
    def Ta(self):
        Tref = self._parameters["Tref"]
        rs = self._parameters["rs"]
        scale_popu_Tref = self._parameters["scale_popu_Tref"]
        scale_popu_rs = self._parameters["scale_popu_rs"]
        Beta0 = self._parameters["Beta0"]

        _min = ufl.min_value
        _max = ufl.max_value
        if isinstance(self.lmbda, (int, float)):
            _min = min
            _max = max
        lmbda = _min(1.2, self.lmbda)
        h_lambda_prima = 1 + Beta0 * (lmbda + _min(lmbda, 0.87) - 1.87)
        h_lambda = _max(0, h_lambda_prima)

        return (
            h_lambda
            * (Tref * scale_popu_Tref / (rs * scale_popu_rs))
            * (self.XS * (self.Zetas + 1) + self.XW * self.Zetaw)
        )

    def Wactive(self, F, **kwargs):
        """Active stress energy"""

        C = F.T * F
        f = F * self.f0
        self.lmbda.assign(
            dolfin.project(
                dolfin.sqrt(f**2),
                self.function_space,
                form_compiler_parameters={"representation": "quadrature"},
            ),
        )
        self.update_Zetas()
        self.update_Zetaw()

        return pulse.material.active_model.Wactive_transversally(
            Ta=self.Ta,
            C=C,
            f0=self.f0,
            eta=self.eta,
        )


class ContinuationBasedMechanicsProblem(pulse.MechanicsProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_states = []
        self.old_controls = []

    def solve(self):
        self._init_forms()
        return super().solve()

    def solve_for_control(self, control, tol=1e-5):
        """Solve with a continuation step for
        better initial guess for the Newton solver
        """
        if len(self.old_controls) >= 2:
            # Find a better initial guess for the solver
            c0, c1 = self.old_controls
            s0, s1 = self.old_states

            denominator = dolfin.assemble((c1 - c0) ** 2 * dolfin.dx)
            max_denom = self.geometry.mesh.mpi_comm().allreduce(denominator, op=MPI.MAX)

            if max_denom > tol:
                numerator = dolfin.assemble((control - c0) ** 2 * dolfin.dx)
                delta = numerator / denominator
                self.state.vector().zero()
                self.state.vector().axpy(1.0 - delta, s0.vector())
                self.state.vector().axpy(delta, s1.vector())

                # Keep track of the newest state
                self.old_controls = [c1]
                self.old_states = [s1]
            else:
                # Keep track of the old state
                self.old_controls = [c0]
                self.old_states = [s0]

        self.solve()

        self.old_states.append(self.state.copy(deepcopy=True))
        self.old_controls.append(control.copy(deepcopy=True))


class MechanicsProblem(ContinuationBasedMechanicsProblem):
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

    def _init_forms(self):
        u, p = dolfin.split(self.state)

        # Some mechanical quantities
        self._F = dolfin.variable(pulse.DeformationGradient(u))
        self._J = pulse.Jacobian(self._F)
        dx = self.geometry.dx

        internal_energy = self.material.strain_energy(
            self._F,
        ) + self.material.compressibility(p, self._J)

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

    def _init_solver(self):
        self._problem = pulse.NonlinearProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=self._dirichlet_bc,
        )
        self.solver = MechanicsNewtonSolver_ODE(
            problem=self._problem,
            state=self.state,
            update_cb=self.material.active.update_prev,
            parameters=self.solver_parameters,
        )

    def solve(self):
        self._init_forms()
        newton_iteration, newton_converged = self.solver.solve()
        # DEBUGGING
        getattr(self.solver, "check_overloads_called", None)
        return newton_iteration, newton_converged

    def update_lmbda_prev(self):
        self.lmbda_prev.vector()[:] = self.lmbda.vector()


class MechanicsNewtonSolver_ODE(dolfin.NewtonSolver):
    def __init__(
        self,
        problem: pulse.NonlinearProblem,
        state: dolfin.Function,
        update_cb,
        parameters=None,
    ):
        dolfin.PETScOptions.clear()

        self._problem = problem
        self._state = state
        self._update_cb = update_cb
        self._parameters = parameters

        # Initializing Newton solver (parent class)
        self.petsc_solver = dolfin.PETScKrylovSolver()
        super().__init__(
            self._state.function_space().mesh().mpi_comm(),
            self.petsc_solver,
            dolfin.PETScFactory.instance(),
        )

        # Setting default parameters
        params = MechanicsNewtonSolver_ODE.default_solver_parameters()
        for k, v in params.items():
            if self.parameters.has_parameter(k):
                self.parameters[k] = v
            if self.parameters.has_parameter_set(k):
                for subk, subv in params[k].items():
                    self.parameters[k][subk] = subv
        petsc = params.pop("petsc")
        for k, v in petsc.items():
            if v is not None:
                dolfin.PETScOptions.set(k, v)
        self.newton_verbose = params.pop("newton_verbose", False)
        self.ksp_verbose = params.pop("ksp_verbose", False)
        if self.newton_verbose:
            dolfin.set_log_level(dolfin.LogLevel.INFO)
            self.parameters["report"] = True
        if self.ksp_verbose:
            self.parameters["lu_solver"]["report"] = True
            self.parameters["lu_solver"]["verbose"] = True
            self.parameters["krylov_solver"]["monitor_convergence"] = True
            dolfin.PETScOptions.set("ksp_monitor_true_residual")
        self.linear_solver().set_from_options()

    @staticmethod
    def default_solver_parameters():
        return {
            "petsc": {
                # "ksp_type": "preonly",
                "ksp_type": "gmres",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_33": 0,
            },
            "newton_verbose": False,
            "ksp_verbose": False,
            # "linear_solver": "mumps",
            "linear_solver": "gmres",
            "error_on_nonconvergence": True,
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-5,
            "maximum_iterations": 20,
            "report": False,
            # },
            "krylov_solver": {
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 1000,
                "monitor_convergence": False,
            },
            "lu_solver": {"report": False, "symmetric": False, "verbose": False},
        }

    def converged(self, r, p, i):
        self._converged_called = True

        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            # Print all residuals
            residual = r.norm("l2")
            with open("residual.txt", "a") as rfile:
                rfile.write(str(residual) + "\t")

        return super().converged(r, p, i)

    def solver_setup(self, A, J, p, i):
        self._solver_setup_called = True
        super().solver_setup(A, J, p, i)

    def update_solution(self, x, dx, rp, p, i):
        self._update_solution_called = True

        # Update x from the dx obtained from linear solver (Newton iteration) :
        # x = -rp*dx (rp : relax param)
        super().update_solution(x, dx, rp, p, i)

        # Updating form of MechanicsProblem (from current lmbda, zetas, zetaw, ...)
        self._state.vector().set_local(x)
        # self._mech_problem._init_forms()
        # Recompute Zetas, Zetaw, Ta, lmbda
        # self._mech_problem.material.active.update_prev()
        self._update_cb()
        # Re-init this solver with the new problem (note : done in _init_forms)
        # self.__init__(self._mech_problem)

    def solve(self):
        self._solve_called = True
        ret = super().solve(self._problem, self._state.vector())
        self._state.vector().apply("insert")
        return ret

    # DEBUGGING
    # This is just to check if we are using the overloaded functions
    def check_overloads_called(self):
        assert getattr(self, "_converged_called", False)
        assert getattr(self, "_solver_setup_called", False)
        assert getattr(self, "_update_solution_called", False)
        assert getattr(self, "_solve_called", False)


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
        self._F = dolfin.variable(pulse.DeformationGradient(u))
        self._J = pulse.Jacobian(self._F)
        dx = self.geometry.dx

        internal_energy = self.material.strain_energy(
            self._F,
        ) + self.material.compressibility(p, self._J)

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
        self.solver = MechanicsNewtonSolver_ODE(
            problem=self._problem,
            state=self.state,
            update_cb=self.material.active.update_prev,
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
    # V_f = dolfin.VectorFunctionSpace(mesh, "DG", 1)
    V_f = pulse.QuadratureSpace(mesh, degree=3, dim=3)
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
    If value is already a constant, do nothing.

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
    fix_right_plane: bool = False,
) -> typing.Tuple[pulse.BoundaryConditions, pulse.MarkerFunctions]:
    """Completely fix the left side of the mesh in the x-direction (i.e the side with the
    lowest x-values), fix points at x=0 & y=0 in y-direction, fix points at x=0 & z=0 in
    z-direction and apply some boundary condition to the right side.


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
    leftback = dolfin.CompiledSubDomain("near(x[0], 0) && near(x[2], 0)")
    leftbottom = dolfin.CompiledSubDomain("near(x[0], 0) && near(x[1], 0)")
    rightback = dolfin.CompiledSubDomain(
        "near(x[0], Lx) && near(x[2], 0) && on_boundary",
        Lx=Lx,
    )
    rightbottom = dolfin.CompiledSubDomain(
        "near(x[0], Lx) && near(x[1], 0) && on_boundary",
        Lx=Lx,
    )

    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)

    left_marker = 1
    left.mark(boundary_markers, left_marker)
    right_marker = 2
    right.mark(boundary_markers, right_marker)
    leftback_marker = 3
    leftback.mark(boundary_markers, leftback_marker)
    leftbottom_marker = 4
    leftbottom.mark(boundary_markers, leftbottom_marker)

    marker_functions = pulse.MarkerFunctions(ffun=boundary_markers)

    def dirichlet_bc(W):
        # W here refers to the state space

        # BC with fixing left size
        bcs = [
            dolfin.DirichletBC(
                W.sub(0).sub(0),  # u_x
                dolfin.Constant(0.0),
                left,
            ),
            dolfin.DirichletBC(
                W.sub(0).sub(1),  # u_y
                dolfin.Constant(0.0),
                leftbottom,
                method="pointwise",
            ),
            dolfin.DirichletBC(
                W.sub(0).sub(2),  # u_z
                dolfin.Constant(0.0),
                leftback,
                method="pointwise",
            ),
        ]

        if fix_right_plane:
            bcs.extend(
                [
                    dolfin.DirichletBC(
                        W.sub(0).sub(0),  # u_x
                        dolfin.Constant(0.0),
                        right,
                    ),
                    dolfin.DirichletBC(
                        W.sub(0).sub(1),  # u_y
                        dolfin.Constant(0.0),
                        rightbottom,
                        method="pointwise",
                    ),
                    dolfin.DirichletBC(
                        W.sub(0).sub(2),  # u_z
                        dolfin.Constant(0.0),
                        rightback,
                        method="pointwise",
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
