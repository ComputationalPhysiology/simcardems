import dolfin
import pulse
from mpi4py import MPI

from . import utils
from .newton_solver import MechanicsNewtonSolver
from .newton_solver import MechanicsNewtonSolver_ODE


logger = utils.getLogger(__name__)


class ContinuationBasedMechanicsProblem(pulse.MechanicsProblem):
    def __init__(self, *args, **kwargs):
        self._use_custom_newton_solver = kwargs.pop("use_custom_newton_solver", False)
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

        if self._use_custom_newton_solver:
            cls = MechanicsNewtonSolver_ODE
        else:
            cls = MechanicsNewtonSolver

        self.solver = cls(
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


class RigidMotionProblem(MechanicsProblem):
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
        self._dirichlet_bc = []

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
