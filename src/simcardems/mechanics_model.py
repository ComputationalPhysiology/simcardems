import typing
from enum import Enum

import dolfin
import pulse
from mpi4py import MPI

from . import utils
from .newton_solver import MechanicsNewtonSolver
from .newton_solver import MechanicsNewtonSolver_ODE


logger = utils.getLogger(__name__)


class BoundaryConditions(str, Enum):
    dirichlet = "dirichlet"
    rigid = "rigid"


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


class TorsoMechanicsProblem:
    def __init__(self, full_mesh, torso_mesh, heart_d):
        self.mesh = full_mesh
        self.torso_mesh = torso_mesh
        self.heart_mesh = heart_d.function_space().mesh()
        self.heart_d = heart_d
        self.V = dolfin.FunctionSpace(self.torso_mesh, self.heart_d.ufl_element())

    def update_heart_disp(self, d):
        self.heart_d = d

    def set_bcs(self):
        bc_markers = torso_boundary_marking(self.torso_mesh, self.heart_mesh)
        with dolfin.XDMFFile(
            dolfin.MPI.comm_world,
            "./XDMF/HT/torso_marking.xdmf",
        ) as xdmf:
            xdmf.write(bc_markers)

        ## FIXME : This is super expensive!
        class HeartDisp(dolfin.UserExpression):
            def __init__(self, mesh, heart_d, **kwargs):
                self.mesh_ = mesh
                self.heart_disp_ = heart_d
                super().__init__(**kwargs)

            def eval(self, values, x):
                # Check if x is in heart mesh (should be True only for heart boundary points)
                index = self.mesh_.bounding_box_tree().compute_first_entity_collision(
                    dolfin.Point(x),
                )
                x_in_mesh = index < self.mesh_.num_entities(self.mesh_.topology().dim())
                if x_in_mesh:
                    for i in range(self.mesh_.topology().dim()):
                        values[i] = self.heart_disp_(x)[i]

            def value_shape(self):
                return (self.mesh_.topology().dim(),)

        zero = dolfin.Constant((0.0, 0.0, 0.0))
        dh = HeartDisp(self.heart_mesh, self.heart_d, degree=2)

        # DEBUG
        # Interpolation is also super expensive!
        # dh_h = dolfin.interpolate(dh, self.V)
        # with dolfin.XDMFFile(dolfin.MPI.comm_world, "./XDMF/HT/heart_disp1.xdmf") as xdmf:
        #     xdmf.write(dh_h)
        # with dolfin.XDMFFile(dolfin.MPI.comm_world, "./XDMF/HT/heart_disp2.xdmf") as xdmf:
        #     xdmf.write(self.heart_d)

        bc_ext = dolfin.DirichletBC(self.V, zero, bc_markers, 2)
        bc_int = dolfin.DirichletBC(self.V, dh, bc_markers, 1)

        return [bc_int, bc_ext]

    def solve(self):

        u = dolfin.TrialFunction(self.V)
        phi = dolfin.TestFunction(self.V)
        d = dolfin.Function(self.V)

        # Variational formulation - Standard Poisson (for now)
        a = dolfin.inner(dolfin.grad(u), dolfin.grad(phi)) * dolfin.dx
        L = dolfin.inner(dolfin.Constant((0.0, 0.0, 0.0)), phi) * dolfin.dx

        bcs = self.set_bcs()
        dolfin.solve(a == L, d, bcs=bcs, solver_parameters={"linear_solver": "mumps"})

        self.torso_d = d

        # Create a displacement field defined on full mesh (H+T)
        d_all = dolfin.Function(
            dolfin.FunctionSpace(self.mesh, self.heart_d.ufl_element()),
        )

        utils.submesh_to_mesh(self.heart_d, d_all)  # Add heart contrib
        utils.submesh_to_mesh(self.torso_d, d_all)  # Add torso contrib

        return d_all


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
    lowest x-values), fix the plane with y=0 in the y-direction, fix the plane with
    z=0 in the z-direction and apply some boundary condition to the right side.


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
    plane_y0 = dolfin.CompiledSubDomain("near(x[1], 0) && on_boundary")
    plane_z0 = dolfin.CompiledSubDomain("near(x[2], 0) && on_boundary")

    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)

    left_marker = 1
    left.mark(boundary_markers, left_marker)
    right_marker = 2
    right.mark(boundary_markers, right_marker)
    plane_y0_marker = 3
    plane_y0.mark(boundary_markers, plane_y0_marker)
    plane_z0_marker = 4
    plane_z0.mark(boundary_markers, plane_z0_marker)

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
                plane_y0,
            ),
            dolfin.DirichletBC(
                W.sub(0).sub(2),  # u_z
                dolfin.Constant(0.0),
                plane_z0,
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


# FIXME : This can be done in a more elegant way
def torso_boundary_marking(
    torso_mesh: dolfin.Mesh,
    heart_mesh: dolfin.Mesh,
):

    logger.debug("Setup torso diriclet bc")

    # Get the value of the greatest x-coordinate
    Lx_h = heart_mesh.mpi_comm().allreduce(
        heart_mesh.coordinates().max(0)[0],
        op=MPI.MAX,
    )
    Ly_h = heart_mesh.mpi_comm().allreduce(
        heart_mesh.coordinates().max(0)[1],
        op=MPI.MAX,
    )
    Lz_h = heart_mesh.mpi_comm().allreduce(
        heart_mesh.coordinates().max(0)[2],
        op=MPI.MAX,
    )

    Lx_tmin = torso_mesh.mpi_comm().allreduce(
        torso_mesh.coordinates().min(0)[0],
        op=MPI.MIN,
    )
    Ly_tmin = torso_mesh.mpi_comm().allreduce(
        torso_mesh.coordinates().min(0)[1],
        op=MPI.MIN,
    )
    Lz_tmin = torso_mesh.mpi_comm().allreduce(
        torso_mesh.coordinates().min(0)[2],
        op=MPI.MIN,
    )

    Lx_tmax = torso_mesh.mpi_comm().allreduce(
        torso_mesh.coordinates().max(0)[0],
        op=MPI.MAX,
    )
    Ly_tmax = torso_mesh.mpi_comm().allreduce(
        torso_mesh.coordinates().max(0)[1],
        op=MPI.MAX,
    )
    Lz_tmax = torso_mesh.mpi_comm().allreduce(
        torso_mesh.coordinates().max(0)[2],
        op=MPI.MAX,
    )

    # Define domain to apply dirichlet boundary conditions
    heart_x0 = dolfin.CompiledSubDomain("near(x[0], 0) && on_boundary")
    heart_lx = dolfin.CompiledSubDomain("near(x[0], Lx) && on_boundary", Lx=Lx_h)
    heart_y0 = dolfin.CompiledSubDomain("near(x[1], 0) && on_boundary")
    heart_ly = dolfin.CompiledSubDomain("near(x[1], Ly) && on_boundary", Ly=Ly_h)
    heart_z0 = dolfin.CompiledSubDomain("near(x[2], 0) && on_boundary")
    heart_lz = dolfin.CompiledSubDomain("near(x[2], Lz) && on_boundary", Lz=Lz_h)

    torso_x0 = dolfin.CompiledSubDomain("near(x[0], Lx) && on_boundary", Lx=Lx_tmin)
    torso_lx = dolfin.CompiledSubDomain("near(x[0], Lx) && on_boundary", Lx=Lx_tmax)
    torso_y0 = dolfin.CompiledSubDomain("near(x[1], Ly) && on_boundary", Ly=Ly_tmin)
    torso_ly = dolfin.CompiledSubDomain("near(x[1], Ly) && on_boundary", Ly=Ly_tmax)
    torso_z0 = dolfin.CompiledSubDomain("near(x[2], Lz) && on_boundary", Lz=Lz_tmin)
    torso_lz = dolfin.CompiledSubDomain("near(x[2], Lz) && on_boundary", Lz=Lz_tmax)

    boundary_markers = dolfin.MeshFunction(
        "size_t",
        torso_mesh,
        torso_mesh.topology().dim() - 1,
    )
    boundary_markers.set_all(0)

    interior_marker = 1
    heart_x0.mark(boundary_markers, interior_marker)
    heart_y0.mark(boundary_markers, interior_marker)
    heart_z0.mark(boundary_markers, interior_marker)
    heart_lx.mark(boundary_markers, interior_marker)
    heart_ly.mark(boundary_markers, interior_marker)
    heart_lz.mark(boundary_markers, interior_marker)

    exterior_marker = 2
    torso_x0.mark(boundary_markers, exterior_marker)
    torso_y0.mark(boundary_markers, exterior_marker)
    torso_z0.mark(boundary_markers, exterior_marker)
    torso_lx.mark(boundary_markers, exterior_marker)
    torso_ly.mark(boundary_markers, exterior_marker)
    torso_lz.mark(boundary_markers, exterior_marker)

    return boundary_markers
