from dolfin import *


class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class NewtonSolver_ODE(NewtonSolver):
    def __init__(self):
        self.petsc_solver = PETScKrylovSolver()
        NewtonSolver.__init__(
            self, V.mesh().mpi_comm(), self.petsc_solver, PETScFactory.instance(),
        )

    def converged(self, r, p, i):
        self._converged_called = True
        return super(NewtonSolver_ODE, self).converged(r, p, i)

    def solver_setup(self, A, J, p, i):
        self._solver_setup_called = True
        PETScOptions.set("ksp_type", "cg")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "gamg")
        self.linear_solver().set_from_options()
        super(NewtonSolver_ODE, self).solver_setup(A, J, p, i)

    def update_solution(self, x, dx, rp, p, i):
        self._update_solution_called = True
        print("Let's solve the ODEs!")
        super(NewtonSolver_ODE, self).update_solution(x, dx, rp, p, i)

    # This is just to check if we are using the overloaded functions
    def check_overloads_called(self):
        assert getattr(self, "_converged_called", False)
        assert getattr(self, "_solver_setup_called", False)
        assert getattr(self, "_update_solution_called", False)


mesh = UnitSquareMesh(32, 32)

V = FunctionSpace(mesh, "CG", 1)
g = Constant(1.0)
bcs = [DirichletBC(V, g, "near(x[0], 1.0) and on_boundary")]
u = Function(V)
v = TestFunction(V)
f = Expression("x[0]*sin(x[1])", degree=2)
F = inner((1 + u**2) * grad(u), grad(v)) * dx - f * v * dx
J = derivative(F, u)

problem = Problem(J, F, bcs)
x = u.vector()

solver = NewtonSolver_ODE()
# solver.parameters["linear_solver"] = "cg"
# solver.parameters["preconditioner"] = "amg"
# solver.parameters["krylov_solver"]["monitor_convergence"] = True
# solver.parameters["report"] = True

# Check that subsequent solutions work and reuse preconditioner
solver.solve(problem, x)

# Check that overloading NewtonSolver members works
getattr(solver, "check_overloads_called", None)
