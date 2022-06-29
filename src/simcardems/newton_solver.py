import dolfin
import pulse


class MechanicsNewtonSolver(dolfin.NewtonSolver):
    def __init__(
        self,
        problem: pulse.NonlinearProblem,
        state: dolfin.Function,
        update_cb=None,
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
        self.debug = params.pop("debug", False)
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
                "ksp_type": "preonly",
                # "ksp_type": "gmres",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_33": 0,
            },
            "newton_verbose": False,
            "ksp_verbose": False,
            "debug": False,
            "linear_solver": "mumps",
            # "linear_solver": "gmres",
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

        if self.debug:
            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                # Print all residuals
                residual = r.norm("l2")
                with open("residual.txt", "a") as rfile:
                    rfile.write(str(residual) + "\t")

        return super().converged(r, p, i)

    def solver_setup(self, A, J, p, i):
        self._solver_setup_called = True
        super().solver_setup(A, J, p, i)

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


class MechanicsNewtonSolver_ODE(MechanicsNewtonSolver):
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
