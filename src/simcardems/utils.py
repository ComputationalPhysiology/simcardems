import logging
import typing
from pathlib import Path

import dolfin
import numpy as np
import ufl

PathLike = typing.Union[Path, str]


class MPIFilt(logging.Filter):
    def filter(self, record):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            return 1
        else:
            return 0


mpi_filt = MPIFilt()


def getLogger(name):
    import daiquiri

    logger = daiquiri.getLogger(name)
    logger.logger.addFilter(mpi_filt)
    return logger


logger = getLogger(__name__)


class Projector:
    def __init__(
        self,
        V: dolfin.FunctionSpace,
        solver_type: str = "lu",
        preconditioner_type: str = "default",
    ):
        """
        Projection class caching solver and matrix assembly

        Args:
            V (dolfin.FunctionSpace): Function-space to project in to
            solver_type (str, optional): Type of solver. Defaults to "lu".
            preconditioner_type (str, optional): Type of preconditioner. Defaults to "default".

        Raises:
            RuntimeError: _description_
        """
        u = dolfin.TrialFunction(V)
        self._v = dolfin.TestFunction(V)
        self._dx = dolfin.Measure("dx", domain=V.mesh())
        self._b = dolfin.Function(V)
        self._A = dolfin.assemble(ufl.inner(u, self._v) * self._dx)
        lu_methods = dolfin.lu_solver_methods().keys()
        krylov_methods = dolfin.krylov_solver_methods().keys()
        if solver_type == "lu" or solver_type in lu_methods:
            if preconditioner_type != "default":
                raise RuntimeError("LUSolver cannot be preconditioned")
            self.solver = dolfin.LUSolver(self._A, "default")
        elif solver_type in krylov_methods:
            self.solver = dolfin.PETScKrylovSolver(
                self._A,
                solver_type,
                preconditioner_type,
            )
        else:
            raise RuntimeError(
                f"Unknown solver type: {solver_type}, method has to be lu"
                + f", or {np.hstack(lu_methods, krylov_methods)}",
            )
        self.solver.set_operator(self._A)

    def project(self, u: dolfin.Function, f: ufl.core.expr.Expr) -> None:
        """
        Project `f` into `u`.

        Args:
            u (dolfin.Function): The function to project into
            f (ufl.core.expr.Expr): The ufl expression to project
        """
        dolfin.assemble(ufl.inner(f, self._v) * self._dx, tensor=self._b.vector())
        self.solver.solve(u.vector(), self._b.vector())

    def __call__(self, u: dolfin.Function, f: ufl.core.expr.Expr) -> None:
        self.project(u=u, f=f)


def enum2str(x, EnumCls):
    if isinstance(x, str):
        return x
    assert x in EnumCls
    return dict(zip(EnumCls.__members__.values(), EnumCls.__members__.keys()))[x]


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


def local_project(v, V, u=None):
    metadata = {"quadrature_degree": 3, "quadrature_scheme": "default"}
    dxm = dolfin.dx(metadata=metadata)
    dv = dolfin.TrialFunction(V)
    v_ = dolfin.TestFunction(V)
    a_proj = dolfin.inner(dv, v_) * dxm
    b_proj = dolfin.inner(v, v_) * dxm
    solver = dolfin.LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = dolfin.Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return


def compute_norm(x, x_prev):
    x_norm = x.vector().norm("l2")
    e = x.vector() - x_prev.vector()
    norm = e.norm("l2")
    if x_norm > 0:
        norm /= x_norm
    return norm


def remove_file(path):
    path = Path(path)
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        if path.is_file():
            path.unlink()
    dolfin.MPI.barrier(dolfin.MPI.comm_world)


def setup_assigner(vs, index):
    # Set-up separate potential function for post processing
    VS0 = vs.function_space().sub(index)
    V = VS0.collapse()
    v = dolfin.Function(V)
    # Set-up object to optimize assignment from a function to subfunction
    v_assigner = dolfin.FunctionAssigner(V, VS0)
    v_assigner.assign(v, vs.sub(index))
    return v, v_assigner


# No optimization - sub() is costly
# def sub_function(vs, index):
#     return vs.sub(index)

# Optimization level 1 : call copy constructor as done in sub() but without safety checks
# def sub_function(vs, index):
#     return dolfin.Function(vs, index, name='%s-%d' % (str(vs), index))


# Optimization level 2 : define the subfunction as done in the copy constructor with
# even less safety checks
def sub_function(vs, index):
    sub_vs = dolfin.Function(dolfin.cpp.function.Function(vs._cpp_object, index))
    ufl.Coefficient.__init__(
        sub_vs,
        sub_vs.function_space().ufl_function_space(),
        count=sub_vs._cpp_object.id(),
    )
    return sub_vs


def print_mesh_info(mesh, total_dofs):
    logger.info(f"Mesh elements: {mesh.num_entities(mesh.topology().dim())}")
    logger.info(f"Mesh vertices: {mesh.num_entities(0)}")
    logger.info(f"Total degrees of freedom: {total_dofs}")
