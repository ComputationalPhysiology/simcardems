import logging
import os
import typing
from pathlib import Path

import dolfin
import ufl

PathLike = typing.Union[os.PathLike, str]


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
