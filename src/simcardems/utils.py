import logging
from pathlib import Path

import dolfin
import numpy as np
import ufl

logger = logging.getLogger(__name__)


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


def create_boxmesh(Lx, Ly, Lz, dx=0.5, refinements=0):
    # Create computational domain [0, Lx] x [0, Ly] x [0, Lz]
    # with resolution prescribed by benchmark or more refinements

    N = lambda v: int(np.rint(v))
    mesh = dolfin.BoxMesh(
        dolfin.MPI.comm_world,
        dolfin.Point(0.0, 0.0, 0.0),
        dolfin.Point(Lx, Ly, Lz),
        N(Lx / dx),
        2,
        2,
    )

    for i in range(refinements):
        logger.info(f"Performing refinement {i + 1}")
        mesh = dolfin.refine(mesh, redistribute=False)

    return mesh
