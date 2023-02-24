import typing

import dolfin
import pulse

from . import config
from . import lvgeometry
from . import slabgeometry
from . import utils


logger = utils.getLogger(__name__)


def create_slab_boundary_conditions(
    geo: slabgeometry.SlabGeometry,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = config.Config.fix_right_plane,
) -> pulse.BoundaryConditions:
    """Completely fix the left side of the mesh in the x-direction (i.e the side with the
    lowest x-values), fix the plane with y=0 in the y-direction, fix the plane with
    z=0 in the z-direction and apply some boundary condition to the right side.


    Parameters
    ----------
    geo : simcardems.geometry.SlabGeometry
        A slab geometry
    pre_stretch : typing.Union[dolfin.Constant, float], optional
        Value representing the amount of pre stretch, by default None
    traction : typing.Union[dolfin.Constant, float], optional
        Value representing the amount of traction, by default None
    spring : typing.Union[dolfin.Constant, float], optional
        Value representing the stiffness of the string, by default None
    fix_right_plane : bool, optional
        Fix the right plane so that it is not able to move in any direction
        except the x-direction, by default False


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

    logger.debug(
        f"Calling create_slab_boundary_conditions geo: {geo!r}, traction: "
        f"{traction!r}, spring: {spring!r}, pre_stretch: {pre_stretch!r} "
        f"and fix_right_plane: {fix_right_plane}",
    )

    def dirichlet_bc(W):
        # BC with fixing left size
        bcs = [
            dolfin.DirichletBC(
                W.sub(0).sub(0),
                dolfin.Constant(0.0),
                geo.ffun,
                geo.markers["X0"][0],
            ),
            dolfin.DirichletBC(
                W.sub(0).sub(1),  # u_y
                dolfin.Constant(0.0),
                geo.ffun,
                geo.markers["Y0"][0],
            ),
            dolfin.DirichletBC(
                W.sub(0).sub(2),  # u_z
                dolfin.Constant(0.0),
                geo.ffun,
                geo.markers["Z0"][0],
            ),
        ]
        if fix_right_plane:
            bcs.append(
                dolfin.DirichletBC(
                    W.sub(0).sub(0),  # u_x
                    dolfin.Constant(0.0),
                    geo.ffun,
                    geo.markers["X1"][0],
                ),
            )

        if pre_stretch is not None:
            bcs.append(
                dolfin.DirichletBC(
                    W.sub(0).sub(0),
                    utils.float_to_constant(pre_stretch),
                    geo.ffun,
                    geo.markers["X1"][0],
                ),
            )
        return bcs

    neumann_bc = []
    if traction is not None:
        neumann_bc.append(
            pulse.NeumannBC(
                traction=utils.float_to_constant(traction),
                marker=geo.markers["X1"][0],
            ),
        )

    robin_bc = []
    if spring is not None:
        robin_bc.append(
            pulse.RobinBC(
                value=utils.float_to_constant(spring),
                marker=geo.markers["X1"][0],
            ),
        )

    return pulse.BoundaryConditions(
        dirichlet=(dirichlet_bc,),
        neumann=neumann_bc,
        robin=robin_bc,
    )


def create_lv_boundary_conditions(
    geo: lvgeometry.LeftVentricularGeometry,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
):
    logger.debug(
        "Calling create_lv_boundary_conditions with geo: "
        f"{geo!r}, traction: {traction!r} and spring: {spring!r}",
    )

    def dirichlet_bc(W):
        # Completely fix the base
        return [
            dolfin.DirichletBC(
                W.sub(0).sub(0),
                dolfin.Constant(0.0),
                geo.ffun,
                geo.markers["BASE"][0],
            ),
        ]

    neumann_bc = []
    if traction is not None:
        # LV pressure
        neumann_bc.append(
            pulse.NeumannBC(
                traction=utils.float_to_constant(traction),
                marker=geo.markers["ENDO"][0],
            ),
        )

    robin_bc = []
    if spring is not None:
        # Pericardium
        robin_bc.append(
            pulse.RobinBC(
                value=utils.float_to_constant(spring),
                marker=geo.markers["EPI"][0],
            ),
        )

    return pulse.BoundaryConditions(
        dirichlet=(dirichlet_bc,),
        neumann=neumann_bc,
        robin=robin_bc,
    )
