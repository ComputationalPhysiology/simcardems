import abc
import typing

import dolfin
import pulse

from . import config
from . import geometry
from . import utils


class BaseBoundaryConditions(abc.ABC):
    @property
    def bcs(self):
        return pulse.BoundaryConditions(
            dirichlet=self.dirichlet(),
            neumann=self.neumann(),
            robin=self.robin(),
        )

    @abc.abstractmethod
    def dirichlet(
        self,
    ) -> typing.Iterable[
        typing.Union[
            typing.Callable[
                [dolfin.FunctionSpace],
                typing.Iterable[dolfin.DirichletBC],
            ],
            dolfin.DirichletBC,
        ]
    ]:
        ...

    @abc.abstractmethod
    def neumann(self) -> typing.Iterable[pulse.NeumannBC]:
        ...

    @abc.abstractmethod
    def robin(self) -> typing.Iterable[pulse.RobinBC]:
        ...


class SlabBoundaryConditions(BaseBoundaryConditions):
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
        except the x-direction, by default True


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

    def __init__(
        self,
        geo: geometry.SlabGeometry,
        pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
        traction: typing.Union[dolfin.Constant, float] = None,
        spring: typing.Union[dolfin.Constant, float] = None,
        fix_right_plane: bool = config.Config.fix_right_plane,
    ) -> None:
        self.pre_stretch = pre_stretch
        self.traction = traction
        self.spring = spring
        self.fix_right_plane = fix_right_plane
        self.geo = geo

    def dirichlet(self):
        def dirichlet_bc(W):
            # BC with fixing left size
            bcs = [
                dolfin.DirichletBC(
                    W.sub(0).sub(0),  # u_x
                    dolfin.Constant(0.0),
                    sub_domains=self.geo.ffun,
                    sub_domain=self.geo.markers["left"],
                ),
                dolfin.DirichletBC(
                    W.sub(0).sub(1),  # u_y
                    dolfin.Constant(0.0),
                    sub_domains=self.geo.ffun,
                    sub_domain=self.geo.markers["plane_y0"],
                ),
                dolfin.DirichletBC(
                    W.sub(0).sub(2),  # u_z
                    dolfin.Constant(0.0),
                    sub_domains=self.geo.ffun,
                    sub_domain=self.geo.markers["plane_z0"],
                ),
            ]

            if self.fix_right_plane:
                bcs.extend(
                    [
                        dolfin.DirichletBC(
                            W.sub(0).sub(0),  # u_x
                            dolfin.Constant(0.0),
                            sub_domains=self.geo.ffun,
                            sub_domain=self.geo.markers["right"],
                        ),
                    ],
                )

            if self.pre_stretch is not None:
                bcs.append(
                    dolfin.DirichletBC(
                        W.sub(0).sub(0),
                        utils.float_to_constant(self.pre_stretch),
                        sub_domains=self.geo.ffun,
                        sub_domain=self.geo.markers["right"],
                    ),
                )
            return bcs

        return (dirichlet_bc,)

    def neumann(self) -> typing.List[pulse.NeumannBC]:
        neumann_bc = []
        if self.traction is not None:
            neumann_bc.append(
                pulse.NeumannBC(
                    traction=utils.float_to_constant(self.traction),
                    marker=self.geo.markers["right"],
                ),
            )
        return neumann_bc

    def robin(self):
        robin_bc = []
        if self.spring is not None:
            robin_bc.append(
                pulse.RobinBC(
                    value=utils.float_to_constant(self.spring),
                    marker=self.geo.markers["right"],
                ),
            )
        return robin_bc
