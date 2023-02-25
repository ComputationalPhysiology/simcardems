from typing import Dict
from typing import Tuple

import dolfin
import pulse
from cardiac_geometries.geometry import MeshTypes

from .geometry import BaseGeometry


class LeftVentricularGeometry(BaseGeometry):
    @staticmethod
    def default_markers() -> Dict[str, Tuple[int, int]]:
        return {
            "BASE": (5, 2),
            "ENDO": (6, 2),
            "EPI": (7, 2),
        }

    def _default_microstructure(
        self,
        mesh: dolfin.Mesh,
        ffun: dolfin.MeshFunction,
    ) -> pulse.Microstructure:
        from cardiac_geometries import lv_ellipsoid_fibers

        return lv_ellipsoid_fibers.create_microstructure(
            function_space=self.parameters["fiber_space"],
            mesh=mesh,
            ffun=ffun,
            markers=self.markers,
            r_short_endo=self.parameters["r_short_endo"],
            r_short_epi=self.parameters["r_short_epi"],
            r_long_endo=self.parameters["r_long_endo"],
            r_long_epi=self.parameters["r_long_epi"],
            alpha_endo=self.parameters["fibers_angle_endo"],
            alpha_epi=self.parameters["fibers_angle_epi"],
        )

    def _default_ffun(self, mesh: dolfin.Mesh) -> dolfin.MeshFunction:
        raise NotImplementedError

    def _default_mesh(self) -> dolfin.Mesh:
        raise NotImplementedError

    @staticmethod
    def default_parameters():
        return {
            "num_refinements": 1,
            "fiber_space": "Quadrature_3",
            "fibers_angle_endo": -60.0,
            "fibers_angle_epi": 60.0,
            "mesh_type": MeshTypes.lv_ellipsoid.value,
            "mu_apex_endo": -3.141592653589793,
            "mu_apex_epi": -3.141592653589793,
            "mu_base_endo": -1.2722641256100204,
            "mu_base_epi": -1.318116071652818,
            "psize_ref": 3.0,
            "r_long_endo": 17.0,
            "r_long_epi": 20.0,
            "r_short_endo": 7.0,
            "r_short_epi": 10.0,
        }

    def validate(self):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters})"
