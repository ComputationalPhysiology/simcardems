import dolfin

from . import utils


class EMCoupling:
    def __init__(
        self,
        mesh,
        lmbda=dolfin.Constant(1.0),
        Zetas=dolfin.Constant(0.0),
        Zetaw=dolfin.Constant(0.0),
    ) -> None:

        self.mesh = mesh
        V = dolfin.FunctionSpace(mesh, "CG", 1)
        self.lmbda = dolfin.Function(V, name="lambda")
        self.lmbda.assign(lmbda)
        self.Zetas = dolfin.Function(V, name="Zetas")
        self.Zetas.assign(Zetas)
        self.Zetaw = dolfin.Function(V, name="Zetaw")
        self.Zetaw.assign(Zetaw)

    def register_ep_model(self, solver):
        self._ep_solver = solver
        self.vs = solver.solution_fields()[0]
        self.XS, self.XS_assigner = utils.setup_assigner(self.vs, 40)
        self.XW, self.XW_assigner = utils.setup_assigner(self.vs, 41)

    def update_mechanics(self):
        self.XS_assigner.assign(self.XS, utils.sub_function(self.vs, 40))
        self.XW_assigner.assign(self.XW, utils.sub_function(self.vs, 41))

    def update_ep(self):
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 46), self.lmbda)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 47), self.Zetas)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 48), self.Zetaw)
