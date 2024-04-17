import dolfin
import pulse


quadrature_degree = 3
dolfin.parameters["form_compiler"]["quadrature_degree"] = quadrature_degree
dolfin.parameters["form_compiler"]["representation"] = "uflacs"


def local_project(v, V, u=None):
    metadata = {"quadrature_degree": quadrature_degree, "quadrature_scheme": "default"}
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


def subplus(x):
    return dolfin.conditional(dolfin.ge(x, 0.0), x, 0.0)


def strain_energy(F, p, f0, Ta):
    a = 0.059
    b = 8.023
    a_f = 18.472
    b_f = 16.026

    J = pulse.kinematics.Jacobian(F)
    Jm23 = pow(J, -float(2) / 3)
    C = F.T * F
    I1 = Jm23 * dolfin.tr(C)
    I4f = Jm23 * dolfin.inner(C * f0, f0)

    psi_passive = a / (2.0 * b) * (dolfin.exp(b * (I1 - 3)) - 1.0) + a_f / (2.0 * b_f) * (
        dolfin.exp(b_f * subplus(I4f - 1) ** 2) - 1.0
    )
    psi_incompressibilty = p * (J - 1)
    psi_active = dolfin.Constant(0.5) * Ta * (I4f - 1)
    return psi_passive + psi_incompressibilty + psi_active


use_virtual_work = True

mesh = dolfin.UnitCubeMesh(5, 5, 5)

P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

state_space = dolfin.FunctionSpace(mesh, P2 * P1)

state = dolfin.Function(state_space, name="state")
state_test = dolfin.TestFunction(state_space)

u, p = dolfin.split(state)
v, q = dolfin.split(state_test)

V_f = dolfin.FunctionSpace(
    mesh,
    dolfin.VectorElement(
        family="Quadrature",
        cell=mesh.ufl_cell(),
        degree=quadrature_degree,
        quad_scheme="default",
    ),
)
f0 = dolfin.interpolate(
    dolfin.Expression(("1.0", "0.0", "0.0"), degree=1, cell=mesh.ufl_cell()),
    V_f,
)


# Some mechanical quantities
F = dolfin.variable(pulse.kinematics.DeformationGradient(u))
Ta = dolfin.Constant(0.0)
psi = strain_energy(F=F, p=p, f0=f0, Ta=Ta)


# Boundary conditions
left = dolfin.CompiledSubDomain("near(x[0], 0) && on_boundary")
bc = dolfin.DirichletBC(
    state_space.sub(0),
    dolfin.Constant((0.0, 0.0, 0.0)),
    left,
)

if use_virtual_work:
    R = dolfin.derivative(
        psi * dolfin.dx,
        state,
        state_test,
    )

else:
    P = dolfin.diff(psi, F)
    J = pulse.kinematics.Jacobian(F)
    R = (dolfin.inner(P, dolfin.grad(v)) + q * (J - 1)) * dolfin.dx

dolfin.solve(
    R == 0,
    state,
    bcs=bc,
)

Ta.assign(dolfin.Constant(0.1))
dolfin.solve(R == 0, state, bcs=bc)

u, p = state.split()
dolfin.File("u.pvd") << u


f = F * f0
lmbda = dolfin.sqrt(f**2)

V_cg1 = dolfin.FunctionSpace(mesh, "CG", 1)
lmbda_cg1 = dolfin.project(lmbda, V_cg1)
dolfin.File("lmbda_cg1.pvd") << lmbda_cg1


V_quad = dolfin.FunctionSpace(
    mesh,
    dolfin.FiniteElement(
        family="Quadrature",
        cell=mesh.ufl_cell(),
        degree=quadrature_degree,
        quad_scheme="default",
    ),
)
lmbda_quad = dolfin.project(
    lmbda,
    V_quad,
    form_compiler_parameters={"representation": "quadrature"},
)

try:
    import ldrb  # pip install ldrb

    # Method for saving quadrature functions to xdmf
    ldrb.save.fun_to_xdmf(lmbda_quad, "lmbda_quad_project")
except ImportError:
    pass

lmbda_quad_local = local_project(
    lmbda_quad,
    V_cg1,
)
# Method for saving quadrature functions to xdmf
(dolfin.File("lmbda_cg1_local_project.pvd") << lmbda_quad_local,)
