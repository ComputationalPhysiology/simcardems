import dolfin
import numpy as np

Lx = 2.0
Ly = 0.7
Lz = 0.3
dx = 0.5
N = lambda v: int(np.rint(v))
coarse_mesh = dolfin.BoxMesh(
    dolfin.MPI.comm_world,
    dolfin.Point(0.0, 0.0, 0.0),
    dolfin.Point(Lx, Ly, Lz),
    N(Lx / dx),
    2,
    2,
)

refinements = 4

mesh = coarse_mesh
for i in range(refinements):
    print("Performing refinement", i + 1)
    mesh = dolfin.refine(mesh, redistribute=False)

fine_mesh = mesh

print("Number of elements")
print(f"Coarse mesh: {coarse_mesh.num_cells()}")
print(f"Fine mesh: {fine_mesh.num_cells()}")

V_coarse = dolfin.FunctionSpace(coarse_mesh, "CG", 1)
V_fine = dolfin.FunctionSpace(fine_mesh, "CG", 1)

f_coarse = dolfin.interpolate(
    dolfin.Expression("sin(x[0])*cos(x[1])", degree=1),
    V_coarse,
)
f_fine = dolfin.interpolate(dolfin.Expression("sin(x[0])*cos(x[1])", degree=1), V_fine)


g_fine = dolfin.interpolate(f_coarse, V_fine)
g_coarse = dolfin.interpolate(f_fine, V_coarse)

diff_coarse = dolfin.Function(V_coarse)
diff_coarse.vector()[:] = f_coarse.vector() - g_coarse.vector()
print("Diff coarse: ")
print(
    f"Max: {dolfin.norm(diff_coarse.vector(), 'linf')}, L2: {dolfin.norm(diff_coarse.vector(), 'l2')}",
)

diff_fine = dolfin.Function(V_fine)
diff_fine.vector()[:] = f_fine.vector() - g_fine.vector()
print("Diff fine: ")
print(
    f"Max: {dolfin.norm(diff_fine.vector(), 'linf')}, L2: {dolfin.norm(diff_fine.vector(), 'l2')}",
)


dolfin.File("coarse_mesh.pvd") << coarse_mesh
dolfin.File("fine_mesh.pvd") << fine_mesh
dolfin.File("f_coarse.pvd") << f_coarse
dolfin.File("f_fine.pvd") << f_fine
dolfin.File("g_coarse.pvd") << g_coarse
dolfin.File("g_fine.pvd") << g_fine
dolfin.File("diff_coarse.pvd") << diff_coarse
dolfin.File("diff_fine.pvd") << diff_fine
