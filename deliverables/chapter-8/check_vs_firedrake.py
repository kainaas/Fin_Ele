"""Cross-check solve_elliptic against Firedrake (used only for verification).

Both codes use the same triangulation (Firedrake's diagonal="left", which is
the split used by mesh.py), so the linear systems coincide and the nodal
values must agree to round-off.  Run from a shell where PETSC_DIR/PETSC_ARCH
match the Firedrake install, e.g.:
    env -u PETSC_DIR -u PETSC_ARCH <firedrake-venv>/bin/python check_vs_firedrake.py
"""
import numpy as np
from firedrake import (DirichletBC, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, TrialFunction, UnitSquareMesh,
                       VectorFunctionSpace, assemble, dx, grad, inner, solve)

from mesh import Mesh
from solve_elliptic import solve_elliptic

N = 16
fmesh = UnitSquareMesh(N, N, diagonal="left")
V = FunctionSpace(fmesh, "CG", 1)
x, y = SpatialCoordinate(fmesh)
u, v = TrialFunction(V), TestFunction(V)

# Firedrake boundary ids: 1 left, 2 right, 3 bottom, 4 top
cases = {
    "case 1": dict(fd=(1, 1, 1, []), mine=(1.0, 1.0, 1.0, None)),
    "case 2": dict(fd=(1, 1, 1, [DirichletBC(V, 0, "on_boundary")]),
                   mine=(1.0, 1.0, 1.0, {s: (lambda p: 0.0)
                                         for s in ("left", "right", "bottom", "top")})),
    "case 3": dict(fd=(1 + x**2 + y**2, x * y, x + y,
                       [DirichletBC(V, y * (1 - y), 1), DirichletBC(V, x * (1 - x), 3)]),
                   mine=(lambda p: 1 + p[0]**2 + p[1]**2, lambda p: p[0] * p[1],
                         lambda p: p[0] + p[1],
                         {"left": lambda p: p[1] * (1 - p[1]),
                          "bottom": lambda p: p[0] * (1 - p[0])})),
}

coords = assemble(Function(VectorFunctionSpace(fmesh, "CG", 1))
                  .interpolate(SpatialCoordinate(fmesh))).dat.data_ro
fd_to_mine = (np.rint(coords[:, 1] * N).astype(int) * (N + 1)
              + np.rint(coords[:, 0] * N).astype(int))  # mine: id = j*(N+1)+i

mesh = Mesh.UnitarySquareMesh(N, N, "CG1")
for name, c in cases.items():
    mu, sigma, f, bcs = c["fd"]
    sol = Function(V)
    solve((mu * inner(grad(u), grad(v)) + sigma * u * v) * dx == f * v * dx, sol,
          bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    ref = np.zeros(mesh.num_dofs)
    ref[fd_to_mine] = sol.dat.data_ro
    mine = solve_elliptic(mesh, *c["mine"][:3], dirichlet=c["mine"][3]).coeffs
    print(f"{name}: max nodal difference vs Firedrake = {np.abs(ref - mine).max():.2e}")
