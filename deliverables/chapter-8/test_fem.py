"""TEST MADE WITH AI"""
"""Sanity checks for the from-scratch FEM code.  Run with:  pytest test_fem.py"""
from math import factorial

import numpy as np
import pytest

from mesh import Function, Mesh
from quadrature import Quadrature
from solve_elliptic import solve_elliptic
from solve_projection import H1_error, H1_projection, L2_error, L2_projection

RULES = {"strang3": 3, "strang7": 5, "strang10": 7}


@pytest.mark.parametrize("rule,degree", RULES.items())
def test_quadrature_exact_on_monomials(rule, degree):
    for d in range(degree + 1):
        for i in range(d + 1):
            exact = factorial(i) * factorial(d - i) / factorial(d + 2)
            got = Quadrature.integrate(lambda p: p[0] ** i * p[1] ** (d - i), rule)
            assert got == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("space", ["CG1", "DG1"])
def test_mesh_geometry(space):
    m = Mesh.UnitarySquareMesh(6, 6, space)
    assert all(e.detBk > 0 for e in m.elements)
    assert sum(e.detBk for e in m.elements) / 2 == pytest.approx(1.0)


def test_dof_counts_and_boundary():
    assert Mesh.UnitarySquareMesh(8, 8, "CG1").num_dofs == 81
    assert Mesh.UnitarySquareMesh(8, 8, "DG1").num_dofs == 3 * 2 * 64
    m = Mesh.UnitarySquareMesh(8, 8, "CG1")
    for side in ("left", "right", "bottom", "top"):
        assert len(m.boundary_dofs(side)) == 9


def test_unknown_space_raises():
    with pytest.raises(ValueError):
        Mesh.UnitarySquareMesh(4, 4, "CG2")


LIN = lambda x: 1 + 2 * x[0] - 3 * x[1]
GLIN = lambda x: np.array([2.0, -3.0])


@pytest.mark.parametrize("space,proj", [("DG1", L2_projection),
                                        ("CG1", L2_projection),
                                        ("CG1", H1_projection)])
def test_projection_reproduces_p1(space, proj):
    m = Mesh.UnitarySquareMesh(5, 5, space)
    uh = Function(m)
    proj(uh, LIN, GLIN)
    assert L2_error(uh, LIN) < 1e-12


def test_h1_projection_rejects_dg():
    uh = Function(Mesh.UnitarySquareMesh(4, 4, "DG1"))
    with pytest.raises(ValueError):
        H1_projection(uh, LIN, GLIN)


def test_case1_exact_solution_is_one():
    uh = solve_elliptic(Mesh.UnitarySquareMesh(8, 8, "CG1"), 1.0, 1.0, 1.0)
    assert np.abs(uh.coeffs - 1).max() < 1e-10


def test_case3_manufactured_convergence():
    pi = np.pi
    u = lambda x: np.cos(pi * x[0]) * np.cos(pi * x[1])
    gu = lambda x: np.array([-pi * np.sin(pi * x[0]) * np.cos(pi * x[1]),
                             -pi * np.cos(pi * x[0]) * np.sin(pi * x[1])])
    mu = lambda x: 1 + x[0] ** 2 + x[1] ** 2
    sigma = lambda x: x[0] * x[1]
    f = lambda x: (2 * x[0] * pi * np.sin(pi * x[0]) * np.cos(pi * x[1])
                   + 2 * x[1] * pi * np.cos(pi * x[0]) * np.sin(pi * x[1])
                   + (2 * pi ** 2 * mu(x) + sigma(x)) * u(x))
    errs = []
    for N in (8, 16, 32):
        m = Mesh.UnitarySquareMesh(N, N, "CG1")
        uh = solve_elliptic(m, mu, sigma, f, {"left": u, "bottom": u})
        errs.append((L2_error(uh, u), H1_error(uh, u, gu)))
    errs = np.array(errs)
    rates = np.log2(errs[:-1] / errs[1:])  # columns: L2, H1
    assert rates[-1, 0] > 1.9 and rates[-1, 1] > 0.95
