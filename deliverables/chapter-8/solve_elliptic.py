import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from mesh import *
from quadrature import *
from solve_projection import phi_hat, grad_phi_hat


def _as_function(c):
    '''
    Coefficients can be given as numbers or as callables c(x), x in R^2.
    '''
    return c if callable(c) else (lambda x: float(c))


def solve_elliptic(mesh: Mesh, mu, sigma, f, dirichlet=None, QuadRule="strang10") -> Function:
    """
    Solve with CG1 elements

        -div(mu grad u) + sigma u = f      in Omega = (0,1)^2
                              u = g        on Gamma_D
                -mu grad u . n  = 0        on Gamma_N

    Weak form: find u_h with u_h = g on Gamma_D such that
        integral (mu grad u_h . grad v_h + sigma u_h v_h) = integral f v_h
    for all v_h vanishing on Gamma_D. The homogeneous Neumann condition is
    natural, so nothing is imposed on Gamma_N.

    Parameters
    ----------
    mesh : Mesh
        Must use the "CG1" function space.
    mu, sigma, f : float or callable
        Coefficients, evaluated at the physical quadrature points x.
    dirichlet : dict, optional
        Maps a side ("left", "right", "bottom", "top") to the callable g(x)
        imposed there. Sides that are absent are homogeneous Neumann. Where
        two Dirichlet sides meet, their data must agree (later entries win).
    QuadRule : str
        Quadrature rule: "strang3", "strang7", or "strang10".

    Returns
    -------
    Function whose coeffs are the nodal values of u_h.
    """
    if mesh.FunctionSpace != "CG1":
        raise ValueError("solve_elliptic is implemented for CG1 only")

    mu, sigma, f = _as_function(mu), _as_function(sigma), _as_function(f)

    n_dofs = mesh.num_dofs
    A = lil_matrix((n_dofs, n_dofs), dtype=float)
    b = np.zeros(n_dofs)

    grad_hat = grad_phi_hat()
    points = Quadrature.points[QuadRule]
    weights = Quadrature.weights[QuadRule]

    for element in mesh.elements:
        dofs = np.array([d.id for d in element.dofs])
        det_J = abs(element.detBk)
        grad_phi = grad_hat @ element.Bk_inv

        ae = np.zeros((3, 3))
        rhs_e = np.zeros(3)

        for x_hat, w in zip(points, weights):
            phi = phi_hat(x_hat)
            x = element.Fk(x_hat)
            factor = 0.5 * det_J * w

            # Local matrix: integral_K mu grad(phi_r).grad(phi_s) + sigma phi_r phi_s
            ae += factor * (
                mu(x) * grad_phi @ grad_phi.T
                + sigma(x) * np.outer(phi, phi)
            )
            # Local RHS: integral_K f phi_r
            rhs_e += factor * f(x) * phi

        b[dofs] += rhs_e
        for r in range(3):
            for s in range(3):
                A[dofs[r], dofs[s]] += ae[r, s]

    A = A.tocsr()

    # Dirichlet conditions by lifting: set u = g on the boundary DoFs, move the
    # corresponding columns to the right-hand side, solve for the free DoFs.
    uh = Function(mesh)
    coeffs = np.zeros(n_dofs)
    fixed = np.zeros(n_dofs, dtype=bool)
    for side, g in (dirichlet or {}).items():
        for d in mesh.boundary_dofs(side):
            coeffs[d.id] = g(d.pos)
            fixed[d.id] = True

    free = np.flatnonzero(~fixed)
    fix = np.flatnonzero(fixed)
    rhs = b[free] - A[free][:, fix] @ coeffs[fix]
    coeffs[free] = spsolve(A[free][:, free].tocsc(), rhs)

    uh.coeffs = coeffs
    return uh
