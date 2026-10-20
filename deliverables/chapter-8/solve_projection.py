import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from mesh import *
from quadrature import *

def phi_hat(x_hat: np.array):
    '''
    Compute all three basis functions
    '''
    return np.array([
        1.0 - x_hat[0] - x_hat[1],
        x_hat[0],
        x_hat[1]
    ])

def grad_phi_hat():
    '''
    Returns a matrix in which each line is the gradient of a basis function
    '''
    return np.array([
        [-1.0, -1.0],
        [ 1.0,  0.0],
        [ 0.0,  1.0]
    ])


def H1_projection(uh: Function, u, grad_u, QuadRule="strang10"):
    """
    Compute the H1 projection of u onto the finite element space
    associated with mesh.

    Parameters
    ----------
    uh: Function
        Function where the coefficients are going to be returned
    u : callable
        Function u(x), where x is a physical point in R^2.
    grad_u : callable
        Gradient of u, returned as an array of shape (2,).
    QuadRule : str
        Quadrature rule: "strang3", "strang7", or "strang10".

    Returns
    -------
    None. The coefficients are stored in uh.coeffs.
    """
    mesh = uh.mesh

    if mesh.FunctionSpace != "CG1":
        raise ValueError(
            "The H1 projection needs weak gradients, so the space must be "
            "contained in H1: use CG1 (DG1 functions jump across edges)."
        )

    n_dofs = mesh.num_dofs

    # Sparse global matrix and dense right-hand side.
    A = lil_matrix((n_dofs, n_dofs), dtype=float)
    b = np.zeros(n_dofs)

    # Gradients of the reference P1 basis functions.
    grad_hat = grad_phi_hat()

    points = Quadrature.points[QuadRule]
    weights = Quadrature.weights[QuadRule]

    
    # Loop over elements.
    for element in mesh.elements:

        dofs = np.array([d.id for d in element.dofs])

        det_J = abs(element.detBk)

        # grad(phi) = grad_hat @ Bk^{-1}
        grad_phi = grad_hat @ element.Bk_inv

        ae = np.zeros((3, 3))
        rhs_e = np.zeros(3)

        # Loop over quadrature points.
        for x_hat, w in zip(points, weights):

            phi = phi_hat(x_hat)

            x = element.Fk(x_hat)

            u_x = u(x)
            grad_u_x = np.asarray(grad_u(x), dtype=float)

            factor = 0.5 * det_J * w

            # Local RHS:
            # integral_K (u*phi_r + grad(u).grad(phi_r)) dx
            rhs_e += factor * (
                u_x * phi + grad_phi @ grad_u_x
            )

            # Local matrix:
            # integral_K (phi_r*phi_s
            #             + grad(phi_r).grad(phi_s)) dx
            ae += factor * (
                np.outer(phi, phi)
                + grad_phi @ grad_phi.T
            )

        # Assemble the local RHS.
        b[dofs] += rhs_e

        # Assemble the local matrix into the sparse global matrix.
        for r in range(3):
            for s in range(3):
                A[dofs[r], dofs[s]] += ae[r, s]

    # CSR format is suitable for sparse linear system solution.
    A = A.tocsr()

    coeffs = spsolve(A, b)

    uh.coeffs = coeffs



def L2_projection(uh: Function, u, grad_u=None, QuadRule="strang10"):
    """
    Compute the L2 projection of u onto the finite element space
    associated with mesh.

    Parameters
    ----------
    uh: Function
        Function where the coefficients are going to be returned
    u : callable
        Function u(x), where x is a physical point in R^2.
    grad_u : callable
        Not used by the L2 projection, kept so that both projections share
        the same call signature.
    QuadRule : str
        Quadrature rule: "strang3", "strang7", or "strang10".

    Returns
    -------
    None. The coefficients are stored in uh.coeffs.
    """
    mesh = uh.mesh

    n_dofs = mesh.num_dofs

    # Sparse global matrix and dense right-hand side.
    A = lil_matrix((n_dofs, n_dofs), dtype=float)
    b = np.zeros(n_dofs)

    points = Quadrature.points[QuadRule]
    weights = Quadrature.weights[QuadRule]

    # Loop over elements.
    for element in mesh.elements:

        dofs = np.array([d.id for d in element.dofs])

        det_J = abs(element.detBk)

        ae = np.zeros((3, 3))
        rhs_e = np.zeros(3)

        # Loop over quadrature points.
        for x_hat, w in zip(points, weights):

            phi = phi_hat(x_hat)

            x = element.Fk(x_hat)

            u_x = u(x)

            factor = 0.5 * det_J * w

            # Local RHS:  integral_K u*phi_r dx
            rhs_e += factor * (
                u_x * phi
            )

            # Local matrix:  integral_K phi_r*phi_s dx
            ae += factor * (
                np.outer(phi, phi)
            )

        # Assemble the local RHS.
        b[dofs] += rhs_e

        # Assemble the local matrix into the sparse global matrix.
        for r in range(3):
            for s in range(3):
                A[dofs[r], dofs[s]] += ae[r, s]

    # CSR format is suitable for sparse linear system solution.
    A = A.tocsr()

    coeffs = spsolve(A, b)

    uh.coeffs = coeffs



def H1_error(uh: Function, u, grad_u, QuadRule="strang10"):
    """
    Approximate the continuous H1 error ||u - u_h||_{H1(Omega)}
    using the chosen quadrature rule on each element (CG1 only).
    """
    if uh.mesh.FunctionSpace != "CG1":
        raise ValueError("The H1 error is only defined for CG1 functions.")


    grad_hat = grad_phi_hat()

    error_squared = 0.0

    coeffs = uh.coeffs
    if coeffs is None:
        raise ValueError("uh has no coefficients: compute a projection first.")

    mesh = uh.mesh
    for element in mesh.elements:
        dofs = np.array([d.id for d in element.dofs])
        c_e = coeffs[dofs]

        det_J = abs(element.detBk)
        grad_phi = grad_hat @ element.Bk_inv

        # Constant gradient of the P1 approximation on this element.
        grad_uh = c_e @ grad_phi

        def integrand(x_hat):
            phi = phi_hat(x_hat)

            x = element.Fk(x_hat)

            u_exact = u(x)
            u_h = c_e @ phi
            grad_exact = np.asarray(grad_u(x), dtype=float)

            value_error = u_exact - u_h
            grad_error = grad_exact - grad_uh

            return value_error**2 + np.dot(grad_error, grad_error)

        # The quadrature class integrates over the reference triangle.
        # Apply the Jacobian determinant to obtain the physical integral.
        error_squared += det_J * Quadrature.integrate(
            integrand, QuadRule
        )

    return np.sqrt(max(error_squared, 0.0))


def L2_error(uh: Function, u, grad_u=None, QuadRule="strang10"):
    """
    Approximate the continuous L2 error ||u - u_h||_{L2(Omega)}
    using the chosen quadrature rule on each element (grad_u is unused).
    """

    error_squared = 0.0

    coeffs = uh.coeffs
    if coeffs is None:
        raise ValueError("uh has no coefficients: compute a projection first.")

    mesh = uh.mesh
    for element in mesh.elements:
        dofs = np.array([d.id for d in element.dofs])
        c_e = coeffs[dofs]

        det_J = abs(element.detBk)

        def integrand(x_hat):
            phi = phi_hat(x_hat)

            x = element.Fk(x_hat)

            u_exact = u(x)
            u_h = c_e @ phi

            value_error = u_exact - u_h

            return value_error**2

        # The quadrature class integrates over the reference triangle.
        # Apply the Jacobian determinant to obtain the physical integral.
        error_squared += det_J * Quadrature.integrate(
            integrand, QuadRule
        )

    return np.sqrt(max(error_squared, 0.0))



if __name__ == "__main__":
    for N in [8, 16, 32, 64, 128]:
        mesh = Mesh.UnitarySquareMesh(N, N, "DG1")
        u = lambda x: np.sin(2.0* np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])
        grad_u = lambda x: np.array([
            2.0*np.pi * np.cos(2.0* np.pi * x[0]) * np.sin(2.0 * np.pi * x[1]), 
            2.0*np.pi * np.sin(2.0* np.pi * x[0]) * np.cos(2.0 * np.pi * x[1])
        ])

        uh = Function(mesh)
        L2_projection(uh, u, grad_u)
        if N == 8:
            uh.plot_surface()
            plt.show()

        print(L2_error(uh, u, grad_u))