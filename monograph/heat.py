import numpy as np
import firedrake as fd
from firedrake.output import VTKFile


def solve_heat(
    mesh, 
    Vh, 
    method:str,
    dt: float,
    Ti: tuple[float, float],
    alpha, 
    f, 
    ic, 
    dbc: list, 
    nbc: dict|None = None, 
    save_name:str|None = None
):
    '''
    Solves the heat equation.

    Parameters
    ----------
    mesh: gmsh mesh
        mesh in which we're solving the equation.
    Vh: fd.FunctionSpace
        Chosen function space to get the result
    method: str
        may be "IE" (implicit euler) or "CN" (Crank-Nicolson)
    dt: float
        Timestep size
    Ti: tuple[float, float]
        The time interval. First field is the first time. SEcond field is the final time
    alpha: UFL expression
        Thermal diffusivity
    f: callable
        The source term of the PDE
    ic: UFL expression
        Initial condition of the IVP
    dbc: list
        list of fd.DirichletBC
    nbc: dict
        Dictionary with keys being where this bc is and value its UFL expression
    save_name: str|None
        If None, does not save the output for visualization. Else, saves the .pvd file
    -----------
    Returns
        returns the solution
    '''

    outfile: VTKFile
    if not(save_name is None):
        outfile = VTKFile(save_name)

    V_out = fd.FunctionSpace(mesh, 'CG', 1)
    
    u_ = fd.Function(Vh)
    u, v = fd.TrialFunction(Vh), fd.TestFunction(Vh)
    x = fd.SpatialCoordinate(mesh)

    u_.project(ic)

    uh = fd.Function(Vh)

    t = Ti[0] + dt
    Tf = Ti[1]
    if method == "IE":
        a = (
            alpha * fd.inner(fd.grad(u), fd.grad(v))
            + u * v / dt
        ) * fd.dx
        while(t <= Tf):
            L = (
                f(t) * v
                + u_ * v / dt
            ) * fd.dx

            if not(nbc is None):
                for key, val in nbc.items():
                    l += fd.inner(v, val) * fd.ds(key)
            fd.solve(a == L, uh, bcs=dbc,solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
            u_.assign(uh)
            t += dt
            outfile.write(fd.project(uh, V_out, name="Temperature"))

    if method == "CN":
        a = (
            fd.inner(u, v) /dt + fd.inner(fd.grad(u), fd.grad(v)) / 2.0
        ) * fd.dx
        while(t <= Tf):
            L = (
                (f(t) + f(t-dt)) * v / 2.0
                + u_ * v / dt - fd.inner(fd.grad(u_), fd.grad(v)) / 2.0
            ) * fd.dx

            if not(nbc is None):
                for key, val in nbc.items():
                    l += fd.inner(v, val) * fd.ds(key)
            fd.solve(a == L, uh, bcs=dbc,solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
            u_.assign(uh)
            t += dt
            outfile.write(fd.project(uh, V_out, name="Temperature"))
    else:
        raise RuntimeError("No known method for", method)





n = 32
mesh = fd.RectangleMesh(nx=n, ny=n, Lx=np.pi, Ly=np.pi)
Vh = fd.FunctionSpace(mesh, 'CG', degree=1)
x = fd.SpatialCoordinate(mesh)

alpha = fd.Constant(1.0)
ic = fd.sin(x[0]) * fd.sin(x[1])
source = lambda t: fd.exp(-t) * fd.sin(x[0]) * fd.sin(x[1])
dbcs = [fd.DirichletBC(Vh, fd.Constant(0.0), sub_domain="on_boundary")]

T = (0.0, 5.0)
dt = 0.05


solve_heat(mesh, Vh, "CN", dt, T, alpha, source, ic, dbcs, save_name="heat_CN.pvd")

