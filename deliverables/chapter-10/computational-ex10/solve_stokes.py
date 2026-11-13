import firedrake as fd
from firedrake.output import VTKFile 
from firedrake.__future__ import interpolate 
import numpy as np

# Finite element mesh
Nx, Ny = 32, 32
Lx, Ly = 1.0, 1.0
mymesh = fd.RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=False)
tags = {'Bottom': 3, 'Top': 4, 'Left': 1, 'Right': 2}

def viscosity(u):
    return fd.Constant(1.0)

x = fd.SpatialCoordinate(mymesh)

body_force = fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)])
surface_force = fd.as_vector([fd.Constant(0.0), -fd.exp(-20.0*(x[0]-0.5*Lx)**2)])
noslip = fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)])

mybcs = {'Bottom': noslip}

def SolveStokes(msh, bf, sf, bcs):

    P2 = fd.VectorElement('Lagrange', 'triangle', 2)
    P1 = fd.FiniteElement('Lagrange', 'triangle', 1)
    TH = fd.MixedElement([P2,P1]) # Taylor-Hood
    Vh = fd.FunctionSpace(msh, TH)
    
    # Alternative way of defining it
    #P2 = fd.VectorFunctionSpace(msh, family="CG", degree=2)
    #P1 = fd.FunctionSpace(msh, family="CG", degree=1)
    #Vh = P2 * P1

    bclst = []
    for key in bcs:
        bclst.append(fd.DirichletBC(Vh.sub(0), bcs[key], tags[key]))
    
    sol = fd.Function(Vh)
    v, q = fd.TestFunctions(Vh)
    u, p = fd.split(sol)
    
    mu = viscosity(u)
    
    F  = 
    F -= 
    F -= 
    Jac = fd.derivative(F, sol)
    opts={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "ksp_monitor": None}
    fd.solve(F == 0, sol, bcs=bclst, J=Jac, solver_parameters=opts)
    
    sol.sub(0).rename("Velocity")
    VTKFile("velocity.pvd").write(sol.sub(0))
    sol.sub(0).rename("Pressure")
    VTKFile("pressure.pvd").write(sol.sub(1))

    return

SolveStokes(mymesh, body_force, surface_force, mybcs)

