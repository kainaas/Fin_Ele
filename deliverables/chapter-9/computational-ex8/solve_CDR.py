import firedrake as fd
from firedrake.output import VTKFile 

import gmsh
import numpy as np

#--------------------------------------------------------------------
#--- Preprocess: Mesh generation, boundary and region identification

leftdom_marker = 2
rightdom_marker = 1
leftwall_marker = 1
rightwall_marker = 0

def CreateMesh(Lx, Ly, flag_refine_interface, flag_refine_exit):

    gmsh.initialize()

    # We create one rectangle for each subdomain
    gmsh.model.occ.addRectangle(0, 0, 0, Lx/2, Ly, tag=1)
    gmsh.model.occ.addRectangle(Lx/2, 0, 0, Lx/2, Ly, tag=2)
    # We fuse the two rectangles and keep the interface between them 
    whole_domain = gmsh.model.occ.fragment([(2,1)],[(2,2)])
    gmsh.model.occ.synchronize()
   
    # Mark the left and right subdomains
    leftdom, rightdom = [], [] # None, None
    for surface in whole_domain[0]:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [Lx/4,Ly/2, 0]):
            leftdom.append(surface)
            gmsh.model.addPhysicalGroup(2, [surface[1]], leftdom_marker)
        else:
            rightdom.append(surface)
            gmsh.model.addPhysicalGroup(2, [surface[1]], rightdom_marker)

    # Tag the left and right boundaries
    leftwall, rightwall, interface = [], [], []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], 0):
            leftwall.append(line[1])
        if np.isclose(com[0], Lx):
            rightwall.append(line[1])
        if np.isclose(com[0], Lx/2):
            interface.append(line[1])
    gmsh.model.addPhysicalGroup(1, leftwall, leftwall_marker)
    gmsh.model.addPhysicalGroup(1, rightwall, rightwall_marker)

    if(flag_refine_exit):
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", rightwall)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        r = 0.05
        gmsh.model.mesh.field.setNumber(2, "LcMin", r / 2)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 1.5 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 2 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 4 * r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
    elif(flag_refine_interface):
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", interface)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        r = 0.05
        gmsh.model.mesh.field.setNumber(2, "LcMin", r / 2)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 1.5 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 2 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 4 * r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.05)
            
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.write("rectangle.msh")
    gmsh.finalize()
    mesh = fd.Mesh("rectangle.msh")
    
    return mesh

Lx, Ly = 2.0, 1.0
flag_refine_exit = False
flag_refine_interface = False
mymesh = CreateMesh(Lx, Ly, flag_refine_interface, flag_refine_exit)
VTKFile('mymesh.pvd').write(mymesh)
    
#--- End preprocessing
#-------------------------------------------------

#-------------------------------------------------
# Convection-Diffusion-Reaction (CDR) problem

#icase = 'REAC_BL'
icase = 'CONV_BL'
iestab = 'SUPG'
#iestab = 'NONE'

V = fd.FunctionSpace(mymesh, "CG", degree=1)
u, v = fd.TrialFunction(V), fd.TestFunction(V)

# Diffusive term
kappa = fd.Constant(1.0e-2)
a  = fd.inner(kappa*fd.grad(u), fd.grad(v)) * fd.dx

bcs = []
if  (icase == 'REAC_BL'): # Reaction and Source term
    sigma_val = 1.0
    sigma = fd.Constant(sigma_val)
    a += sigma * u * v * fd.dx
    #---------------------------------------------------------------------------
    # Auxiliary problem to define a piecewise constant function
    Qh = fd.FunctionSpace(mymesh, "DG", degree=0)
    source = fd.TrialFunction(Qh)
    vp = fd.TestFunction(Qh)
    ap = source*vp*fd.dx
    sourceL, sourceR = 0.0, sigma_val
    Lp = fd.Constant(sourceL)*vp*fd.dx(leftdom_marker)
    Lp += fd.Constant(sourceR)*vp*fd.dx(rightdom_marker)
    sourceh = fd.Function(Qh, name='sigma')
    opts={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "ksp_monitor": None}
    fd.solve(ap==Lp, sourceh, bcs=[], solver_parameters=opts)
    VTKFile("source.pvd").write(sourceh)
    #--------------------------------------------------------------------------
    L = ...
    
elif(icase == 'CONV_BL'): # Convective term
    sigma = fd.Constant(0.0)
    beta = fd.as_vector([1.0, 0.0])
    a = a + ...
    source = fd.Constant(0.0)
    L = ...
    bcs.append(...)
    bcs.append(...)
    
if(iestab == 'SUPG'): # Stabilization
    hk = fd.CellDiameter(mymesh)
    tauk = 1.0/(4*kappa/hk**2 + 2.0*fd.sqrt(fd.dot(beta,beta))/hk + sigma)
    
    lhs_eq = -fd.div(kappa*fd.grad(u)) + fd.dot(beta,fd.grad(u)) + sigma*u
    rhs_eq = source

    a = a + tauk * lhs_eq * ...
    L = L + tauk * rhs_eq * ...

# Solve the problem
uh = fd.Function(V, name='Temperature')
opts={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "ksp_monitor": None}
fd.solve(a == L, uh, bcs=bcs, solver_parameters=opts)

# Save the results
VTKFile('solution.pvd').write(uh)


    
