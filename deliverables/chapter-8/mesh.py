import numpy as np
import matplotlib.pyplot as plt

def point(x: float, y: float) -> np.ndarray:
    return np.array([x,y])

class DoF:
    def __init__(self, x:float, y:float, dof_id: int):
        self.pos = point(x,y)
        self.id = dof_id

class Element:
    def __init__(self, d1:DoF, d2:DoF, d3:DoF):
        self.dofs = [d1, d2, d3]
        self.barycenter = (d1.pos + d2.pos + d3.pos) / 3

        a = self.dofs[1].pos[0] - self.dofs[0].pos[0]
        b = self.dofs[2].pos[0] - self.dofs[0].pos[0]
        c = self.dofs[1].pos[1] - self.dofs[0].pos[1]
        d = self.dofs[2].pos[1] - self.dofs[0].pos[1]
        self.Bk = np.array([
            [a,b],
            [c,d]
        ])
        self.detBk = a * d - b* c

        self.Bk_inv = (1.0/self.detBk) * np.array([
            [d, -b],
            [-c, a]
        ])

        self.bk = self.dofs[0].pos

    def Fk(self, x_hat: np.array):
        '''
            Affine mapping from the master triangle to the element
        '''
        return self.Bk @ x_hat + self.bk

    def plot(self, ax: plt.Axes, show_ids:bool = True):
        x_plot = [self.dofs[i].pos[0] for i in range(3)]
        y_plot = [self.dofs[i].pos[1] for i in range(3)]
        ax.scatter(x_plot, y_plot, c='r')
        x_plot.append(self.dofs[0].pos[0])
        y_plot.append(self.dofs[0].pos[1])
        ax.plot(x_plot, y_plot, 'k')
        if show_ids:
            for d in self.dofs:
                pos_txt = 0.5 * d.pos + 0.5 * self.barycenter
                ax.text(pos_txt[0], pos_txt[1], f"{d.id}", fontsize='xx-small')

class Mesh:
    def __init__(self, FunctionSpace: str):
        self.elements = []
        self.h = 0.0
        self.num_dofs = 0
        self.FunctionSpace = FunctionSpace

    @classmethod
    def UnitarySquareMesh(cls, Nx:int, Ny:int, FunctionSpace: str):
        '''
            Nx and Ny are the number of elements in the x and y directions respectively.
            FunctionSpace might be "CG1" or "DG1".
        '''
        mesh = Mesh(FunctionSpace)
        id = 0
        hx = 1.0 / Nx
        hy = 1.0 / Ny
        mesh.h = np.sqrt(hx**2 + hy**2)

        if FunctionSpace == "CG1":
            last_top = []
            for j in range(Nx+1):
                v = DoF(j*hx, 0.0,id)
                last_top.append(v)
                id += 1
            
            for i in range(Ny):
                top = []
                for j in range(Nx+1):
                    v = DoF(j*hx, (i+1)*hy, id)
                    top.append(v)
                    id += 1
                    
                    if  j >= 1:
                        e1 = Element(last_top[j-1], last_top[j], top[j-1])
                        mesh.elements.append(e1)

                        e2 = Element(top[j], top[j-1], last_top[j])
                        mesh.elements.append(e2)
                
                last_top = top
                        

        elif FunctionSpace == "DG1":
            for i in range(Ny):
                for j in range(Nx):
                    e1 = Element(
                        DoF(j*hx, i*hy, id),
                        DoF((j+1)*hx, i*hy, id+1),
                        DoF(j*hx, (i+1)*hy, id+2)
                    )
                    id += 3
                    mesh.elements.append(e1)

                    e2 = Element(
                        DoF((j+1)*hx, (i+1)*hy, id),
                        DoF(j*hx, (i+1)*hy, id+1),
                        DoF((j+1)*hx, i*hy, id+2)
                    )
                    id += 3
                    mesh.elements.append(e2)

        else:
            raise ValueError(f"{FunctionSpace} is not a known function space")
        
        mesh.num_dofs = id

        return mesh

    def plot(self, ax: plt.Axes, show_ids: bool = True):
        for e in self.elements:
            e.plot(ax, show_ids)

    def boundary_dofs(self, side: str, tol: float = 1e-12) -> list:
        '''
            DoFs lying on one side of the unit square ("left", "right",
            "bottom" or "top"). Only meaningful for CG1, where a DoF is a
            vertex shared by neighbouring elements; each DoF is returned once.
        '''
        if self.FunctionSpace != "CG1":
            raise ValueError("boundary DoFs are only defined for CG1")
        on_side = {
            "left":   lambda p: abs(p[0]) < tol,
            "right":  lambda p: abs(p[0] - 1.0) < tol,
            "bottom": lambda p: abs(p[1]) < tol,
            "top":    lambda p: abs(p[1] - 1.0) < tol,
        }
        if side not in on_side:
            raise ValueError(f"unknown side {side!r}")
        found = {d.id: d for e in self.elements for d in e.dofs
                 if on_side[side](d.pos)}
        return [found[i] for i in sorted(found)]



class Function:
    def __init__(self, mesh:Mesh):
        self.mesh = mesh
        self.dofs = {
            d.id: d
            for e in mesh.elements
            for d in e.dofs
        }
        self.coeffs: np.ndarray | None = None

    def triangulation(self):
        '''
            Returns (points, values, triangles) with one entry per DoF, for
            plotting; DG1 DoFs of neighbouring elements overlap in space but
            have distinct ids, so the jumps between elements are preserved.
        '''
        if self.coeffs is None:
            raise ValueError("Function coefficients have not been set.")
        points = np.zeros((self.mesh.num_dofs, 2))
        for dof_id, dof in self.dofs.items():
            points[dof_id] = dof.pos
        triangles = np.array([[d.id for d in e.dofs] for e in self.mesh.elements])
        return points, np.asarray(self.coeffs), triangles

    def plot_contour(self, ax, levels=20, cmap="viridis", title=""):
        '''
            Filled contours of the finite element function.
        '''
        points, values, triangles = self.triangulation()
        cs = ax.tricontourf(points[:, 0], points[:, 1], triangles, values,
                            levels=levels, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.figure.colorbar(cs, ax=ax)
        return cs

    def plot_surface(self, ax=None, cmap="viridis"):
        """
        Plot the surface of the finite-element function.

        Assumes self.coeffs contains the global nodal coefficients.
        """

        if self.coeffs is None:
            raise ValueError("Function coefficients have not been set.")

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        # Collect mesh vertices and their function values.
        points = np.zeros((self.mesh.num_dofs, 2))
        values = np.zeros(self.mesh.num_dofs)

        for dof_id, dof in self.dofs.items():
            points[dof_id] = dof.pos
            values[dof_id] = self.coeffs[dof_id]

        # Triangular connectivity, using global DoF indices.
        triangles = np.array([
            [d.id for d in element.dofs]
            for element in self.mesh.elements
        ])

        ax.plot_trisurf(
            points[:, 0],
            points[:, 1],
            values,
            triangles=triangles,
            cmap=cmap,
            edgecolor="none"
        )

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$f_h(x,y)$")
        ax.set_title("Finite Element Function")

        return fig, ax


if __name__ == "__main__":
    mesh = Mesh.UnitarySquareMesh(8, 8, "CG1")
    fig = plt.figure(figsize=(1000,1000, 'px'))
    ax = fig.add_subplot(111)
    mesh.plot(ax)

    print(mesh.num_dofs)
    plt.show()