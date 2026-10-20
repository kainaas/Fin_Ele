# Chapter 8 report: Computational Exercise 7

Source of the assignment: `notes-2026-8.pdf`, section 8.5 "Computational Exercise 7"
(pp. 83–84). The exercise belongs to the notes' chapter 8 ("Firedrake behind the
scenes"), so everything lives in `deliverables/chapter-8/`. The notebook is named
`c_7-Kaina-15466391.ipynb` because the course numbers computational exercises
independently of chapters (exercise 7 sits in chapter 8, exercise 8 in chapter 9).

## 1. Requirements and status

| # | Requirement (notes, 8.5) | Status | Where |
|---|---|---|---|
| 1 | No Firedrake/UFL for assembly; own quadrature, basis functions, local and global sparse matrices | done | `mesh.py`, `quadrature.py`, `solve_projection.py`, `solve_elliptic.py` |
| 2 | L2 projection on discontinuous P1 and continuous P1; H1 projection only on the continuous space ("why?") | done, explanation written | `solve_projection.py`, notebook |
| 3 | Error vs. h and contours of the projections | done | notebook |
| 4 | Solve −∇·(μ∇u)+σu=f with mixed BCs, cases 1–3 | done | `solve_elliptic.py`, notebook |
| – | Tests and independent verification | done | `test_fem.py`, `check_vs_firedrake.py` |

Not in scope for this chapter: the notes' chapter 8 has no theoretical exercises
(only Computational Exercise 7). Chapter 7 ("Interpolation and error estimates")
does have inline "show that…" items; no chapter-7 files exist yet.

## 2. Audit of the existing code (commit `1c14303`, "Half of chapter 8 done")

I ran the draft before changing anything:

* P1 reproduction (a P1 function must be recovered exactly): error 3e-16 (CG1 L2),
  5e-15 (CG1 H1), 3e-16 (DG1 L2). The mathematics of the draft is correct.
* Quadrature: `strang3`, `strang7`, `strang10` have 4, 7 and 13 points and are exact
  up to degree 3, 5 and 7 (weights sum to 1; `integrate` divides by the area 1/2).
  The name follows the tabulated rule from the Burkardt page cited in the file
  (the "10" is not the number of points), so this is **not** a bug; I only clarified
  the docstring, which could be read as listing the point counts.
* Convergence of the draft's projections: L2 order ≈ 2 (DG1 and CG1), H1-projection
  H1 order ≈ 1.
* Speed is adequate: assembly + solve takes 0.1 s (N=16), 1.4 s (N=64); error
  evaluation 0.6 s (N=64). N=128 is feasible, so no rewrite was needed.

### Mistakes and weaknesses found

Behavioural (could silently give wrong results or confusing failures):

1. `H1_projection` on a DG1 mesh printed a message and returned `None`, leaving
   `uh.coeffs = None`; the failure surfaced later (or never). Now raises `ValueError`
   with the reason.
2. `Mesh.UnitarySquareMesh` with an unknown space printed a message and returned
   `None`. Now raises `ValueError`.
3. `H1_error`/`L2_error` printed "Must have coefficients" and returned `None`.
   Now raise `ValueError`. `H1_error` also rejects DG1 (the broken gradient would
   ignore the jumps, so the number would not be an H1 error).

Cosmetic / API:

4. `L2_projection` carried comments copy-pasted from the H1 projection (it mentioned
   `grad(u).grad(phi)` terms) and an unused `grad_hat`; docstrings had an empty
   "Returns" section.
5. `L2_projection` and `L2_error` required a `grad_u` argument they never use. It is
   now optional (still accepted, so existing calls work).
6. The error docstrings said "Gaussian quadrature" but the rules are Strang–Fix.
7. `solve_projection.py` imported `Quadrature` twice; `point()` was annotated with
   the function `np.array` instead of the type `np.ndarray`.

Missing with respect to the assignment (what had to be implemented):

8. No way to identify boundary DoFs (needed for Dirichlet conditions).
9. `plot_surface` draws a 3-D surface; the assignment asks for contours.
10. No elliptic solver (item 4), no study of error vs. h, no explanation of why the
    H1 projection needs the continuous space, no tests, no notebook.

Left as is on purpose (not wrong): the element/DoF object model, `lil_matrix`
assembly, the per-quadrature-point Python loops (`np.einsum`-style vectorisation
would be faster but is not needed and would replace the student's own design).

## 3. What was implemented, and why

* `mesh.py`: `Mesh.boundary_dofs(side)` returns the unique CG1 DoFs on a side of the
  square (tolerance 1e-12). `Function.triangulation()` and `Function.plot_contour()`
  plot contours for both spaces. In DG1 every cell owns three DoFs with distinct ids
  at coinciding positions, so the contour plot shows the jumps between cells.
* `solve_elliptic.py` follows the draft's structure (element loop, `Element.Fk` at the
  quadrature points, local→global `lil_matrix`). Coefficients may be numbers or
  callables `c(x)`, evaluated at the physical points `Fk(x̂_g)`. Local matrix:
  `μ ∇φ_r·∇φ_s + σ φ_r φ_s`; local RHS: `f φ_r`.
  * *Neumann* (−μ∇u·n = 0) is the natural condition, so nothing is assembled on Γ_N.
  * *Dirichlet* by lifting: set u=g at boundary DoFs, subtract `A[:,fixed]·g` from the
    RHS, solve the reduced system on the free DoFs. This keeps the matrix symmetric
    positive definite and avoids penalty parameters. Where two Dirichlet sides meet
    their data must agree; in case 3 the only corner shared by Γ_left and
    Γ_bottom is (0,0), where both g vanish.
  * Case 1 (Γ_D=∅) is a pure Neumann problem; it is well posed only because σ=1>0.
* **Why H1 projection only on CG1:** the H1 inner product requires weak gradients;
  continuous piecewise polynomials are in H¹, discontinuous ones are not (jump ⇒
  Dirac in the distributional gradient). Written out in the notebook and enforced in
  the code.
* Error norms use the 13-point (degree 7) rule, much more accurate than the
  discretisation error, so quadrature does not pollute the convergence study.
* Test function for projections: `u = sin(2πx) sin(2πy)` (the same one the draft's
  `__main__` used).
* Manufactured solution for the solver, with the case-3 coefficients:
  `u = cos(πx) cos(πy)` satisfies the homogeneous Neumann condition on x=1 and y=1,
  so only `f` and the Dirichlet data on x=0, y=0 are needed.
  `f = 2πx sin(πx)cos(πy) + 2πy cos(πx) sin(πy) + (2π²μ + σ) u`.

## 4. Verification (all numbers measured on the final code)

* `pytest test_fem.py`: 13 tests pass (quadrature exactness for all three rules,
  mesh orientation/area, DoF counts and boundary sets, error on unknown space,
  P1 reproduction for the three projections, H1-on-DG1 rejected, case 1 exact,
  manufactured-solution rates).
* `check_vs_firedrake.py` (N=16, same triangulation, Firedrake `diagonal="left"`):
  max nodal difference 1.2e-13 (case 1), 2.9e-16 (case 2), 1.2e-15 (case 3). The
  triangulations must match: with the other diagonal the solutions differ at O(h²)
  and the comparison is meaningless. Firedrake had to be run with
  `env -u PETSC_DIR -u PETSC_ARCH` because the shell exports different PETSc paths
  than the install expects.
* While developing I also wrote an independent array-based implementation; it gave the
  same errors to the digits printed. It was not kept in the repo, so the reproducible
  evidence is the Firedrake comparison and the tests above.

Projection errors (N = 4…128; h = √2/N):

| N | DG1 L2-proj, L2 | CG1 L2-proj, L2 | CG1 H1-proj, L2 | CG1 H1-proj, H1 |
|---|---|---|---|---|
| 16 | 4.950e-03 | 6.638e-03 | 1.965e-02 | 8.513e-01 |
| 64 | 3.110e-04 | 4.025e-04 | 1.280e-03 | 2.177e-01 |
| 128 | 7.776e-05 | 1.005e-04 | 3.206e-04 | 1.090e-01 |

Observed rates at the finest refinement: L2 errors 2.0, H1 error 1.0, as theory
predicts for P1. The L2 projection onto CG1 is more accurate in L2 than the H1
projection (it minimises exactly that norm), while the H1 projection wins in H1.

Nodal extrema of the projections of u = sin(2πx) sin(2πy) (|u| ≤ 1), measured:

| N | L2 DG1 | L2 CG1 | H1 CG1 |
|---|---|---|---|
| 8 | ±1.103 | ±1.110 | −0.984 / 0.983 |
| 16 | ±1.030 | ±1.026 | −0.996 / 0.996 |

So the L2 projections overshoot the range of u by about 10 % at N=8 (shrinking with
refinement), while the H1 projection stays inside [−1, 1] in these runs. I did not
investigate why; the H1 projection has no general maximum principle. (The colorbars in
the contour figure reach ±1.2 only because matplotlib rounds the contour levels.)

Elliptic solver: case 1 gives u=1 to 1.3e-12; cases 2 and 3 have max 0.0698 and
0.3676 at N=64; manufactured solution rates 2.00 (L2) and 1.00 (H1) at N=128.

## 5. Mistakes I made along the way (for the record)

* I first put this work in `deliverables/chapter-7/` and wrote a new vectorised
  code base. The exercise belongs to chapter 8 and the existing draft was sound.
  I discarded the chapter-7 folder (it was untracked output of mine) and built on
  the draft instead.
* In my first audit I called `strang10` misnamed and said the chapter-8 files were
  uncommitted and "too slow". All three were wrong: the name follows the tabulated
  rule, the files are in commit `1c14303`, and the timings above show it is fast
  enough.
* My first Firedrake comparison showed 7e-6 and 1e-3 differences; the cause was a
  different diagonal split, not a bug in either code.

## 6. Remaining / open items

* Nothing of this exercise is committed; `git status` shows the modified draft
  files and the new files.
* Environment note: the notebook needs numpy, scipy and matplotlib. The Firedrake
  virtualenv on this machine has no matplotlib, so the notebook was executed in a
  separate virtualenv (outputs are stored in the `.ipynb`). `check_vs_firedrake.py`
  needs Firedrake plus matplotlib (imported by `mesh.py`).
* Possible extras if wanted: element-wise (block-diagonal) DG1 L2 projection; a
  vectorised assembly; chapter-7 theoretical list and notebook.
