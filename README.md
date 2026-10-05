# gnoma

## Background: physics and approach

**gnoma** stands for **guided-wave nonlinear optical mode analysis**. It connects a waveguide's geometry and material properties to its optical modes, and then uses those modes to calculate quantities such as coupling and nonlinear frequency conversion. The central workflow is:

**Geometry and materials → refractive-index profile → optical modes → device properties.**

This section assumes introductory electromagnetism, differential equations, and eigenvectors. No previous experience with numerical waveguide modeling is required.

### What is an optical mode?

A dielectric waveguide confines light transversely while allowing it to propagate along its length. A simple example is a higher-index core surrounded by a lower-index cladding. At dimensions comparable to the wavelength, the useful description is not a collection of rays, but a set of electromagnetic field patterns called **modes**.

Consider a straight waveguide whose cross-section is independent of the propagation coordinate, $`z`$. A monochromatic mode has the form

```math
\mathbf E(x,y,z,t)
=
\mathrm{Re}\left[
 a\,\mathbf e(x,y)e^{i(\beta z-\omega t)}
\right].
```

Here $`\mathbf e(x,y)`$ is the transverse distribution of the electric-field vector, $`a`$ is its excitation amplitude, $`\omega`$ is the angular frequency, and $`\beta`$ is the **propagation constant**. The magnetic field has the same longitudinal and temporal dependence. Although its distribution is specified on a transverse plane, the electric field can have a longitudinal component, $`e_z`$.

For a lossless mode, propagation changes the phase but not the transverse profile. A mode is therefore analogous to a normal mode of coupled oscillators: it is a pattern that evolves without changing its shape. The effective index is defined by

```math
n_{\mathrm{eff}}=\frac{\beta}{k_0},
\qquad
k_0=\frac{\omega}{c}=\frac{2\pi}{\lambda},
```

where $`c`$ is the speed of light in vacuum and $`\lambda`$ is the **vacuum wavelength**.

**Material index and effective index are different quantities.** The material index describes the local dielectric response. The effective index describes the phase accumulation of an entire guided field pattern; it depends on geometry, wavelength, and polarization as well as the materials.

For an ordinary lossless step-index guide with a uniform cladding, a bound mode satisfies

```math
n_{\mathrm{clad}} \lt n_{\mathrm{eff}} \lt n_{\mathrm{core}}.
```

The field is not restricted to the core: it has an evanescent tail in the cladding. This simple index bound is a useful check for that geometry, not a universal classification rule for every anisotropic or multilayer structure.

### Why is finding a mode an eigenvalue problem?

A scalar model makes the connection to familiar mathematics explicit. Writing a field as $`u(x,y)e^{i\beta z}`$ and substituting into the scalar Helmholtz equation gives

```math
\left[\nabla_\perp^2+k_0^2n^2(x,y)\right]u(x,y)
=\beta^2u(x,y),
```

where $`\nabla_\perp^2=\partial_x^2+\partial_y^2`$ is the transverse Laplacian.

At a specified wavelength, the index profile is known. We seek the functions $`u`$ and numbers $`\beta^2`$ that satisfy the equation and the boundary conditions. These are the **eigenfunctions** and **eigenvalues** of the transverse wave operator.

For a uniform cladding, rearrange the equation as

```math
\left[
-\nabla_\perp^2-k_0^2\left(n^2-n_{\mathrm{clad}}^2\right)
\right]u
=
-\left(\beta^2-k_0^2n_{\mathrm{clad}}^2\right)u.
```

This has the mathematical structure of a Schrödinger bound-state problem. The higher-index core acts like an attractive potential well, and a bound optical mode corresponds to a negative eigenvalue in this rearranged equation. A deeper or wider well can support additional modes with more transverse structure.

The scalar equation is exact for the appropriate transverse-electric polarization of a one-dimensional isotropic slab, and useful as an approximation for some weakly guiding structures. It is **not** the general vector waveguide equation.

For nonmagnetic materials, the source-free monochromatic electric field instead satisfies

```math
\nabla\times\nabla\times\mathbf E
=
 k_0^2\boldsymbol{\varepsilon}_r\mathbf E,
```

where $`\boldsymbol{\varepsilon}_r`$ is the relative-permittivity tensor. A full-vector solver retains the coupling between field components and the material-interface conditions. After substituting the longitudinal dependence and eliminating suitable components, this too can be arranged as a transverse eigenvalue problem.

### Polarization, crystal axes, and dispersion

An isotropic material has the same refractive index for every electric-field direction. In an anisotropic crystal, the response depends on direction. When the numerical axes align with the material's principal axes, the tensor takes the diagonal form

```math
\boldsymbol{\varepsilon}_r(x,y)
=
\begin{pmatrix}
n_x^2&0&0\\
0&n_y^2&0\\
0&0&n_z^2
\end{pmatrix}.
```

These are three **material-index distributions**, not three effective indices of a mode. A vector mode samples the tensor through its field components. Arbitrarily rotated crystal axes generally require off-diagonal tensor elements and should not be represented merely by relabeling three scalar indices.

There are also two sources of wavelength dependence: the materials' indices change with wavelength, and the mode's confinement changes with wavelength. Consequently, the propagation constant must generally be recalculated at each wavelength of interest.

### How the numerical calculation works

The transverse cross-section is sampled on a grid. Spatial derivatives become finite differences; for example,

```math
\frac{\partial^2u}{\partial x^2}\bigg|_{x_j}
\approx
\frac{u_{j-1}-2u_j+u_{j+1}}{\Delta x^2}.
```

Collecting the unknown field samples into a vector $`\mathbf f`$ converts the differential equation into a matrix problem of the form

```math
A\mathbf f=\beta^2\mathbf f.
```

The matrix is sparse: most entries are zero because finite differences connect nearby grid points. Sparse eigensolvers can calculate a small set of modes near a target propagation constant without finding every eigenvector. The computational strategy used by the Python solver in gnoma is this full-vector, finite-difference approach.

The numerical window also needs boundary conditions. These are part of the computational model, not automatically the physical boundary of the device. A finite window can produce extended, boundary-dependent solutions as well as localized guided modes.

**Three checks should accompany a result:** refine the grid, enlarge the window, and inspect the field to confirm that the intended mode was found. Check convergence of the quantity needed for the application—not only the printed digits of the effective index.

A converged calculation can still describe the wrong device. Incorrect dimensions, crystal orientation, or material models are physical-model errors; a smaller grid does not correct them.

### Finding modes is different from launching light

A mode solver determines the field patterns the structure supports. It does not determine how strongly a particular input excites them.

At one frequency, a launched field can contain several guided modes:

```math
\mathbf E_\omega(x,y,z)
=
\sum_m a_m\mathbf e_m(x,y)e^{i\beta_m z}
+\text{radiation contributions}.
```

The input and interface determine the amplitudes $`a_m`$. Their phases subsequently evolve at different rates, so a superposition can change its transverse intensity pattern even though each constituent mode retains its own shape.

In a scalar, same-polarization approximation that neglects interface reflection and impedance differences, the spatial power overlap of two fields is

```math
\eta_{\mathrm{overlap}}
=
\frac{\left|\iint u_1^*u_2\,dx\,dy\right|^2}
{\left(\iint|u_1|^2\,dx\,dy\right)
 \left(\iint|u_2|^2\,dx\,dy\right)}.
```

This normalized projection explains why matching the full field shape matters: equal spot sizes alone do not ensure good coupling. Quantitative vector coupling requires the appropriate electromagnetic power normalization.

A directional coupler provides another example. Two identical nearby guides can support symmetric and antisymmetric supermodes. In the lossless two-mode approximation, a field localized in one guide is a superposition of the two. Their relative phase reaches $`\pi`$ after the complete-transfer length

```math
L_c=\frac{\pi}{|\beta_s-\beta_a|}
=\frac{\lambda}{2|n_{\mathrm{eff},s}-n_{\mathrm{eff},a}|}.
```

Thus, a propagation effect can be predicted from a pair of cross-sectional mode calculations.

### From linear modes to nonlinear frequency conversion

Nonlinear frequency conversion starts with the material's nonlinear polarization. A second-order response contains products of electric fields, schematically

```math
P_a^{(2)}\propto
\varepsilon_0\sum_{b,c}\chi_{abc}^{(2)}E_bE_c,
```

where $`\varepsilon_0`$ is the vacuum permittivity, $`\chi^{(2)}`$ is the second-order susceptibility tensor, and $`a,b,c`$ label spatial components. Products of oscillating fields produce new frequencies. **Sum-frequency generation (SFG)** combines $`\omega_1`$ and $`\omega_2`$ to produce $`\omega_3=\omega_1+\omega_2`$. **Second-harmonic generation (SHG)** is the case $`\omega_1=\omega_2=\omega`$, producing $`2\omega`$.

Two separate conditions govern efficient interaction: transverse overlap and longitudinal phase matching.

The transverse coupling contains an integral of the form

```math
\mathcal O\propto
\iint\sum_{a,b,c}
\chi_{abc}^{(2)}(x,y)
 e_{3,a}^*(x,y)e_{1,b}(x,y)e_{2,c}(x,y)
\,dx\,dy,
```

with power-normalization factors needed to obtain a physical coupling coefficient. The complex conjugate projects the nonlinear source onto the generated mode. Field signs, polarization, and the spatial distribution of the nonlinearity all matter; overlapping intensity spots alone are not enough.

Longitudinally, the driving polarization has propagation constant $`\beta_1+\beta_2`$, while the generated mode has propagation constant $`\beta_3`$. Define

```math
\Delta\beta=\beta_3-\beta_1-\beta_2.
```

In a uniform, lossless interaction of length $`L`$, assume that conversion does not appreciably deplete the input waves. Contributions from successive positions then add with relative phase $`-\Delta\beta z`$. Their integral is

```math
\int_0^L e^{-i\Delta\beta z}\,dz
=
L e^{-i\Delta\beta L/2}
\mathrm{sinc}\left(\frac{\Delta\beta L}{2}\right),
\qquad
\mathrm{sinc}(x)=\frac{\sin x}{x}.
```

The generated power therefore has a factor $`L^2\mathrm{sinc}^2(\Delta\beta L/2)`$. Phase matching makes contributions add coherently rather than cancel.

**Quasi-phase matching (QPM)** uses a periodic modulation of the nonlinear coefficient, commonly a reversal of its sign. A grating Fourier component with wavevector $`K=\pm2\pi/\Lambda`$ changes the residual mismatch to $`\Delta k=\Delta\beta-K`$. For first-order compensation of nonzero mismatch, the positive period is

```math
\Lambda=\frac{2\pi}{|\Delta\beta|}.
```

For SHG, $`\Delta\beta=\beta_{2\omega}-2\beta_\omega`$. The period is calculated from the **modal** effective indices, not simply the bulk material indices. The grating changes the longitudinal phase relation; it does not remove the need for transverse overlap or the frequency relation.

This explains the overall approach: solve the **linear** modes at the participating wavelengths, calculate their overlaps and phase mismatch, then use these quantities in an interaction model. Pump depletion, propagation loss, pulse dispersion, and longitudinal nonuniformity require the corresponding amplitude-evolution model; they do not follow from a single cross-sectional eigenmode calculation.

### References

- [Waveguide theory, mode expansion, and coupled-mode background][waveguide-theory]
- [Mode-solver background and vector-solver formulation][mode-solver-background]
- [Sparse-eigenvalue methods][sparse-eigenvalues]
- [Nonlinear-polarization background][nonlinear-crystals]
- [Coupled-wave background][nonlinear-waves]

[waveguide-theory]: https://ocw.mit.edu/courses/6-974-fundamentals-of-photonics-quantum-electronics-spring-2006/resources/chapter2/
[mode-solver-background]: https://optics.ansys.com/hc/en-us/articles/360034917233-MODE-Finite-Difference-Eigenmode-FDE-solver-introduction
[sparse-eigenvalues]: https://docs.scipy.org/doc/scipy/tutorial/arpack.html
[nonlinear-crystals]: https://byucamacholab.github.io/nonlinear-optics/pages/8.%20Nonlinear%20Optics%20in%20Crystals.html
[nonlinear-waves]: https://byucamacholab.github.io/nonlinear-optics/pages/5.%20Three-Wave%20Mixing.html

## Introduction to the code

### The main objects

The code follows the physical workflow. A **`Waveguide`** object describes the structure at a wavelength. Its **`solve()`** method returns **`Modedata`**, containing mode indices and fields. **`Qpmdata`** combines three selected modes for nonlinear-interaction calculations. In examples, `wg` means “waveguide” and `md` means “mode data.”

| Location | Role |
|---|---|
| `waveguide.py` | Waveguide models, the solve interface, mode data, and derived optical quantities |
| `zhumodes.py` | The Python full-vector finite-difference solver |
| `examples.py` | Examples using fiber, crystal, ridge, and exchanged-waveguide models |
| `waveguidetests.py` | Checks and comparisons with other calculations |

The external `wavedata` package supplies `Wave` and `Wave2D`, which keep sampled data together with their coordinates. The `sellmeier` package supplies material-index models.

### Setup, units, and coordinates

From a terminal, clone the repository and install its dependencies into the Python environment you intend to use:

```bash
git clone https://github.com/decoherer/gnoma.git
cd gnoma
python -m pip install -r requirements.txt
```

Run the following examples from the repository directory. The example selects the Python solver, so it does not require an Octave installation. The current source also initializes a machine-specific cache path in `waveguide.py`; change that path to a writable local directory if it causes an import error.

The principal conventions are:

| Quantity | Convention |
|---|---|
| Wavelength, `λ` | Vacuum wavelength in nanometers (`nm`) |
| Cross-sectional dimensions, grid spacing, and bounds | Micrometers (`µm`) |
| Refractive and effective indices | Dimensionless |
| `Qpmdata.Λ` | Signed period in micrometers; its magnitude is the positive physical period |
| Many longitudinal helpers, including coupling length | Millimeters (`mm`); check the individual method |

The numerical coordinates are $`x`$ horizontally, $`y`$ vertically, and $`z`$ along propagation. `pol='h'` selects predominantly horizontal electric fields; `pol='v'` selects predominantly vertical electric fields. These are dominant-component labels, not a guarantee that the other components vanish.

### A first waveguide calculation

Start with an artificial rectangular guide. Its indices are chosen for illustration rather than as a model of a particular material. This avoids introducing crystal orientation or fabrication parameters before the basic workflow is clear.

```python
import numpy as np
from waveguide import Boxwaveguide

wg = Boxwaveguide(
    w=3.0,                       # Core width in micrometers.
    h=2.0,                       # Core height in micrometers.
    n=1.50,                      # Core refractive index.
    n0=1.45,                     # Cladding refractive index.
    λ=1550.0,                    # Vacuum wavelength in nanometers.
    pol="v",                     # Predominantly vertical electric field.
    bounds=(-6.0, 6.0, -6.0, 6.0),  # xmin, xmax, ymin, ymax, in micrometers.
    step=0.15,                   # Grid spacing in micrometers.
)

md = wg.solve(
    solver="zhu",
    method="exact",
    boundary="neumann",
    mode=0,
    nummodes=6,
)

if md.modecount() == 0:
    raise RuntimeError("No modes of the requested polarization were retained.")

for i, neff in enumerate(md.neffs):
    print(f"Candidate {i}: effective index = {neff}")

# For this simple guide, start by inspecting the retained candidate
# with the largest real effective index. Do not rely on return order.
i0 = int(np.argmax(np.real(md.neffs)))
mode = md[i0]

print(f"Selected candidate: {i0}")
print(f"Selected effective index: {mode.neff.real:.8f}")
```

`Boxwaveguide` constructs the index profile. `solve()` searches for candidate eigenmodes and filters them by the requested polarization. `md[i0]` creates mode data containing the selected candidate alone.

`nummodes=6` requests an initial candidate count, **not six guided modes of the selected polarization**. Polarization filtering can reduce the count, and the solver can retry with more candidates. Some returned solutions may not be bound modes. The largest-index selection above is an initial choice for this simple guide; the field and convergence checks still matter.

The intended fundamental mode should be centered on the core, have a dominant field component with one main lobe, and decay into the cladding. Its effective index should lie between 1.45 and 1.50. Do not infer a precise numerical answer from these qualitative checks.

### Understanding `Modedata`

A `Modedata` object can hold several modes while also designating one selected mode.

| Expression | Meaning |
|---|---|
| `md.neffs` | Effective indices of the retained modes |
| `md.modecount()` | Number of retained modes |
| `md.modenum` | Index of the currently selected mode within that object |
| `md.neff` | Effective index of the selected mode |
| `md[i]` | A new `Modedata` object containing retained mode `i` |
| `md.Exs[i]`, `md.Eys[i]`, `md.Ezs[i]` | Electric-field components of retained mode `i` |
| `md.ee` | Real-valued convenience field for the selected polarization and mode |
| `md.ex`, `md.ey` | Horizontal and vertical line cuts through the convenience field's absolute maximum |

**`md.ex` and `md.ey` are line cuts, not the Cartesian electric-field components.** Use `Exs`, `Eys`, and `Ezs` for the component fields, particularly when their complex phase matters. Similarly, the `Hxs`, `Hys`, and `Hzs` collections contain magnetic-field data, whose scaling should be checked before using them for absolute-power calculations.

### Solver choice is different from model choice

`solver` chooses the numerical implementation. The example uses `solver='zhu'`, the Python implementation in `zhumodes.py`. The current default is `solver='octave'`, which requires a separately configured Octave-based solver.

`method` controls how the material-index model is treated:

| `method` | Treatment |
|---|---|
| `'exact'` | Retains the supplied directional-index distributions |
| `'isotropic'` | Uses the index distribution associated with the requested polarization for every direction |
| `'suppress'` | Can lower the competing transverse-polarization index to favor the requested polarization |

`'suppress'` is the default method. It changes the model; it is not simply a faster solution of the unchanged equations. Conversely, `'exact'` does not mean zero numerical error. The box example is already isotropic, and `'exact'` leaves that model unchanged.

`boundary='neumann'` makes the finite-window choice explicit. It is not an absorbing boundary or an exact representation of an infinite cladding. For a well-confined mode, moving the boundaries farther away should make their influence negligible. The Zhu implementation applies component-dependent boundary conditions.

### Changing a parameter and checking convergence

Waveguide objects are callable: supplying changed constructor arguments creates another waveguide. For example:

```python
narrower = wg(w=2.4)
finer = wg(step=0.10)
larger_window = wg(bounds=(-8.0, 8.0, -8.0, 8.0))
```

These lines construct new models; they do not solve them. Apply the same explicit `solve(...)` settings to each, and inspect the same physical mode in every result.

Changing the width changes the intended physical device. Changing the grid spacing or enlarging the numerical window should converge toward the same device prediction. Separating those two kinds of change is an essential modeling habit.

For this artificial guide, also try changing only `λ`. Because the specified material indices remain fixed, the resulting wavelength dependence isolates **waveguide dispersion**. A real-material calculation must additionally update the material indices.

Do not assume that a mode keeps the same list index during a sweep. Compare effective indices, polarization, and field patterns; near crossings, field-overlap tracking may be needed.

### Moving to material-specific models

Once the simple calculation is understood, the examples introduce `Ridgewaveguide`, `Ktpwaveguide`, and `Rpewaveguide`, among others. **KTP** is potassium titanyl phosphate. **RPE** means reverse proton exchange; **LN**, used in material descriptions, means lithium niobate. These models construct material-index profiles from geometry or fabrication parameters. The subsequent mode-solving workflow is the same.

For crystal-based models, `cut` specifies crystal directions in **vertical, horizontal, propagation** order. For example, `cut='zyx'` maps the vertical numerical direction to crystal Z, the horizontal direction to crystal Y, and propagation to crystal X. Numerical coordinates and crystal axes must not be confused.

For nonlinear calculations, select the participating mode at each wavelength before combining the results. `Qpmdata` holds three such mode-data objects; for SHG, the first two represent the same fundamental mode and the third represents the second harmonic. Its `.Λ` property evaluates the signed first-order period from their propagation constants. A period calculation alone does not establish high efficiency: inspect overlap, polarization, the nonlinear coefficient, and convergence as well.

The basic habit remains the same throughout: **define the structure, inspect its index profile, solve for candidate modes, identify the intended mode, and only then interpret derived device quantities.**
