# Report 8: MHD Extension — Methods and Preliminary Validation

## Extension of the wind-bubble model to MHD

The single-star wind-bubble experiment was extended from hydrodynamics to
ideal magnetohydrodynamics by adding a spatially uniform background magnetic
field. The field is oriented along the Cartesian $z$-axis,

\[
\mathbf{B}_0 = (0,0,B_0).
\]

The implementation uses the magnetic variable stored by `astronomix`,
$\mathbf{B}/\sqrt{\mu_0}$. In these units, the magnetic-pressure contribution
is

\[
P_{\mathrm{mag}} = \frac{B_0^2}{2}.
\]

The initial field strength is set through the ambient plasma beta,

\[
\beta = \frac{P_{\mathrm{gas}}}{P_{\mathrm{mag}}}
      = \frac{2P_{\mathrm{gas}}}{B_0^2},
\]

which gives

\[
B_0 = \sqrt{\frac{2P_{\mathrm{gas}}}{\beta}}.
\]

For the ambient gas pressure $P_{\mathrm{gas}}=0.01$, the two configurations
requested for the initial study are therefore

| Configuration | Plasma beta | Initial $B_z/\sqrt{\mu_0}$ | Initial magnetic-energy density |
|---|---:|---:|---:|
| Weak field | 100 | 0.0141421 | $1.0\times10^{-4}$ |
| Strong field | 1 | 0.1414214 | $1.0\times10^{-2}$ |

Omitting the plasma-beta argument retains the original hydrodynamic setup. The
wind and ambient-gas parameters otherwise remain unchanged: ambient density
$\rho_0=1$, ambient pressure $P_0=0.01$, adiabatic index
$\gamma=5/3$, mass-loss rate $\dot M=0.01$, terminal wind velocity
$v_\infty=10$, and injection radius $R_{\mathrm{inj}}=0.0625$.

## Guarding the interpretation of MHD output

The existing shock analysis is intentionally disabled whenever MHD is active.
Its Mach-number and thermalization estimates use the hydrodynamic sound speed,
gas-pressure jumps, and hydrodynamic Rankine--Hugoniot relations. An MHD shock
must instead be classified relative to the local magnetic-field orientation
and the appropriate fast, Alfvén, and slow characteristic speeds. Applying the
hydrodynamic estimator unchanged would therefore produce numbers without a
validated physical interpretation.

The present MHD stage consequently tests initialization, numerical stability,
magnetic-field evolution, and bubble morphology only. It does **not** yet
report MHD shock Mach numbers, surface-integrated thermalization rates, or
cumulative dissipated energies.

## Numerical validation diagnostics

Each MHD run records the following quantities for every saved snapshot:

- maximum absolute magnetic divergence, $\max|\nabla\!\cdot\!\mathbf{B}|$;
- minimum gas density and minimum gas pressure;
- RMS values of $B_x$, $B_y$, and $B_z$;
- mean magnetic-energy density, $\langle B^2/2\rangle$;
- median cell-wise plasma beta;
- bubble extents parallel and perpendicular to the initial field;
- the corresponding axial aspect ratios.

The divergence, positivity, field-component, and magnetic-energy histories are
written to `mhd_validation.csv` and visualized in `mhd_validation.png`.
Central density and pressure slices are saved in two orientations. The
original `central_slices.png` shows the $x$-$y$ plane perpendicular to the
background field. The additional `field_aligned_slices.png` shows the
$x$-$z$ plane containing the field direction, which is the relevant view
for detecting magnetically induced elongation or confinement.

## Bubble morphology relative to the field

Two complementary morphology diagnostics are used. First, the disturbed
bubble is defined by cells whose gas pressure exceeds the ambient value by one
percent,

\[
P > 1.01P_0.
\]

The outer extents of this mask along $z$ and in the transverse $x$-$y$
directions provide a simple geometric size estimate. The associated aspect
ratio is

\[
\mathcal{A}_{\mathrm{outer}}
  = \frac{R_{\parallel}}{R_{\perp}},
\qquad
R_{\perp}=\frac{R_x+R_y}{2}.
\]

This measure is transparent but changes only in whole-cell increments. A
pressure-excess-weighted second-moment diagnostic was therefore added to
resolve weaker, sub-cell-scale morphological differences. With

\[
w_i = \max(P_i-P_0,0),
\]

the weighted RMS extent along coordinate $j$ is

\[
R_{j,\mathrm{w}}
 = \left[
   \frac{\sum_i w_i(x_{i,j}-x_{\star,j})^2}{\sum_i w_i}
   \right]^{1/2}.
\]

The continuous field-aligned aspect ratio is then

\[
\mathcal{A}_{\mathrm{w}}
 = \frac{R_{z,\mathrm{w}}}
        {\sqrt{(R_{x,\mathrm{w}}^2+R_{y,\mathrm{w}}^2)/2}}.
\]

Values above unity indicate elongation parallel to the initial field, whereas
values below unity indicate a larger transverse extent. These measurements
are morphological diagnostics and should not be interpreted as identified MHD
shock surfaces.

## Initial 64³ stability results

Short smoke tests with three snapshots up to $t=0.01$ completed successfully
for both field strengths. Full 20-snapshot runs at $64^3$ subsequently
reached $t=0.2$. The weak-field run required 867 integration iterations, and
the strong-field run required 863. Both retained positive density and gas
pressure and produced finite snapshot states.

At the final time, the weak-field run had

- $\max|\nabla\cdot\mathbf{B}| = 2.53\times10^{-6}$;
- minimum density $=7.86\times10^{-3}$;
- minimum gas pressure $=1.00\times10^{-2}$;
- mean magnetic-energy density $=1.168\times10^{-4}$.

The strong-field run had

- $\max|\nabla\cdot\mathbf{B}| = 2.65\times10^{-5}$;
- minimum density $=7.80\times10^{-3}$;
- minimum gas pressure $=1.00\times10^{-2}$;
- mean magnetic-energy density $=1.140\times10^{-2}$.

The stronger run naturally develops a larger absolute divergence diagnostic
because its field amplitude is ten times larger. In both cases the divergence
history grows smoothly, without an abrupt numerical runaway. The transverse
field components also grow smoothly from zero as the initially vertical field
is deformed by the expanding bubble.

The cell-threshold outer-extent measure gives
$R_{\parallel}=R_{\perp}=0.3046875$ at $t=0.2$ for both runs, corresponding
to $\mathcal{A}_{\mathrm{outer}}=1$. The perpendicular and field-aligned
slices likewise appear very similar at this resolution. This result does not
establish that the magnetic field has no morphological influence: the outer
measure is quantized at the $64^3$ cell width, $\Delta x=0.015625$, and the
wind driving is strong compared with the initial magnetic pressure. The new
pressure-weighted aspect ratio must be evaluated before deciding whether a
higher-resolution comparison is warranted.

## Planned comparison

After the $\beta=100$ and $\beta=1$ cases have been rerun with the weighted
diagnostic, a CPU-only aggregation script will produce a common comparison of
magnetic divergence, fractional magnetic-energy evolution, parallel and
perpendicular weighted extents, and weighted aspect ratio. A $128^3$ study
will be started only after the (64^3) comparison has been interpreted. This
keeps the next numerical experiment tied to a measurable MHD question rather
than resolution alone.
