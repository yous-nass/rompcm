# ROMPCM

Reduced-Order Modeling for Phase Change Materials (PCM) including pure conductive and nonlinear convective regimes. Conventional POD-Galerkin is compared with non-intrusive POD-RBF surogate model in both configurations. Data are uploaded from the papier:  [https://doi.org/10.1016/j.cpc.2020.107492].

## ROMS
```
- POD-Galerkin | conventionnal intrusive Galerkin projection method based on snapshot method
- POD-RBF interpolation | direct SVD plus RBF interpolation to ensure no-intrusive surogate ROM
- GPOD-RBF | specific only for conductive PCM which self-similiar coordinate transformation is first performed to align all snapshots at fixed solid-liquid interface
```


## Examples

```bash
from rompcm.inference import PODRBFInference
u = model.predict_time("cond")
```

## 1. Conductive PCM regime
In the conductive regime the temperature equation is governing by:

$$
\begin{aligned}
\partial_t T - \frac{1}{\mathrm{Re} \mathrm{Pr}} \Delta T + \partial_t S(T) &= 0,
\qquad \text{on }\Omega \times (0, t_{\max}) \\
T(\mathbf{x}, 0) &= -0.01, \qquad \text{on } \Omega \\
T &= 1.075, \qquad  \text{on } \partial \Gamma_l \\
\end{aligned}
$$

## 2. Convective PCM regime
In this second example, the PDE model describes the melting process of PCM governed by the Navier–Stokes–Boussinesq equations coupled with energy conservation:

$$
\nabla \cdot \mathbf{u} = 0
$$


$$
\partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p - \frac{1}{\mathrm{Re}} \nabla^2 \mathbf{u} =
\frac{\mathrm{Ra}\,T}{\mathrm{Pr} \mathrm{Re}^2} \mathbf{e}_y - \frac{C_{ck}(1-\zeta_l(T))^2}{\zeta_l(T)^3 + 10^{-6}} \mathbf{u}
$$

$$
\partial_t T+ \mathbf{u} \cdot \nabla T - \frac{1}{\mathrm{Re}\,\mathrm{Pr}} \nabla^2 T +\partial_t S(T) = 0.
$$


# Instalation
Requirements: Python, Pyfreefem
```bash
python -m venv rompcm
pip install pyfreefem
pip install -e .
pytest
```
