
(section:mechanics)=
# Tissue level mechanics model

The equations of motion at the tissue level can be derived from the conservations laws of linear and angular momentum. The reader is referred to {cite}`holzapfel2000nonlinear` for more details.

We represent the tissue as a non-linear, incompressible hyperelastic material that satisfies the equilibrium equation

```{math}
\nabla \cdot \mathbf{P} = 0 \;\; \mathbf{x} \in \Omega.
```

Here $\Omega$ refers to the reference configuration and $\mathbf{P}$ is the first Piola–Kirchhoff stress tensor. For hyperelastic materials the first Piola–Kirchhoff is given by

```{math}
\mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}
```

where $\Psi$ is the strain energy density function and $\mathbf{F}$ is the deformation gradient.

## Passive mechanics

We use the model from {cite}`holzapfel2009constitutive` which is given by

```{math}
:label: holzapfel_full
\begin{align}
  \begin{split}
  \Psi(I_1, I_{4_{f}},  I_{4_{s}},  I_{8_{fs}}) =& \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)\\
  +& \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4_{f}} - 1)_+^2} -1 \right) \\
  +& \frac{a_s}{2 b_s} \left( e^{ b_s (I_{4_{s}} - 1)_+^2} -1 \right)\\
  +& \frac{a_{fs}}{2 b_{fs}} \left( e^{ b_{fs} I_{8_{fs}}^2} -1 \right).
\end{split}
\end{align}
```
Here $f$ and $s$ refers to the fiber and sheet direction which in the case of a slab is taken as unit vectors in the $x$- and $y$-direction respectively. $I_1$ is the first principal invariant of the right Cauchy-Green deformation tensor $\mathbf{C} = \mathbf{F}^T \mathbf{F}$, $I_1 = \mathrm{tr} \; \mathbf{C}$, $I_{4_{a}}$ is the stretch along the $a$ direction, i.e $I_{4_{a}} = a \cdot (\mathbf{C} a)$ and $I_{8_{ab}} = a \cdot (\mathbf{C} b)$.

For more info about this model the reader is referred to the original paper. Note that the choice of material model here represents what is called constitutive relations. Constitutive relations is what makes the mechanics model specific to the heart tissue material.

One important quantity that will be relevant for the electromechanical coupling is the *stretch*. While it is common to speak about stretch along a specific direction, we will always refer to stretch as the stretch along the fiber direction $f$,

```{math}
:label: stretch
\lambda = |\mathbf{F}f| = \sqrt{(\mathbf{F}f)^T (\mathbf{F}f)}
```

## Active mechanics
We know that the heart tissue is able to contract by itself without any external loads. To model the activate contraction of the cardiomyocites we use the active stress approach where the
active contribution naturally decomposes the total stress into a sum
of passive and active stresses {cite}`nash2004electromechanical`,

```{math}
\mathbf{P} = \mathbf{P}_p + \mathbf{P}_a =  \frac{\partial \Psi_p}{\partial \mathbf{F}} +  \frac{\partial \Psi_a}{\partial \mathbf{F}}
```

Here $\Psi_p$ is the strain energy density function defined in {eq}`holzapfel_full` and

```{math}
:label: active_stress_strain_energy
\Psi_a = \frac{T_a}{2J} ( I_{4_f} - 1)
```
where $J = \mathrm{det} \; \mathbf{F}$ and $T_a$ is the active tension coming from the cellular mechanics model.

## Discretization

Incompressibility is enforced by employing a Lagrange multiplier method where one solves for both the displacement $\mathbf{u}$ and the Lagrange multiplier $p$. The equations are solved using the finite element method discretized using Taylor-hood finite elements with $\mathbb{P}_2$ elements for $\mathbf{p}$ and $\mathbb{P}_1$ elements for $p$.


## Software

We use a cardiac mechanics solver called [`pulse`](https://computationalphysiology.github.io/pulse) to declare and solve the mechanics models.


## References

```{bibliography}
:filter: docname in docnames
```
