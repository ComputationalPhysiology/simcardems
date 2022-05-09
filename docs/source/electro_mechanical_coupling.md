# Electro-Mechanical coupling

In this section we will describe the models that encompasses a cardiac electro-mechanics solver.

A cardiac electro-mechanics model contains several models that integrate the processes happening at the cellular and tissue level. There are three main components involved; a {ref}`section:mechanics`, a {ref}`section:ep` and a  {ref}`section:cell`

The way to think about this is that both the mechanics and the electrophysiology model are 3D models solved on some discretized geometry, i.e a mesh. For each node in the mesh there is a separate cell model, i.e and ODE to be solved, and the way information is passed from the cell model to the tissue model is via some state variables that are common in both. In other words for each time step, we need to recompute the value of some variables that can be obtained through the solution of the cellular ode model.

For the monodomain model in the electrophysiology model, the important variable is the membrane potential, while in the mechanics model the active tension defined in {eq}`active_stress_strain_energy` that represents the strength of the contraction is the important one. Futhermore, the cellular model also contains so called stretch-activated channels that depends on the stretch (equation {eq}`stretch`) of the tissue. Consequently, the stretch needs to be transferred back from the tissue mechanics model to the cell model. This is phenomena is called mechano-electric feedback.

## Variable transfers between models

To ensure stability of the numerical methods for solving the equations, special schemes has to be implemented. This is further discussed in {cite}`sundnes2014improved`.


## Solving the EP model and Mechanics models on different meshes
The electrophysiology model requires high spatial resolution in order to ensure convergent solutions. This is not the case for the mechanics model which can be solved on much coarser meshes whiteout any significant loss of precision. We have therefore implemented a way to solve the EP and mechanics model on different meshes, and interpolate the relevant variables between two meshes.


## References

```{bibliography}
:filter: docname in docnames
```
