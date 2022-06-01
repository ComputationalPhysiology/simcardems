<!-- #region -->
# Release test

The release test is a test to make sure that the coupling between the mechanics and the electrophysiology is implemented correctly.

The idea is to apply a rapid length reduction of 1-2% during the middle o the activation. If the coupling is implemented correctly, this would make the active force $T_a$ drop to zero. If the coupling is not implemented correctly, the solver will most likely fail.

We can simulate a rapid length reduction by fixing one side of the slab, and control the position of the opposite side through a varying dirichlet boundary condition, i.e


```{math}
\begin{align}
\mathbf{u} &= 0,  & \mathbf{x} \in \Gamma_0 \\
\mathbf{u} &= f(t),  & \mathbf{x} \in \Gamma_1
\end{align}
```

where $\mathbf{u}$ is the displacement fields, $\Gamma_0$ is the fixed boundary and $\Gamma_1$ is the boundary where we apply a rapid length reduction. In this case we set the value of the boundary to be a time dependent function

```{math}
f(t) = \begin{cases}
0.1 & \text{if } t  < T^* \\
-0.02 l_x & \text{if } t  \geq T^*
\end{cases}
```

In other words, we pre-stretch the slap with a 10% stretch in the beginning, and when $t \geq T^*$ we change this value to be some value that depends on the length of the slab.

First we make the necessary imports
<!-- #endregion -->

```python
import logging
from pathlib import Path
import dolfin
import pulse
import simcardems
```

And we also turn off logging from pulse in order to not flow the output

```python
pulse.set_log_level(logging.WARNING)
```

Next we create a new `Runner` by subclassing the `simcardems.Runner` class.
We also make this runner take in an argument `T_release` that will be the the time when we trigger the rapid length reduction. We also store the pre-stretch variable which is the variable we will change when applying the rapid length reduction.
Also, we can only do this on a slab type geometry so we add an extra test for that.


```python
class ReleaseRunner(simcardems.Runner):
    def __init__(self, T_release, *args, **kwargs):
        self._T_release = T_release
        self.pre_stretch = kwargs.get("pre_stretch")
        assert self.pre_stretch is not None
        super().__init__(*args, **kwargs)
        # Make sure we have a SlabGeometry
        assert isinstance(self.coupling.geometry, simcardems.geometry.SlabGeometry)
        self.Lx = self.coupling.geometry.lx

    def _post_mechanics_solve(self) -> None:
        if self.t >= self._T_release:
            print("Release")
            pulse.iterate.iterate(self.mech_heart, self.pre_stretch, -0.02 * self.Lx)
        return super()._post_mechanics_solve()
```


The rapid length reduction is implemented as a part of the `_post_mechanics_solve` method which is an internal method of the `Runner` class that is called after the mechanics system is solved. Note that we also call the `_post_mechanics_solve` method of the parent class at the end (through `super`).
Now we can set up the main function


```python
def main(outdir: Path):
    T = 200
    T_release = 100
    pre_stretch = dolfin.Constant(0.1)
    runner = ReleaseRunner(outdir=outdir, T_release=T_release, pre_stretch=pre_stretch)
    runner.solve(T=T)
```


and a function for plotting the traces


```python
def postprocess(outdir: Path):
    simcardems.postprocess.plot_state_traces(outdir.joinpath("results.h5"))
```

```python
if __name__ == "__main__":
    outdir = Path("release_test_results")
    main(outdir=outdir)
    postprocess(outdir=outdir)
```

In {numref}`Figure {number} <release_test_state_traces>` we see the resulting state traces, and can also see the instant drop in the active tension ($T_a$) at the time of the triggered release.

```{figure} figures/release_test_state_traces.png
---
name: release_test_state_traces
---
Traces of the stretch ($\lambda$), the active tension ($T_a$), the membrane potential ($V$) and the intercellular calcium concentration ($Ca$) at the center of the geometry.
```

