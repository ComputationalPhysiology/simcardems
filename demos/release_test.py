# # Release test
#
# The release test is test to make sure that the coupling between the mechanics and the electrophysiology is implemented correctly
import logging
from pathlib import Path

import dolfin
import pulse
import simcardems

# Suppress logging from pulse
pulse.set_log_level(logging.WARNING)


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
        if self._t >= self._T_release:
            print("Release")
            pulse.iterate.iterate(self.mech_heart, self.pre_stretch, -0.02 * self.Lx)
        return super()._post_mechanics_solve()


def main(outdir: Path):
    T = 200
    T_release = 100
    pre_stretch = dolfin.Constant(0.1)
    runner = ReleaseRunner(outdir=outdir, T_release=T_release, pre_stretch=pre_stretch)
    runner.solve(T=T)


def postprocess(outdir: Path):
    simcardems.postprocess.plot_state_traces(outdir.joinpath("results.h5"))


if __name__ == "__main__":
    outdir = Path("release_test_results")
    main(outdir=outdir)
    postprocess(outdir=outdir)
