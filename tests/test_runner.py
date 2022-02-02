from simcardems import Runner


def test_runner():
    runner = Runner(lx=1, ly=1, lz=1, dx=1)
    runner.solve(0.02)
