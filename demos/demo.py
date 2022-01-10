from pathlib import Path

import dolfin
from simcardems.cli import default_parameters
from simcardems.cli import main


here = Path(__file__).parent.absolute()

args = default_parameters()

cell_init_file = args["cell_init_file"]
if not args["load_state"] and cell_init_file:
    cell_init_file = here.parent.joinpath(
        "initial_conditions",
    ).joinpath(cell_init_file)

main(**args)
time_table = dolfin.timings(dolfin.TimingClear.keep, [dolfin.TimingType.user])
print("time table = ", time_table.str(True))
with open(args["outdir"] + "_timings.log", "w+") as out:
    out.write(time_table.str(True))
