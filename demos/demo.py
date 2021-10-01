from pathlib import Path

import dolfin
from simcardems.cli import get_parser
from simcardems.cli import main


here = Path(__file__).parent.absolute()
parser = get_parser()
args = vars(parser.parse_args())

cell_init_file = args["cell_init_file"] or None
if args["reset_state"] and cell_init_file:
    cell_init_file = here.parent.joinpath(
        "initial_conditions",
    ).joinpath(cell_init_file)

main(
    args["outdir"],
    T=args["endtime"],
    T_release=args["T_release"],
    bnd_cond=args["bnd_cond"],
    add_release=args["add_release"],
    cell_init_file=cell_init_file,
    reset_state=args["reset_state"],
)
time_table = dolfin.timings(dolfin.TimingClear.keep, [dolfin.TimingType.user])
print("time table = ", time_table.str(True))
with open(args["outdir"] + "_timings.log", "w+") as out:
    out.write(time_table.str(True))
