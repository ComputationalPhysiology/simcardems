from .cli import get_parser
from .cli import main

parser = get_parser()
kwargs = vars(parser.parse_args())

main(**kwargs)
