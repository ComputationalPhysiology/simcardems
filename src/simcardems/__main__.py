import json
import logging
import pprint
from pathlib import Path

from .cli import get_parser
from .cli import main

logger = logging.getLogger()


def check_json_path(path):
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find file {path}")
    if not path.suffix == ".json":
        raise ValueError("Invalid file type {path.suffix}, expected .json")


parser = get_parser()
kwargs = vars(parser.parse_args())
if kwargs["from_json"] != "":
    path = Path(kwargs["from_json"])
    check_json_path(path)
    with open(path, "r") as f:
        kwargs = json.load(f)

logger.info("Running simcardems with following parameters:")
logger.info(pprint.pformat(kwargs))
main(**kwargs)
