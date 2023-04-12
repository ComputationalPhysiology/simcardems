import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Sequence

import requests


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "folder",
        type=str,
        help="Folder where to search for json files",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("CONV_TOKEN"),
        type=str,
        help="Gist token",
    )
    parser.add_argument(
        "--gist-id",
        type=str,
        default="73fa6531f28da2b3633a7ddaca38a7cd",
        help="Gist ID",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="convergence_test.json",
        help="Filename in the gist",
    )
    args = vars(parser.parse_args(argv))
    if args["token"] is None:
        print("Please provide a valid token")
        return 1

    folder = Path(args["folder"])
    if not folder.is_dir():
        print(f"Path {folder} is not a valid folder")
        return 2

    data: Dict[str, Any] = defaultdict(dict)
    for f in folder.iterdir():
        if f.suffix != ".json":
            continue
        d = json.loads(f.read_text())
        key = f"dx{d['dx']}_dt{d['dt']}"
        sha = d["sha"]
        data[sha].update(**{key: d})

    if sha == "":
        print("No valid sha found in data")
        return 3

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {args['token']}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    print("Get existing data from gist")
    response = requests.get(
        f"https://api.github.com/gists/{args['gist_id']}",
        headers=headers,
    )
    data.update(**json.loads(response.json()["files"][args["filename"]]["content"]))

    data = {"files": {args["filename"]: {"content": json.dumps(data, indent=2)}}}
    print("Update gist")
    response = requests.patch(
        f"https://api.github.com/gists/{args['gist_id']}",
        headers=headers,
        data=json.dumps(data),
    )

    print(response.reason)

    if response.status_code != 200:
        return 0
    else:
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
