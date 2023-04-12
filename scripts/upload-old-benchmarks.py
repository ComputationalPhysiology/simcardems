import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import requests


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token",
        default=os.getenv("TOKEN"),
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

    data = {}
    for shadir in Path("benchmarks").iterdir():
        sha = shadir.name
        fname = shadir / "results.json"
        if not fname.is_file():
            continue
        d = json.loads(fname.read_text())
        if "mesh" in d:
            continue

        data[sha] = {"dx0.2_dt0.05": d}

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

    data = {"files": {args["filename"]: {"content": json.dumps(data, indent=2)}}}
    print("Update gist")
    response = requests.patch(
        f"https://api.github.com/gists/{args['gist_id']}",
        headers=headers,
        data=json.dumps(data),
    )
    print(response.reason)

    if response.status_code != 200:
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
