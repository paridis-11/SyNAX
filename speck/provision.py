import argparse
import json
import os
from pathlib import Path
import secrets


def provision(directory, ann_clients=2, executors=4):
    if ann_clients < 1 or executors < 0:
        raise ValueError("Require at least one ANN client and a nonnegative executor count")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    directory.chmod(0o700)
    roster = []
    for role, count in (("ann", ann_clients), ("executor", executors)):
        for index in range(count):
            member = dict(id=f"{role}-{index + 1}", role=role, token=secrets.token_urlsafe(32))
            roster.append(member)
            path = directory / (member["id"] + ".json")
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
                json.dump(member, stream, indent=2)
    with os.fdopen(os.open(directory / "roster.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(roster, stream, indent=2)
    return roster


def main():
    parser = argparse.ArgumentParser(description="Generate per-client credentials")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--ann-clients", type=int, default=2)
    parser.add_argument("--executors", type=int, default=4)
    args = parser.parse_args()
    provision(args.directory, args.ann_clients, args.executors)
    print("Credentials created; keep roster.json on the coordinator and distribute one client file per worker.")


if __name__ == "__main__":
    main()
