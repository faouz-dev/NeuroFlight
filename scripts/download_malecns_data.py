"""Download the public MaleCNS v1.0 inputs and build NeuroFlight's local cache.

The source is the official Janelia/Google Research MaleCNS release.  The full
connectome is about 1.1 GB and is deliberately not committed to this repository.

Run:
    python -m scripts.download_malecns_data
    python -m scripts.download_malecns_data --build-cache
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome/"
)
FILES = (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
)


def download(url: str, destination: Path) -> None:
    """Download atomically; preserve an already complete-looking source file."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Exists: {destination} ({destination.stat().st_size:,} bytes)")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading: {destination.name}")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cache", action="store_true")
    args = parser.parse_args()

    data = Path("data/male_cns")
    data.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(BASE_URL + name, data / name)

    if args.build_cache:
        command = [sys.executable, "-m", "scripts.build_malecns_cache_v3"]
        subprocess.run(command, check=True)
        print("Cache built: data/male_cns/cache_v3")


if __name__ == "__main__":
    main()
