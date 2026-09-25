"""Move a canonical results directory (or named files) into ``_superseded/``.

``results/`` is gitignored, so nothing under it may be deleted; superseded
artefacts are moved, never overwritten.  The destination is
``<image_quality_analysis>/_superseded/<name>_<tag>/``; the call refuses to run
if it already exists, so a rerun cannot bury an earlier move.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IQ = REPO_ROOT / "results/adv_rob_eval_results/image_quality_analysis"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path, help="canonical directory under image_quality_analysis")
    parser.add_argument("--tag", default="zones_unconfined_2026-09-18")
    parser.add_argument("--files", nargs="*", default=None, help="move only these files (relative to target)")
    parser.add_argument("--keep", nargs="*", default=(), help="files to copy back after a whole-dir move")
    args = parser.parse_args()

    target = args.target if args.target.is_absolute() else IQ / args.target
    if not target.exists():
        raise SystemExit(f"nothing to supersede at {target}")
    destination = IQ / "_superseded" / f"{target.name}_{args.tag}"
    if destination.exists():
        raise SystemExit(f"refusing: {destination} already exists")

    if args.files is None:
        shutil.move(str(target), str(destination))
        target.mkdir()
        for name in args.keep:
            src = destination / name
            if src.is_dir():
                shutil.copytree(src, target / name)
            else:
                shutil.copy2(src, target / name)
        print(f"moved {target} -> {destination}; kept {list(args.keep)}")
    else:
        destination.mkdir(parents=True)
        for name in args.files:
            src = target / name
            if not src.exists():
                raise SystemExit(f"missing {src}")
            shutil.move(str(src), str(destination / name))
        print(f"moved {len(args.files)} files from {target} -> {destination}")


if __name__ == "__main__":
    main()
