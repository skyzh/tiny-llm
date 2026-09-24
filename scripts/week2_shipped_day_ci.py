"""Run the reference tests that belong to the shipped Week 2 day.

The JSON manifest is the reviewable temporary gate. Day 5 removes this
selector and restores the unfiltered reference suite.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "week2_shipped_day_ci.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    included = manifest["included"]
    deferred = manifest["deferred"]
    declared = set(included) | {item["path"] for item in deferred}
    discovered = {
        str(path.relative_to(ROOT))
        for folder in ("tests_refsol", "benches")
        for path in (ROOT / folder).glob("test*.py")
    }
    if len(declared) != len(included) + len(deferred):
        raise SystemExit(
            "duplicate included/deferred test path in shipped-day manifest"
        )
    unaccounted = discovered - declared
    missing_included = set(included) - discovered
    if unaccounted or missing_included:
        raise SystemExit(
            f"unaccounted test files: {sorted(unaccounted)}; "
            f"missing included files: {sorted(missing_included)}"
        )
    if manifest["shipped_day"] != 3:
        raise SystemExit("this temporary CI gate is bound to shipped_day=3")
    print("shipped_day=3", flush=True)
    print(f"included_files={len(included)}", flush=True)
    for path in included:
        print(f"INCLUDE {path}", flush=True)
    for item in deferred:
        print(
            f"DEFER {item['path']} reenable_day={item['reenable_day']} "
            f"reason={item['reason']}",
            flush=True,
        )
    if manifest["deferred_builds"]:
        raise SystemExit("Day 3 requires both extension builds")
    if manifest["required_builds"] != ["pdm run build-ext", "pdm run build-ext-ref"]:
        raise SystemExit("Day 3 native build gates changed")
    for command in manifest["required_builds"]:
        print(f"BUILD {command}", flush=True)
    for path in manifest["retired"]:
        print(f"RETIRED {path}", flush=True)
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *included],
        cwd=ROOT,
        check=False,
        text=True,
    )
    if collect.returncode or args.collect_only:
        return collect.returncode
    return subprocess.call([sys.executable, "-m", "pytest", "-q", *included], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
