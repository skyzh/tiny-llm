import argparse
import importlib
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent / "src"


def build_parser() -> argparse.ArgumentParser:
    """Define the inert Week 4 evaluation inspection commands."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect or statically grade an inert Week 4 task package without "
            "running an agent or candidate code."
        )
    )
    parser.add_argument(
        "--solution",
        choices=["tiny_llm", "tiny_llm_ref", "ref"],
        default="tiny_llm_ref",
        help="evaluation implementation to use (default: tiny_llm_ref)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser(
        "inspect",
        help="validate a package and print only its public manifest",
    )
    inspect.add_argument("package", type=Path)

    grade = commands.add_parser(
        "grade",
        help="statically grade a freshly staged, unchanged package workspace",
    )
    grade.add_argument("package", type=Path)
    return parser


def load_evaluation_api(solution: str):
    """Load course evaluation types without importing any candidate module."""

    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    package = "tiny_llm_ref" if solution == "ref" else solution
    return importlib.import_module(f"{package}.agent.evaluation")


def print_json(value) -> None:
    """Render stable machine-readable CLI output."""

    print(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True))


def inspect_package(api, package_path: Path) -> int:
    """Validate a package while keeping held-out expected values private."""

    package = api.TaskPackage.load(package_path)
    print_json(asdict(package.manifest))
    return 0


def grade_unchanged_package(api, package_path: Path) -> int:
    """Grade an untouched frozen stage using declarative checks only."""

    package = api.TaskPackage.load(package_path)
    with tempfile.TemporaryDirectory(prefix="tiny-llm-static-eval-") as temporary:
        scratch = Path(temporary)
        staged = package.stage(scratch / "stage")
        candidate = staged.freeze(scratch / "candidate")
        report = api.StaticHeldOutGrader().grade(staged, candidate)
    print_json(
        {
            "grade": asdict(report),
            "task_id": package.manifest.id,
        }
    )
    if report.status == "passed":
        return 0
    if report.status == "failed":
        return 1
    return 2


def main(argv: list[str] | None = None) -> int:
    """Run a non-executing package inspection or baseline grade."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        api = load_evaluation_api(args.solution)
        if args.command == "inspect":
            return inspect_package(api, args.package)
        if args.command == "grade":
            return grade_unchanged_package(api, args.package)
    except (AttributeError, ImportError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
