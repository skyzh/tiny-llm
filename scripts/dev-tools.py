import argparse
import os
import shutil
from pathlib import Path

import pytest


def validate_week_day(args, required=False):
    week_provided = args.week is not None
    day_provided = args.day is not None
    if week_provided != day_provided or (required and not week_provided):
        print("Please provide both week and day")
        return False
    if week_provided and (args.week <= 0 or args.day <= 0):
        print("Week and day must be positive integers")
        return False
    return True


def copy_test(args, skip_if_exists=False, force=False):
    if not validate_week_day(args, required=True):
        return 1
    source_file = f"tests_refsol/test_week_{args.week}_day_{args.day}.py"
    target_file = f"tests/test_week_{args.week}_day_{args.day}.py"
    if skip_if_exists and os.path.exists(target_file) and not force:
        # diff the two files and warn if they are different
        if Path(source_file).read_text() != Path(target_file).read_text():
            print(
                f"[WARNING] {target_file} already exists and is different from {source_file}"
            )
            print(
                f"You can run `pdm run copy-test --week {args.week} --day {args.day} --force` to update it"
            )
        return 0
    print(f"copying {source_file} to {target_file}")
    shutil.copyfile(source_file, target_file)
    return 0


def test(args):
    if not validate_week_day(args):
        return 1
    if args.week is not None:
        copy_test(args, skip_if_exists=True)
        return pytest.main(
            ["-v", f"tests/test_week_{args.week}_day_{args.day}.py"] + args.remainders
        )
    return pytest.main(["-v", "tests"] + args.remainders)


def test_refsol(args):
    if not validate_week_day(args):
        return 1
    if args.week is not None:
        return pytest.main(
            ["-v", f"tests_refsol/test_week_{args.week}_day_{args.day}.py"]
            + args.remainders
        )
    return pytest.main(["-v", "tests_refsol"] + args.remainders)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    copy_test_parser = subparsers.add_parser("copy-test")
    copy_test_parser.add_argument("--week", type=int, required=True)
    copy_test_parser.add_argument("--day", type=int, required=True)
    copy_test_parser.add_argument("--force", action="store_true")
    copy_test_parser.set_defaults(copy_test_parser=True)
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--week", type=int, required=False)
    test_parser.add_argument("--day", type=int, required=False)
    test_parser.add_argument("remainders", nargs="*")
    test_parser.set_defaults(test_parser=True)
    test_refsol_parser = subparsers.add_parser("test-refsol")
    test_refsol_parser.add_argument("--week", type=int, required=False)
    test_refsol_parser.add_argument("--day", type=int, required=False)
    test_refsol_parser.add_argument("remainders", nargs="*")
    test_refsol_parser.set_defaults(test_refsol_parser=True)
    args = parser.parse_args()
    if hasattr(args, "copy_test_parser"):
        return copy_test(args, force=args.force)
    if hasattr(args, "test_parser"):
        return test(args)
    if hasattr(args, "test_refsol_parser"):
        return test_refsol(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
