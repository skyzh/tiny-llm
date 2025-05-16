import argparse
import shutil
import os
import pytest


def copy_test(args, skip_if_exists=False):
    source_file = f"tests_refsol/test_week_{args.week}_day_{args.day}.py"
    target_file = f"tests/test_week_{args.week}_day_{args.day}.py"
    if skip_if_exists and os.path.exists(target_file):
        return
    print(f"copying {source_file} to {target_file}")
    shutil.copyfile(source_file, target_file)


def test(args):
    if args.week and args.day:
        copy_test(args, skip_if_exists=True)
        pytest.main(["-v", f"tests/test_week_{args.week}_day_{args.day}.py"] + args.remainders)
    else:
        pytest.main(["-v", "tests"] + args.remainders)


def test_refsol(args):
    if args.week and args.day:
        pytest.main(["-v", f"tests_refsol/test_week_{args.week}_day_{args.day}.py"] + args.remainders)
    else:
        pytest.main(["-v", "tests_refsol"] + args.remainders)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    copy_test_parser = subparsers.add_parser("copy-test")
    copy_test_parser.add_argument("--week", type=int, required=True)
    copy_test_parser.add_argument("--day", type=int, required=True)
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
        copy_test(args)
    if hasattr(args, "test_parser"):
        test(args)
    if hasattr(args, "test_refsol_parser"):
        test_refsol(args)


if __name__ == "__main__":
    main()
