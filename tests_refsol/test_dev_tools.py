import subprocess
import sys
from pathlib import Path

import pytest


DEV_TOOLS = Path(__file__).parents[1] / "scripts" / "dev-tools.py"


def run_test_command(tmp_path, command, test_directory, test_source):
    tests = tmp_path / test_directory
    tests.mkdir(exist_ok=True)
    (tests / "test_sample.py").write_text(test_source)
    return subprocess.run(
        [sys.executable, str(DEV_TOOLS), command, "--", "-q", "-s"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("command", "test_directory"),
    [("test", "tests"), ("test-refsol", "tests_refsol")],
)
def test_pytest_status_and_output_are_propagated(tmp_path, command, test_directory):
    passing = run_test_command(
        tmp_path,
        command,
        test_directory,
        'def test_pass():\n    print("PASS-MARKER")\n',
    )
    assert passing.returncode == 0
    assert "PASS-MARKER" in passing.stdout
    assert "1 passed" in passing.stdout

    failing = run_test_command(
        tmp_path,
        command,
        test_directory,
        'def test_fail():\n    print("FAIL-MARKER")\n    assert False\n',
    )
    assert failing.returncode == 1
    assert "FAIL-MARKER" in failing.stdout
    assert "1 failed" in failing.stdout


@pytest.mark.parametrize("command", ["test", "test-refsol"])
def test_partial_week_day_selection_fails_closed(tmp_path, command):
    result = subprocess.run(
        [sys.executable, str(DEV_TOOLS), command, "--week", "3"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Please provide both week and day" in result.stdout


@pytest.mark.parametrize("command", ["test", "test-refsol"])
@pytest.mark.parametrize(
    "selectors",
    [
        ("--week", "0"),
        ("--day", "0"),
        ("--week", "0", "--day", "0"),
        ("--week", "-1", "--day", "1"),
        ("--week", "1", "--day", "-1"),
    ],
)
def test_non_positive_week_day_selection_fails_closed(tmp_path, command, selectors):
    result = subprocess.run(
        [sys.executable, str(DEV_TOOLS), command, *selectors],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    if len(selectors) == 2:
        assert "Please provide both week and day" in result.stdout
    else:
        assert "Week and day must be positive integers" in result.stdout


def test_copy_test_missing_source_fails_closed(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(DEV_TOOLS),
            "copy-test",
            "--week",
            "99",
            "--day",
            "99",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "tests_refsol/test_week_99_day_99.py" in result.stderr


@pytest.mark.parametrize(
    "selectors",
    [
        ("--week", "0", "--day", "1"),
        ("--week", "1", "--day", "0"),
        ("--week", "-1", "--day", "1"),
    ],
)
def test_copy_test_non_positive_selection_fails_closed(tmp_path, selectors):
    result = subprocess.run(
        [sys.executable, str(DEV_TOOLS), "copy-test", *selectors],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Week and day must be positive integers" in result.stdout


def test_missing_command_fails_closed(tmp_path):
    result = subprocess.run(
        [sys.executable, str(DEV_TOOLS)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "the following arguments are required" in result.stderr
