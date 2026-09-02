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


def test_copy_and_test_missing_source_fail_closed(tmp_path):
    for command in ("copy-test", "test"):
        result = subprocess.run(
            [
                sys.executable,
                str(DEV_TOOLS),
                command,
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


def test_day_selection_refreshes_supplied_tests_without_overwriting_source(tmp_path):
    supplied = tmp_path / "tests_refsol"
    learner = tmp_path / "tests"
    supplied.mkdir()
    learner.mkdir()
    for day in range(1, 5):
        (supplied / f"test_week_4_day_{day}.py").write_text(
            f'def test_day_{day}():\n    print("FRESH-DAY-{day}")\n',
            encoding="utf-8",
        )
    (learner / "test_week_4_day_2.py").write_text(
        'def test_stale():\n    print("STALE-DAY-2")\n    assert False\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(DEV_TOOLS),
            "test",
            "--week",
            "4",
            "--day",
            "3",
            "--",
            "-q",
            "-s",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert all(f"FRESH-DAY-{day}" in result.stdout for day in range(1, 4))
    assert "STALE-DAY-2" not in result.stdout
    assert "FRESH-DAY-4" not in result.stdout
    assert "3 passed" in result.stdout
    assert not (learner / "test_week_4_day_4.py").exists()
    assert (learner / "test_week_4_day_2.py").read_bytes() == (
        supplied / "test_week_4_day_2.py"
    ).read_bytes()

    # The ordinary Week 1-3 path also refreshes a stale copied test on every run.
    # Only the selected test fixture is replaced; learner implementation stays intact.
    implementation = tmp_path / "src" / "tiny_llm" / "attention.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("LEARNER-IMPLEMENTATION\n", encoding="utf-8")
    fresh_test = 'def test_fresh():\n    print("FRESH-DAY-1")\n'
    stale_test = 'def test_stale():\n    print("STALE-DAY-1")\n    assert False\n'
    supplied_test = supplied / "test_week_1_day_1.py"
    learner_test = learner / "test_week_1_day_1.py"
    supplied_test.write_text(fresh_test, encoding="utf-8")

    for _ in range(2):
        learner_test.write_text(stale_test, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(DEV_TOOLS),
                "test",
                "--week",
                "1",
                "--day",
                "1",
                "--",
                "-q",
                "-s",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "FRESH-DAY-1" in result.stdout
        assert "STALE-DAY-1" not in result.stdout
        assert "1 passed" in result.stdout
        assert learner_test.read_text(encoding="utf-8") == fresh_test
        assert implementation.read_text(encoding="utf-8") == "LEARNER-IMPLEMENTATION\n"
