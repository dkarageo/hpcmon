"""Tests for :mod:`hpcmon.runner`: script building, output splitting and SSH execution.

Nothing here contacts a remote host.  Unit tests replace ``subprocess.Popen``
with a fake, and the end-to-end tests run the generated script with a *local*
``bash -s`` (through ``sh -c 'eval "$1"'``) instead of ssh.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

import pytest

import hpcmon.runner as runner
from hpcmon.runner import (
    MARKER,
    SESSION_GRACE,
    SSH_ERROR_EXIT_CODE,
    TERMINATE_GRACE,
    TIMEOUT_EXIT_CODE,
    CommandResult,
    RemoteError,
    SSHRunner,
    build_script,
    last_line,
    outside_output,
    split_output,
    terminate_all,
)

#: A local stand-in for ``ssh host``: ``argv`` ends with ``bash -s`` (or
#: ``bash -l -s``), which this evaluates, so bash reads the script from stdin.
LOCAL_SHELL = ["sh", "-c", 'eval "$1"', "_"]

needs_timeout = pytest.mark.skipif(shutil.which("timeout") is None, reason="needs coreutils timeout")


def begin(key: str) -> str:
    return f"{MARKER}BEGIN@@{key}@@"


def end(key: str, rc: int) -> str:
    return f"{MARKER}END@@{key}@@{rc}@@"


def section(key: str, body: str, rc: Optional[int]) -> str:
    """Text exactly as printed by the script for one command (stdout or stderr)."""
    text = "\n" + begin(key) + "\n" + body
    if rc is not None:
        text += "\n" + end(key, rc) + "\n"
    return text


def local_runner(timeout: float = 20) -> SSHRunner:
    return SSHRunner(LOCAL_SHELL, timeout=timeout, login_shell=False)


def run_script(script: str, env: Optional[Dict[str, str]] = None, argv0: str = "bash") -> subprocess.CompletedProcess:
    """Run a script built by :func:`build_script` with a local ``bash -s``."""
    return subprocess.run([argv0, "-s"], input=script.encode("utf-8"), capture_output=True, env=env, timeout=10)


def advance_clock_before(script: str, key: str, seconds: int) -> str:
    """Make bash's ``SECONDS`` jump forward right before command ``key`` starts.

    This simulates the passage of time instantly and deterministically.
    """
    line = f"printf '\\n%s\\n' '{begin(key)}'\n"
    assert script.count(line) == 1
    return script.replace(line, f"SECONDS=$((SECONDS + {seconds}))\n" + line)


def start_early_in_a_second() -> None:
    """Wait, if needed, until early in a wall-clock second.

    Bash's ``SECONDS`` ticks on wall-clock second boundaries, so a deadline of
    ``SECONDS + 1`` leaves between 0 and 1 s.  Starting a script early in a
    second guarantees that no boundary passes before its first command starts.
    """
    fraction = time.time() % 1
    if fraction > 0.5:
        time.sleep(1.0 - fraction + 0.01)


@pytest.fixture
def active(monkeypatch):
    """A fresh, empty set of tracked SSH processes for the test."""
    processes: set = set()
    monkeypatch.setattr(runner, "_active_processes", processes)
    return processes


# ---------------------------------------------------------------------------
# Constants and CommandResult
# ---------------------------------------------------------------------------


def test_exit_code_constants():
    assert TIMEOUT_EXIT_CODE == 124
    assert SSH_ERROR_EXIT_CODE == 255
    assert MARKER == "@@HPCMON@@"


def test_grace_periods():
    assert SESSION_GRACE == 15.0
    assert TERMINATE_GRACE == 2.0


def test_remote_error_is_an_exception():
    assert issubclass(RemoteError, Exception)


def test_command_result_defaults_to_unfinished():
    result = CommandResult()
    assert (result.stdout, result.stderr, result.returncode, result.reason) == ("", "", None, None)
    assert not result.ok
    assert result.timed_out


def test_command_result_keeps_the_reason():
    result = CommandResult("out", "err", None, "the SSH session timed out after 20 s")
    assert result.reason == "the SSH session timed out after 20 s"
    assert result != CommandResult("out", "err", None)


@pytest.mark.parametrize(
    "returncode, expected_ok, expected_timed_out",
    [
        (0, True, False),
        (1, False, False),
        (2, False, False),
        (123, False, False),
        (124, False, True),
        (125, False, False),
        (255, False, False),
        (None, False, True),
    ],
)
def test_command_result_ok_and_timed_out(returncode, expected_ok, expected_timed_out):
    result = CommandResult(stdout="x", stderr="y", returncode=returncode)
    assert result.ok is expected_ok
    assert result.timed_out is expected_timed_out


# ---------------------------------------------------------------------------
# build_script
# ---------------------------------------------------------------------------

#: The helper without a time limit.
PLAIN_HELPER = [
    "__hpcmon_run() {",
    '  bash -c "$1"',
    "}",
]


def deadline_helper(seconds: int) -> List[str]:
    """The helper (and deadline) for a time budget of ``seconds``."""
    return [
        f"__hpcmon_deadline=$((SECONDS + {seconds}))",
        "__hpcmon_run() {",
        "  __hpcmon_left=$((__hpcmon_deadline - SECONDS))",
        '  if [ "$__hpcmon_left" -le 0 ]; then',
        "    return 124",
        "  fi",
        "  if command -v timeout >/dev/null 2>&1; then",
        '    timeout "$__hpcmon_left" bash -c "$1"',
        "  else",
        '    bash -c "$1"',
        "  fi",
        "}",
    ]


def command_lines(key: str, quoted_command: str) -> List[str]:
    return [
        f"printf '\\n%s\\n' '@@HPCMON@@BEGIN@@{key}@@'",
        f"printf '\\n%s\\n' '@@HPCMON@@BEGIN@@{key}@@' >&2",
        f"__hpcmon_run {quoted_command} </dev/null",
        "__hpcmon_rc=$?",
        f"printf '\\n%s%s@@\\n' '@@HPCMON@@END@@{key}@@' \"$__hpcmon_rc\"",
        f"printf '\\n%s%s@@\\n' '@@HPCMON@@END@@{key}@@' \"$__hpcmon_rc\" >&2",
    ]


def test_build_script_exact_layout_for_two_commands_with_timeout():
    script = build_script({"first": "echo hi", "second": "squeue --me"}, command_timeout=30)
    expected = (
        deadline_helper(30)
        + command_lines("first", "'echo hi'")
        + command_lines("second", "'squeue --me'")
        + ["exit 0"]
    )
    assert script == "\n".join(expected) + "\n"


def test_build_script_exact_layout_without_timeout():
    script = build_script({"first": "echo hi", "second": "true"})
    expected = PLAIN_HELPER + command_lines("first", "'echo hi'") + command_lines("second", "true") + ["exit 0"]
    assert script == "\n".join(expected) + "\n"


def test_build_script_without_commands_only_defines_the_helper():
    assert build_script({}) == "\n".join(PLAIN_HELPER + ["exit 0"]) + "\n"
    assert build_script({}, command_timeout=10) == "\n".join(deadline_helper(10) + ["exit 0"]) + "\n"


@pytest.mark.parametrize("command_timeout", [None, math.inf, -math.inf, math.nan])
def test_build_script_without_finite_timeout_has_no_deadline(command_timeout):
    script = build_script({"k": "true"}, command_timeout=command_timeout)
    assert script == "\n".join(PLAIN_HELPER + command_lines("k", "true") + ["exit 0"]) + "\n"
    assert "deadline" not in script and "timeout" not in script


@pytest.mark.parametrize(
    "command_timeout, expected_budget",
    [
        (0, 1),
        (1e-9, 1),
        (0.2, 1),
        (-5, 1),
        (1, 1),
        (2.9, 2),
        (120.0, 120),
        (3600, 3600),
    ],
)
def test_build_script_budget_is_whole_seconds_at_least_one(command_timeout, expected_budget):
    script = build_script({"k": "true"}, command_timeout=command_timeout)
    assert script.startswith(f"__hpcmon_deadline=$((SECONDS + {expected_budget}))\n")
    # The commands themselves no longer carry a limit argument.
    assert "\n__hpcmon_run true </dev/null\n" in script


def test_build_script_begin_markers_start_on_a_new_line_on_both_streams():
    script = build_script({"k": "true"})
    assert "printf '\\n%s\\n' '@@HPCMON@@BEGIN@@k@@'\n" in script
    assert "printf '\\n%s\\n' '@@HPCMON@@BEGIN@@k@@' >&2\n" in script


def test_build_script_quotes_commands_safely():
    command = "echo \"it's $HOME\" | tr a-z A-Z; ls 'x y'"
    script = build_script({"k": command})
    assert "__hpcmon_run 'echo \"it'\"'\"'s $HOME\" | tr a-z A-Z; ls '\"'\"'x y'\"'\"'' </dev/null" in script


def test_build_script_keeps_command_order():
    script = build_script({"zeta": "true", "alpha": "true", "mid": "true"})
    positions = [script.index(begin(key)) for key in ("zeta", "alpha", "mid")]
    assert positions == sorted(positions)


@pytest.mark.parametrize("key", ["a", "A-Z_0.9", "lumi.projects", "x-y_z", "...", "123"])
def test_build_script_accepts_valid_keys(key):
    assert f"'{begin(key)}'" in build_script({key: "true"})


@pytest.mark.parametrize(
    "key",
    ["", "a b", "a;b", "a'b", 'a"b', "a/b", "a@@b", "a$b", "ä", "a\tb", "a\nb", "\na", "a|b"],
)
def test_build_script_rejects_invalid_keys(key):
    with pytest.raises(ValueError) as excinfo:
        build_script({"good": "true", key: "true"})
    assert str(excinfo.value) == f"invalid command key: {key!r}"


def test_build_script_rejects_key_with_trailing_newline():
    with pytest.raises(ValueError, match="invalid command key"):
        build_script({"a\n": "true"}, command_timeout=5)


# ---------------------------------------------------------------------------
# The generated script, run by a local bash
# ---------------------------------------------------------------------------


def test_script_without_timeout_runs_every_command():
    script = build_script({"a": "echo A; echo a-err >&2; exit 3", "b": "echo B"})
    completed = run_script(script)
    assert completed.returncode == 0
    assert split_output(completed.stdout.decode()) == {"a": ("A", 3), "b": ("B", 0)}
    assert split_output(completed.stderr.decode()) == {"a": ("a-err", 3), "b": ("", 0)}


@needs_timeout
def test_script_deadline_stops_the_overrunning_command_and_skips_the_rest():
    script = build_script(
        {
            "slow": "echo started; sleep 5; echo never",
            "after": "echo after; echo after-err >&2",
            "last": "echo last",
        },
        command_timeout=1,
    )
    start_early_in_a_second()
    start = time.monotonic()
    completed = run_script(script)
    elapsed = time.monotonic() - start

    assert completed.returncode == 0
    # ``timeout`` stops the slow command (and its ``sleep``) when the budget
    # runs out; the later commands are skipped without running.
    assert 0.9 <= elapsed < 3.5
    stdout = split_output(completed.stdout.decode("utf-8"))
    stderr = split_output(completed.stderr.decode("utf-8"))
    assert stdout == {"slow": ("started", TIMEOUT_EXIT_CODE), "after": ("", TIMEOUT_EXIT_CODE), "last": ("", 124)}
    assert stderr == {"slow": ("", TIMEOUT_EXIT_CODE), "after": ("", TIMEOUT_EXIT_CODE), "last": ("", 124)}
    slow = CommandResult(*stdout["slow"])
    assert slow.timed_out and not slow.ok


def test_script_passes_the_remaining_budget_to_timeout(tmp_path):
    # A fake ``timeout`` that logs its arguments and runs the command.
    fake = tmp_path / "timeout"
    fake.write_text('#!/bin/sh\nprintf "timeout %s\\n" "$*" >&2\nshift\nexec "$@"\n')
    fake.chmod(0o755)
    script = build_script({"first": "echo one", "second": "echo two", "third": "echo three"}, command_timeout=30)
    script = advance_clock_before(script, "second", 10)
    script = advance_clock_before(script, "third", 25)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    completed = run_script(script, env=env)

    assert completed.returncode == 0
    stdout = split_output(completed.stdout.decode())
    stderr = split_output(completed.stderr.decode())
    assert stdout == {"first": ("one", 0), "second": ("two", 0), "third": ("", 124)}
    # SECONDS may tick once between the deadline and the call.
    assert re.fullmatch(r"timeout (29|30) bash -c echo one", stderr["first"][0])
    assert re.fullmatch(r"timeout (19|20) bash -c echo two", stderr["second"][0])
    # The exhausted budget skips the command: ``timeout`` is never called.
    assert stderr["third"] == ("", 124)


def bash_only_path(tmp_path) -> str:
    """A directory holding only ``bash``, so ``command -v timeout`` fails."""
    (tmp_path / "bash").symlink_to(shutil.which("bash"))
    return str(tmp_path)


def test_script_falls_back_to_plain_bash_without_timeout_binary(tmp_path):
    # Only ``bash`` is on PATH, so ``command -v timeout`` fails and the helper
    # runs the command with plain ``bash -c`` (calling ``timeout`` would give 127).
    path = bash_only_path(tmp_path)
    script = build_script({"k": "echo direct; exit 4"}, command_timeout=5)
    completed = run_script(script, env={"PATH": path}, argv0=str(tmp_path / "bash"))
    assert completed.returncode == 0
    assert split_output(completed.stdout.decode("utf-8")) == {"k": ("direct", 4)}
    assert split_output(completed.stderr.decode("utf-8")) == {"k": ("", 4)}


def test_script_without_timeout_binary_still_skips_after_the_deadline(tmp_path):
    path = bash_only_path(tmp_path)
    script = build_script({"first": "echo one", "second": "echo two", "third": "echo three"}, command_timeout=5)
    script = advance_clock_before(script, "second", 2)
    script = advance_clock_before(script, "third", 3)
    completed = run_script(script, env={"PATH": path}, argv0=str(tmp_path / "bash"))
    assert completed.returncode == 0
    # 2 s are spent before "second" (3 s left), 5 s before "third" (none left).
    assert split_output(completed.stdout.decode()) == {"first": ("one", 0), "second": ("two", 0), "third": ("", 124)}


def test_script_with_non_finite_timeout_never_skips():
    script = advance_clock_before(build_script({"a": "echo A", "b": "echo B"}, command_timeout=math.inf), "b", 10**6)
    completed = run_script(script)
    assert split_output(completed.stdout.decode()) == {"a": ("A", 0), "b": ("B", 0)}


# ---------------------------------------------------------------------------
# split_output
# ---------------------------------------------------------------------------


def test_split_output_of_empty_text():
    assert split_output("") == {}


def test_split_output_two_sections():
    text = section("a", "line 1\nline 2\n", 0) + section("b", "", 3)
    assert split_output(text) == {"a": ("line 1\nline 2", 0), "b": ("", 3)}


def test_split_output_ignores_text_outside_markers():
    text = (
        "Welcome to the cluster!\n"
        "Last login: yesterday\n"
        + section("a", "inside\n", 0)
        + "stray text between sections\n"
        + section("b", "also inside\n", 1)
        + "trailing junk\n"
    )
    assert split_output(text) == {"a": ("inside", 0), "b": ("also inside", 1)}


def test_split_output_only_text_without_markers():
    assert split_output("banner\nmore banner\n") == {}


def test_split_output_begin_without_end_has_no_returncode():
    text = section("a", "done\n", 0) + section("b", "partial 1\npartial 2", None)
    assert split_output(text) == {"a": ("done", 0), "b": ("partial 1\npartial 2", None)}


def test_split_output_unterminated_section_keeps_trailing_blank_line():
    # Only the end-marker path drops the guard blank line.
    assert split_output(begin("a") + "\nx\n\n") == {"a": ("x\n", None)}


def test_split_output_begin_while_section_open_closes_it_unfinished():
    text = begin("a") + "\nfirst\n\n" + section("b", "second\n", 0)
    assert split_output(text) == {"a": ("first\n\n", None), "b": ("second", 0)}


def test_split_output_end_for_another_key_is_content():
    text = begin("a") + "\nbefore\n" + end("b", 5) + "\nafter\n\n" + end("a", 0) + "\n"
    assert split_output(text) == {"a": ("before\n" + end("b", 5) + "\nafter", 0)}


def test_split_output_end_without_open_section_is_ignored():
    text = end("a", 0) + "\n" + section("b", "x\n", 2) + end("b", 7) + "\n"
    assert split_output(text) == {"b": ("x", 2)}


def test_split_output_end_directly_after_begin():
    assert split_output(begin("a") + "\n" + end("a", 0) + "\n") == {"a": ("", 0)}


@pytest.mark.parametrize(
    "body, expected",
    [
        ("", ""),  # empty output: only the guard blank line
        ("x", "x"),  # no trailing newline: no guard line to remove
        ("x\n", "x"),  # usual case: the guard blank line is removed
        ("x\n\n", "x\n"),  # a genuine trailing blank line is kept
        ("x\n\n\n", "x\n\n"),
        ("\n\nx\n", "\n\nx"),  # leading blank lines are kept
        ("a\n\nb\n", "a\n\nb"),  # internal blank lines are kept
    ],
)
def test_split_output_removes_only_the_guard_blank_line(body, expected):
    # The script prints "\n" + END marker after the command's own output.
    text = "\n" + begin("k") + "\n" + body + "\n" + end("k", 0) + "\n"
    assert split_output(text) == {"k": (expected, 0)}


@pytest.mark.parametrize("rc", [0, 1, 124, 127, 255, -1])
def test_split_output_parses_return_codes(rc):
    assert split_output(section("k", "out\n", rc)) == {"k": ("out", rc)}


@pytest.mark.parametrize(
    "line",
    [
        f"{MARKER}END@@k@@x@@",  # non-numeric rc
        f"{MARKER}END@@k@@1",  # missing trailing @@
        f" {MARKER}END@@k@@0@@",  # indented
        f"{MARKER}END@@k@@0@@ ",  # trailing space
        f"text {MARKER}END@@k@@0@@",  # marker in the middle of a line
        f"{MARKER}END@@k@@@@",  # empty rc
    ],
)
def test_split_output_malformed_end_marker_is_content(line):
    text = begin("k") + "\nout\n" + line + "\n"
    assert split_output(text) == {"k": ("out\n" + line, None)}


@pytest.mark.parametrize(
    "line",
    [
        f" {MARKER}BEGIN@@other@@",
        f"{MARKER}BEGIN@@other@@x",
        f"{MARKER}BEGIN@@bad key@@",
        f"{MARKER}BEGIN@@@@",
        f"prefix{MARKER}BEGIN@@other@@",
    ],
)
def test_split_output_malformed_begin_marker_is_content(line):
    text = begin("k") + "\n" + line + "\n\n" + end("k", 0) + "\n"
    assert split_output(text) == {"k": (line, 0)}


def test_split_output_malformed_begin_marker_outside_section_is_ignored():
    assert split_output(f"{MARKER}BEGIN@@bad key@@\nx\n" + end("bad key", 0) + "\n") == {}


def test_split_output_repeated_key_keeps_last_section():
    text = section("k", "one\n", 1) + section("k", "two\n", 0)
    assert split_output(text) == {"k": ("two", 0)}


def test_split_output_handles_crlf_line_endings():
    text = section("a", "x\n", 0).replace("\n", "\r\n")
    assert split_output(text) == {"a": ("x", 0)}


def test_split_output_banner_without_trailing_newline_does_not_hide_the_marker():
    # The BEGIN marker is printed after a newline, so it starts its own line.
    assert split_output("Welcome" + section("a", "x\n", 0)) == {"a": ("x", 0)}


def test_split_output_keeps_pipes_and_brackets_in_content():
    body = "123|standard-g|project_465|RUNNING|1:00|2:00|1|nid[001-002]|my|job [x]\n"
    assert split_output(section("squeue", body, 0)) == {"squeue": (body.rstrip("\n"), 0)}


def test_split_output_roundtrips_a_real_fixture():
    from tests.conftest import read_fixture

    fixture = read_fixture("slurm/squeue.txt")
    parsed = split_output("banner\n" + section("squeue", fixture, 0))
    assert parsed == {"squeue": (fixture[:-1] if fixture.endswith("\n") else fixture, 0)}


# ---------------------------------------------------------------------------
# outside_output and last_line
# ---------------------------------------------------------------------------


def test_outside_output_of_empty_text():
    assert outside_output("") == ""


def test_outside_output_keeps_only_lines_outside_markers():
    text = (
        "Welcome to the cluster!\n"
        + begin("a")
        + "\ninside a\n"
        + end("a", 0)
        + "\nbetween\n"
        + begin("b")
        + "\ninside b\n\n"
        + end("b", 1)
        + "\nConnection closed.\n"
    )
    assert outside_output(text) == "Welcome to the cluster!\nbetween\nConnection closed."


def test_outside_output_without_markers_is_the_whole_text():
    assert outside_output("one\n\ntwo\n") == "one\n\ntwo"


def test_outside_output_after_an_unterminated_begin_is_inside():
    text = "banner\n" + begin("a") + "\npartial\nssh: Connection closed by remote host.\n"
    assert outside_output(text) == "banner"


def test_outside_output_end_without_begin_is_dropped():
    assert outside_output("x\n" + end("a", 0) + "\ny\n") == "x\ny"


def test_outside_output_malformed_markers_are_ordinary_lines():
    text = f" {MARKER}BEGIN@@a@@\ninside?\n{MARKER}END@@a@@x@@\n"
    assert outside_output(text) == text.rstrip("\n")


def test_outside_output_handles_crlf():
    assert outside_output(("out\n" + begin("a") + "\nin\n" + end("a", 0) + "\nafter\n").replace("\n", "\r\n")) == (
        "out\nafter"
    )


@pytest.mark.parametrize(
    "texts, expected",
    [
        ((), None),
        (("",), None),
        (("  \n\t\n",), None),
        (("", "  ", "\n"), None),
        (("only",), "only"),
        (("a\nb\n\n  \n",), "b"),
        (("  padded line \t\n",), "padded line"),
        (("err 1\nerr 2\n", "out\n"), "err 2"),
        (("", "out 1\nout 2"), "out 2"),
        ((" \n", "\n", "third"), "third"),
        (("a\r\nb\r\n",), "b"),
    ],
)
def test_last_line(texts, expected):
    assert last_line(*texts) == expected


# ---------------------------------------------------------------------------
# SSHRunner construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ssh_command", [[], (), ""])
def test_ssh_runner_rejects_empty_command(ssh_command):
    with pytest.raises(ValueError, match="^the SSH command must not be empty$"):
        SSHRunner(ssh_command)


def test_ssh_runner_defaults():
    ssh = SSHRunner(("ssh", "lumi"))
    assert ssh.ssh_command == ["ssh", "lumi"]
    assert ssh.timeout == 120.0
    assert ssh.login_shell is True


def test_ssh_runner_copies_the_command():
    command = ["ssh", "lumi"]
    ssh = SSHRunner(command)
    command.append("-v")
    assert ssh.ssh_command == ["ssh", "lumi"]


@pytest.mark.parametrize(
    "login_shell, expected",
    [
        (True, ["ssh", "-o", "BatchMode=yes", "lumi", "bash -l -s"]),
        (False, ["ssh", "-o", "BatchMode=yes", "lumi", "bash -s"]),
    ],
)
def test_ssh_runner_argv(login_shell, expected):
    ssh = SSHRunner(["ssh", "-o", "BatchMode=yes", "lumi"], timeout=5, login_shell=login_shell)
    assert ssh.argv == expected
    # The property returns a fresh list every time.
    ssh.argv.append("junk")
    assert ssh.argv == expected


# ---------------------------------------------------------------------------
# SSHRunner.run with a fake Popen
# ---------------------------------------------------------------------------


class FakePopen:
    """Records how it is used and writes canned bytes into the given files.

    :param timeout: Make ``communicate`` raise :class:`subprocess.TimeoutExpired`.
    :param stubborn: Number of ``wait`` calls that time out (ignored SIGTERM).
    :param interrupt: Make ``communicate`` raise :class:`KeyboardInterrupt`.
    :param during: Called (with the fake) from within ``communicate``.
    """

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        timeout: bool = False,
        stubborn: int = 0,
        interrupt: bool = False,
        during: Optional[Callable[["FakePopen"], None]] = None,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._timeout = timeout
        self._stubborn = stubborn
        self._interrupt = interrupt
        self._during = during
        self.returncode: Optional[int] = None
        self.events: List[str] = []
        self.wait_timeouts: List[Optional[float]] = []
        self.argv: Optional[List[str]] = None
        self.kwargs: Dict[str, object] = {}
        self.communicate_kwargs: Dict[str, object] = {}
        self.tracked_during_communicate: Optional[bool] = None

    def __call__(self, argv, **kwargs):
        self.argv = list(argv)
        self.kwargs = kwargs
        self.events.append("popen")
        return self

    def communicate(self, input=None, timeout=None):
        self.events.append("communicate")
        self.communicate_kwargs = {"input": input, "timeout": timeout}
        self.tracked_during_communicate = self in runner._active_processes
        self.kwargs["stdout"].write(self._stdout)
        self.kwargs["stderr"].write(self._stderr)
        if self._during is not None:
            self._during(self)
        if self._interrupt:
            raise KeyboardInterrupt
        if self._timeout:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        if self.returncode is None:
            self.returncode = self._returncode
        return None, None

    def terminate(self):
        self.events.append("terminate")

    def kill(self):
        self.events.append("kill")

    def wait(self, timeout=None):
        self.events.append("wait")
        self.wait_timeouts.append(timeout)
        if self._stubborn > 0:
            self._stubborn -= 1
            raise subprocess.TimeoutExpired(self.argv, timeout)
        if self.returncode is None:
            self.returncode = -9 if "kill" in self.events else -15
        return self.returncode


def install(monkeypatch, fake) -> None:
    monkeypatch.setattr(runner.subprocess, "Popen", fake)


def fake_output(sections: Dict[str, "tuple[str, Optional[int]]"], banner: str = "", trailer: str = "") -> bytes:
    text = banner + "".join(section(key, body, rc) for key, (body, rc) in sections.items()) + trailer
    return text.encode("utf-8")


def test_run_without_commands_does_not_spawn(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Popen must not be called")

    install(monkeypatch, forbidden)
    assert SSHRunner(["ssh", "lumi"]).run({}) == {}


def test_run_rejects_invalid_keys_before_spawning(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Popen must not be called")

    install(monkeypatch, forbidden)
    with pytest.raises(ValueError, match="invalid command key"):
        SSHRunner(["ssh", "lumi"]).run({"bad key": "true"})


def test_run_passes_script_argv_and_session_limit_to_popen(monkeypatch, active):
    commands = {"a": "echo hi", "b": "false"}
    fake = FakePopen(
        stdout=fake_output({"a": ("hi\n", 0), "b": ("", 1)}),
        stderr=fake_output({"a": ("", 0), "b": ("oops\n", 1)}),
    )
    install(monkeypatch, fake)
    results = SSHRunner(["ssh", "lumi"], timeout=42.5).run(commands)

    assert fake.argv == ["ssh", "lumi", "bash -l -s"]
    assert fake.kwargs["stdin"] is subprocess.PIPE
    assert hasattr(fake.kwargs["stdout"], "write") and hasattr(fake.kwargs["stderr"], "write")
    assert fake.kwargs["stdout"] is not fake.kwargs["stderr"]
    # The commands share the time limit; the session gets SESSION_GRACE more.
    assert fake.communicate_kwargs == {
        "input": build_script(commands, command_timeout=42.5).encode("utf-8"),
        "timeout": 42.5 + 15.0,
    }
    assert fake.events == ["popen", "communicate"]
    assert results == {
        "a": CommandResult(stdout="hi", stderr="", returncode=0),
        "b": CommandResult(stdout="", stderr="oops", returncode=1),
    }
    assert all(result.reason is None for result in results.values())


def test_run_session_limit_follows_session_grace(monkeypatch):
    monkeypatch.setattr(runner, "SESSION_GRACE", 3.0)
    fake = FakePopen(stdout=fake_output({"a": ("", 0)}))
    install(monkeypatch, fake)
    SSHRunner(["ssh", "lumi"], timeout=7).run({"a": "x"})
    assert fake.communicate_kwargs["timeout"] == 10.0


def test_run_tracks_the_process_only_while_it_runs(monkeypatch, active):
    fake = FakePopen(stdout=fake_output({"a": ("", 0)}))
    install(monkeypatch, fake)
    SSHRunner(["ssh", "lumi"]).run({"a": "x"})
    assert fake.tracked_during_communicate is True
    assert active == set()


def test_run_stops_tracking_after_a_failure(monkeypatch, active):
    fake = FakePopen(stdout=b"", stderr=b"boom\n", returncode=255)
    install(monkeypatch, fake)
    with pytest.raises(RemoteError):
        SSHRunner(["ssh", "lumi"]).run({"a": "x"})
    assert fake.tracked_during_communicate is True
    assert active == set()


def test_run_stops_tracking_after_a_timeout(monkeypatch, active):
    fake = FakePopen(stdout=fake_output({"a": ("", None)}), timeout=True)
    install(monkeypatch, fake)
    SSHRunner(["ssh", "lumi"]).run({"a": "x"})
    assert fake.tracked_during_communicate is True
    assert active == set()


def test_run_ignores_login_banner(monkeypatch):
    fake = FakePopen(
        stdout=fake_output({"a": ("out\n", 0)}, banner="Welcome to LUMI\n\n"),
        stderr=fake_output({"a": ("err\n", 0)}, banner="Lmod: loading defaults\n"),
    )
    install(monkeypatch, fake)
    assert SSHRunner(["ssh", "lumi"]).run({"a": "x"}) == {"a": CommandResult("out", "err", 0)}


def test_run_returns_only_requested_keys_in_request_order(monkeypatch):
    fake = FakePopen(stdout=fake_output({"b": ("B\n", 0), "extra": ("?\n", 0), "a": ("A\n", 2)}))
    install(monkeypatch, fake)
    results = SSHRunner(["ssh", "lumi"]).run({"a": "x", "b": "y"})
    assert list(results) == ["a", "b"]
    assert results == {"a": CommandResult("A", "", 2), "b": CommandResult("B", "", 0)}


def test_run_missing_keys_have_no_returncode_and_a_reason(monkeypatch):
    fake = FakePopen(stdout=fake_output({"a": ("A\n", 0)}), stderr=b"")
    install(monkeypatch, fake)
    results = SSHRunner(["ssh", "lumi"]).run({"a": "x", "b": "y", "c": "z"})
    reason = "the SSH session ended unexpectedly (exit status 0)"
    assert results == {
        "a": CommandResult("A", "", 0),
        "b": CommandResult("", "", None, reason),
        "c": CommandResult("", "", None, reason),
    }
    assert not results["b"].ok and results["b"].timed_out


def test_run_decodes_invalid_utf8_with_replacement(monkeypatch):
    stdout = begin("a").encode() + b"\ncaf\xc3\xa9 \xff\n\n" + end("a", 0).encode() + b"\n"
    install(monkeypatch, FakePopen(stdout=stdout))
    assert SSHRunner(["ssh", "lumi"]).run({"a": "x"})["a"].stdout == "café �"


def test_run_session_timeout_with_partial_output(monkeypatch):
    commands = {"a": "quick", "b": "slow", "c": "never"}
    fake = FakePopen(
        stdout=fake_output({"a": ("A\n", 0), "b": ("partial", None)}),
        stderr=fake_output({"a": ("", 0), "b": ("warn", None)}, banner="ssh: some notice\n"),
        timeout=True,
    )
    install(monkeypatch, fake)
    results = SSHRunner(["ssh", "lumi"], timeout=7).run(commands)

    # SIGTERM is enough: no SIGKILL.
    assert fake.events == ["popen", "communicate", "terminate", "wait"]
    assert fake.wait_timeouts == [2.0]
    reason = "the SSH session timed out after 22 s"
    assert results == {
        "a": CommandResult("A", "", 0),
        "b": CommandResult("partial", "warn", None, reason),
        "c": CommandResult("", "", None, reason),
    }
    assert results["b"].timed_out and results["c"].timed_out
    assert not results["a"].timed_out


def test_run_session_timeout_kills_a_session_ignoring_sigterm(monkeypatch):
    fake = FakePopen(stdout=fake_output({"a": ("", None)}), timeout=True, stubborn=1)
    install(monkeypatch, fake)
    results = SSHRunner(["ssh", "lumi"], timeout=0.5).run({"a": "x"})
    assert fake.events == ["popen", "communicate", "terminate", "wait", "kill", "wait"]
    assert fake.wait_timeouts == [2.0, None]
    assert results == {"a": CommandResult("", "", None, "the SSH session timed out after 15.5 s")}


def test_run_finished_commands_have_no_reason_even_after_a_timeout(monkeypatch):
    fake = FakePopen(stdout=fake_output({"a": ("", 124), "b": ("x\n", 3)}), timeout=True)
    install(monkeypatch, fake)
    results = SSHRunner(["ssh", "lumi"]).run({"a": "x", "b": "y"})
    assert results == {"a": CommandResult("", "", 124), "b": CommandResult("x", "", 3)}
    assert results["a"].timed_out and results["a"].reason is None


@pytest.mark.parametrize(
    "timeout, expected",
    [(120.0, "no response within 135 s"), (0.5, "no response within 15.5 s"), (3600, "no response within 3615 s")],
)
def test_run_session_timeout_without_output_raises(monkeypatch, timeout, expected):
    fake = FakePopen(stdout=b"banner only\n", stderr=b"ssh: connecting...\n", timeout=True)
    install(monkeypatch, fake)
    with pytest.raises(RemoteError) as excinfo:
        SSHRunner(["ssh", "lumi"], timeout=timeout).run({"a": "x"})
    assert str(excinfo.value) == expected
    assert fake.events == ["popen", "communicate", "terminate", "wait"]


@pytest.mark.parametrize(
    "stdout, stderr, expected",
    [
        (
            b"",
            b"ssh: Could not resolve hostname lumi: Name or service not known\n",
            "SSH connection failed: ssh: Could not resolve hostname lumi: Name or service not known",
        ),
        (
            b"",
            b"Warning: Permanently added host\nuser@host: Permission denied (publickey).\n\n  \n",
            "SSH connection failed: user@host: Permission denied (publickey).",
        ),
        (b"", b"", "SSH connection failed: exit status 255"),
        (b"", b" \n\t\n", "SSH connection failed: exit status 255"),
        # Only stderr is used for connection failures.
        (b"some banner\n", b"", "SSH connection failed: exit status 255"),
    ],
)
def test_run_connection_failure_without_markers(monkeypatch, stdout, stderr, expected):
    install(monkeypatch, FakePopen(stdout=stdout, stderr=stderr, returncode=255))
    with pytest.raises(RemoteError) as excinfo:
        SSHRunner(["ssh", "lumi"]).run({"a": "x"})
    assert str(excinfo.value) == expected


@pytest.mark.parametrize(
    "stdout, stderr, returncode, expected",
    [
        (b"", b"", 0, "the remote shell exited with status 0 before running any command"),
        (b"", b"  \n\t\n", 1, "the remote shell exited with status 1 before running any command"),
        (
            b"",
            b"bash: line 1: syntax error near unexpected token\n\n",
            2,
            "the remote shell exited with status 2 before running any command: "
            "bash: line 1: syntax error near unexpected token",
        ),
        (
            b"Welcome\nThis account is locked.\n\n",
            b"",
            1,
            "the remote shell exited with status 1 before running any command: This account is locked.",
        ),
        (
            b"stdout line\n",
            b"stderr line\n",
            126,
            "the remote shell exited with status 126 before running any command: stderr line",
        ),
        (b"", b"", -15, "the remote shell exited with status -15 before running any command"),
        (b"", b"", 254, "the remote shell exited with status 254 before running any command"),
    ],
)
def test_run_shell_exit_without_markers(monkeypatch, stdout, stderr, returncode, expected):
    install(monkeypatch, FakePopen(stdout=stdout, stderr=stderr, returncode=returncode))
    with pytest.raises(RemoteError) as excinfo:
        SSHRunner(["ssh", "lumi"]).run({"a": "x"})
    assert str(excinfo.value) == expected


def test_run_markers_only_on_stderr_still_fail(monkeypatch):
    # Without markers on stdout nothing ran; the detail is stderr's last line.
    install(monkeypatch, FakePopen(stdout=b"", stderr=fake_output({"a": ("err\n", 0)}), returncode=0))
    with pytest.raises(RemoteError) as excinfo:
        SSHRunner(["ssh", "lumi"]).run({"a": "x"})
    assert str(excinfo.value) == (
        "the remote shell exited with status 0 before running any command: " + end("a", 0)
    )


@pytest.mark.parametrize(
    "stderr, returncode, expected_detail",
    [
        # ssh's message after the last completed command.
        (
            fake_output({"a": ("", 0)}, trailer="Connection to lumi closed by remote host.\n"),
            255,
            "Connection to lumi closed by remote host.",
        ),
        # The last of several lines outside the markers.
        (
            fake_output({"a": ("", 0)}, banner="Lmod: notice\n", trailer="first\nsecond\n\n"),
            255,
            "second",
        ),
        # A login banner on stderr is outside the markers too.
        (fake_output({"a": ("", 0)}, banner="Lmod: loading defaults\n"), 1, "Lmod: loading defaults"),
        # Output of the commands is not outside the markers.
        (fake_output({"a": ("inside a\n", 0)}), 255, "exit status 255"),
        # Nor is text after a BEGIN without END.
        (fake_output({"a": ("", 0), "b": ("killed\nssh: closed", None)}), 255, "exit status 255"),
        (fake_output({"a": ("", 0)}, trailer="   \n"), -9, "exit status -9"),
        (b"", 0, "exit status 0"),
    ],
)
def test_run_reason_when_the_session_ends_early(monkeypatch, stderr, returncode, expected_detail):
    fake = FakePopen(stdout=fake_output({"a": ("A\n", 0), "b": ("partial\n", None)}), stderr=stderr, returncode=returncode)
    install(monkeypatch, fake)
    results = SSHRunner(["ssh", "lumi"]).run({"a": "x", "b": "y", "c": "z"})
    reason = f"the SSH session ended unexpectedly ({expected_detail})"
    assert results["a"].stdout == "A" and results["a"].returncode == 0 and results["a"].reason is None
    assert (results["b"].stdout, results["b"].returncode, results["b"].reason) == ("partial", None, reason)
    assert results["c"] == CommandResult("", "", None, reason)


@pytest.mark.parametrize("error", [FileNotFoundError(2, "No such file"), PermissionError(13, "denied"), OSError("boom")])
def test_run_popen_oserror_becomes_remote_error(monkeypatch, active, error):
    def failing(*args, **kwargs):
        raise error

    install(monkeypatch, failing)
    with pytest.raises(RemoteError) as excinfo:
        SSHRunner(["prog", "host"]).run({"a": "x"})
    assert str(excinfo.value) == f"cannot run \"prog host 'bash -l -s'\": {error}"
    assert excinfo.value.__cause__ is error
    assert active == set()


def test_run_missing_binary_raises_remote_error():
    ssh = SSHRunner(["/nonexistent/hpcmon-no-such-binary", "host"], login_shell=False)
    with pytest.raises(RemoteError) as excinfo:
        ssh.run({"a": "true"})
    message = str(excinfo.value)
    assert message.startswith("cannot run \"/nonexistent/hpcmon-no-such-binary host 'bash -s'\": ")
    assert "No such file or directory" in message
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


# ---------------------------------------------------------------------------
# Stopping sessions: _stop and terminate_all
# ---------------------------------------------------------------------------


def test_stop_terminates_and_waits_with_a_grace_period():
    fake = FakePopen()
    runner._stop(fake)
    assert fake.events == ["terminate", "wait"]
    assert fake.wait_timeouts == [TERMINATE_GRACE]
    assert fake.returncode == -15


def test_stop_kills_when_terminate_is_not_enough():
    fake = FakePopen(stubborn=1)
    runner._stop(fake)
    assert fake.events == ["terminate", "wait", "kill", "wait"]
    assert fake.wait_timeouts == [2.0, None]
    assert fake.returncode == -9


def test_stop_uses_the_current_grace_period(monkeypatch):
    monkeypatch.setattr(runner, "TERMINATE_GRACE", 0.25)
    fake = FakePopen()
    runner._stop(fake)
    assert fake.wait_timeouts == [0.25]


def test_stop_terminates_a_real_process():
    process = subprocess.Popen(["sleep", "30"])
    start = time.monotonic()
    runner._stop(process)
    assert process.returncode == -signal.SIGTERM
    assert time.monotonic() - start < 2


def test_stop_kills_a_real_process_that_ignores_sigterm(monkeypatch):
    monkeypatch.setattr(runner, "TERMINATE_GRACE", 0.2)
    # Ignored signals stay ignored across exec, so ``sleep`` ignores SIGTERM.
    process = subprocess.Popen(["sh", "-c", 'trap "" TERM; echo ready; exec sleep 30'], stdout=subprocess.PIPE)
    try:
        assert process.stdout.readline() == b"ready\n"
        start = time.monotonic()
        runner._stop(process)
        assert process.returncode == -signal.SIGKILL
        assert 0.15 <= time.monotonic() - start < 2
    finally:
        process.stdout.close()
        if process.poll() is None:  # pragma: no cover - only if the test failed
            process.kill()
            process.wait()


def test_terminate_all_without_sessions_does_nothing(active):
    terminate_all()
    assert active == set()


def test_terminate_all_stops_every_tracked_session(active):
    first, second = FakePopen(), FakePopen(stubborn=1)
    active.update({first, second})
    terminate_all()
    assert first.events == ["terminate", "wait"]
    assert second.events == ["terminate", "wait", "kill", "wait"]


def test_terminate_all_stops_a_session_while_run_waits_for_it(monkeypatch, active):
    # Simulates Ctrl-C in the main thread while a worker thread is in run().
    fake = FakePopen(stdout=fake_output({"a": ("A\n", 0), "b": ("", None)}), during=lambda _: terminate_all())
    install(monkeypatch, fake)
    results = SSHRunner(["ssh", "lumi"]).run({"a": "x", "b": "y"})
    assert fake.events == ["popen", "communicate", "terminate", "wait"]
    assert results == {
        "a": CommandResult("A", "", 0),
        "b": CommandResult("", "", None, "the SSH session ended unexpectedly (exit status -15)"),
    }
    assert active == set()


def test_terminate_all_stops_a_real_session_from_another_thread(active):
    ssh = SSHRunner(["sh", "-c", "exec sleep 30", "_"], timeout=60, login_shell=False)
    outcome: Dict[str, object] = {}

    def target():
        try:
            outcome["result"] = ssh.run({"a": "echo hi"})
        except RemoteError as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    start = time.monotonic()
    thread.start()
    while not active and time.monotonic() - start < 5:
        time.sleep(0.01)
    assert len(active) == 1
    terminate_all()
    thread.join(5)
    assert not thread.is_alive()
    assert time.monotonic() - start < 5
    assert str(outcome["error"]) == "the remote shell exited with status -15 before running any command"
    assert active == set()


def test_interrupted_run_leaves_no_session_running(monkeypatch, active):
    fake = FakePopen(interrupt=True)
    install(monkeypatch, fake)
    with pytest.raises(KeyboardInterrupt):
        SSHRunner(["ssh", "lumi"]).run({"a": "x"})
    terminate_all()  # what the CLI does on KeyboardInterrupt
    assert "terminate" in fake.events or "kill" in fake.events


# ---------------------------------------------------------------------------
# End-to-end with a local bash instead of ssh
# ---------------------------------------------------------------------------


def test_local_shell_argv_ends_with_bash_s():
    assert local_runner().argv == ["sh", "-c", 'eval "$1"', "_", "bash -s"]


def test_end_to_end_separates_stdout_stderr_and_exit_codes():
    results = local_runner().run(
        {
            "ok": "echo out; echo err >&2",
            "fail": "echo partial; exit 3",
            "false": "false",
            "quiet": "true",
            "stderr-only": "echo 'only err' >&2; exit 2",
        }
    )
    assert results == {
        "ok": CommandResult("out", "err", 0),
        "fail": CommandResult("partial", "", 3),
        "false": CommandResult("", "", 1),
        "quiet": CommandResult("", "", 0),
        "stderr-only": CommandResult("", "only err", 2),
    }


def test_end_to_end_stdin_is_dev_null_and_script_continues():
    results = local_runner().run(
        {
            "cat": "cat",
            "read": "read line; echo \"rc=$? line=[$line]\"",
            "after": "echo still running",
        }
    )
    assert results == {
        "cat": CommandResult("", "", 0),
        "read": CommandResult("rc=1 line=[]", "", 0),
        "after": CommandResult("still running", "", 0),
    }


@pytest.mark.parametrize(
    "command, expected",
    [
        ("printf abc", "abc"),
        ("printf 'abc\\n'", "abc"),
        ("printf 'x\\n\\n'", "x\n"),
        ("printf '\\n\\nx\\n'", "\n\nx"),
        ("printf 'a\\n\\nb'", "a\n\nb"),
        ("printf ''", ""),
        ("printf '   \\n'", "   "),
    ],
)
def test_end_to_end_trailing_newline_handling(command, expected):
    assert local_runner().run({"k": command})["k"] == CommandResult(expected, "", 0)


def test_end_to_end_output_without_newline_on_both_streams_does_not_leak():
    results = local_runner().run({"a": "printf out; printf err >&2", "b": "echo next; echo next-err >&2"})
    assert results == {"a": CommandResult("out", "err", 0), "b": CommandResult("next", "next-err", 0)}


def test_end_to_end_unicode_and_invalid_utf8():
    results = local_runner().run(
        {
            "unicode": "printf '%s\\n' 'αβγ – ✓ Ελληνικά 日本語'",
            "invalid": "printf 'ok \\377\\n'",
        }
    )
    assert results["unicode"] == CommandResult("αβγ – ✓ Ελληνικά 日本語", "", 0)
    assert results["invalid"] == CommandResult("ok �", "", 0)


def test_end_to_end_quoting_and_special_characters():
    results = local_runner().run(
        {
            "quotes": "echo \"it's\" | tr a-z A-Z",
            "dollar": "x=5; echo \"$x\" '$x'",
            "pipes": "printf '%s\\n' '1|part|[a]' 'nid[001-004]|x'",
        }
    )
    assert results == {
        "quotes": CommandResult("IT'S", "", 0),
        "dollar": CommandResult("5 $x", "", 0),
        "pipes": CommandResult("1|part|[a]\nnid[001-004]|x", "", 0),
    }


def test_end_to_end_command_exiting_or_killed_does_not_stop_the_script():
    results = local_runner().run(
        {
            "exit": "exit 7",
            "killed": "kill -9 $$",
            "missing": "hpcmon-surely-missing-command-xyz",
            "after": "echo alive",
        }
    )
    assert results["exit"] == CommandResult("", "", 7)
    assert results["killed"].returncode == 137
    assert results["killed"].reason is None
    assert results["missing"].returncode == 127
    assert "hpcmon-surely-missing-command-xyz" in results["missing"].stderr
    assert results["after"] == CommandResult("alive", "", 0)


def test_end_to_end_fake_end_marker_in_output_is_content():
    fake_end = end("other", 0)
    results = local_runner().run({"k": f"echo before; echo '{fake_end}'; echo after"})
    assert results["k"] == CommandResult(f"before\n{fake_end}\nafter", "", 0)


def test_end_to_end_ignores_newline_terminated_banners():
    ssh = SSHRunner(
        ["sh", "-c", 'echo "Welcome to the cluster"; echo "Lmod notice" >&2; eval "$1"', "_"],
        timeout=20,
        login_shell=False,
    )
    assert ssh.run({"a": "echo hi", "b": "echo there >&2"}) == {
        "a": CommandResult("hi", "", 0),
        "b": CommandResult("", "there", 0),
    }


def test_end_to_end_ignores_banner_without_trailing_newline():
    ssh = SSHRunner(["sh", "-c", 'printf "Welcome"; printf "Notice" >&2; eval "$1"', "_"], timeout=20, login_shell=False)
    assert ssh.run({"a": "echo hi; echo there >&2"}) == {"a": CommandResult("hi", "there", 0)}


@pytest.mark.parametrize(
    "fake_ssh, expected",
    [
        (
            'echo "ssh: Could not resolve hostname nowhere: Name or service not known" >&2; exit 255',
            "SSH connection failed: ssh: Could not resolve hostname nowhere: Name or service not known",
        ),
        ("exit 255", "SSH connection failed: exit status 255"),
        (
            'echo "banner on stdout only"; exit 0',
            "the remote shell exited with status 0 before running any command: banner on stdout only",
        ),
        (
            'echo "motd"; echo "-bash: /etc/profile: Permission denied" >&2; exit 1',
            "the remote shell exited with status 1 before running any command: -bash: /etc/profile: Permission denied",
        ),
        ("exit 3", "the remote shell exited with status 3 before running any command"),
    ],
)
def test_end_to_end_connection_failure(fake_ssh, expected):
    ssh = SSHRunner(["sh", "-c", fake_ssh, "_"], timeout=20, login_shell=False)
    with pytest.raises(RemoteError) as excinfo:
        ssh.run({"a": "echo hi"})
    assert str(excinfo.value) == expected


def test_end_to_end_session_timeout_stops_the_process(monkeypatch, active):
    monkeypatch.setattr(runner, "SESSION_GRACE", 0.0)
    ssh = SSHRunner(["sh", "-c", "exec sleep 30", "_"], timeout=0.3, login_shell=False)
    start = time.monotonic()
    with pytest.raises(RemoteError) as excinfo:
        ssh.run({"a": "echo hi"})
    assert str(excinfo.value) == "no response within 0.3 s"
    assert time.monotonic() - start < 3
    assert active == set()


def test_end_to_end_session_timeout_kills_a_process_ignoring_sigterm(monkeypatch):
    monkeypatch.setattr(runner, "SESSION_GRACE", 0.0)
    monkeypatch.setattr(runner, "TERMINATE_GRACE", 0.2)
    ssh = SSHRunner(["sh", "-c", 'trap "" TERM; exec sleep 30', "_"], timeout=0.4, login_shell=False)
    start = time.monotonic()
    with pytest.raises(RemoteError, match=r"^no response within 0\.4 s$"):
        ssh.run({"a": "echo hi"})
    assert time.monotonic() - start < 3


def test_end_to_end_session_timeout_keeps_partial_output(monkeypatch):
    monkeypatch.setattr(runner, "SESSION_GRACE", 0.0)
    fake_ssh = f"printf '\\n%s\\n' '{begin('a')}'; echo partial; exec sleep 30"
    ssh = SSHRunner(["sh", "-c", fake_ssh, "_"], timeout=1.0, login_shell=False)
    reason = "the SSH session timed out after 1 s"
    assert ssh.run({"a": "slow", "b": "never"}) == {
        "a": CommandResult("partial", "", None, reason),
        "b": CommandResult("", "", None, reason),
    }


def test_end_to_end_session_dropped_between_commands():
    fake_ssh = (
        f"printf '\\n%s\\nA\\n\\n%s\\n' '{begin('a')}' '{end('a', 0)}'; "
        f"printf '\\n%s\\n\\n%s\\n' '{begin('a')}' '{end('a', 0)}' >&2; "
        "echo 'Connection to host closed by remote host.' >&2; exit 255"
    )
    ssh = SSHRunner(["sh", "-c", fake_ssh, "_"], timeout=20, login_shell=False)
    assert ssh.run({"a": "echo A", "b": "echo B"}) == {
        "a": CommandResult("A", "", 0),
        "b": CommandResult(
            "", "", None, "the SSH session ended unexpectedly (Connection to host closed by remote host.)"
        ),
    }


def test_end_to_end_remote_shell_killed_mid_script(monkeypatch, tmp_path):
    # Without ``timeout`` on PATH the commands are children of the script's
    # bash, so ``kill -9 $PPID`` ends the remote shell in the middle of "b".
    for tool in ("sh", "bash"):
        (tmp_path / tool).symlink_to(shutil.which(tool))
    monkeypatch.setenv("PATH", str(tmp_path))
    fake_ssh = 'eval "$1"; echo "Connection to host closed by remote host." >&2; exit 255'
    ssh = SSHRunner(["sh", "-c", fake_ssh, "_"], timeout=20, login_shell=False)
    results = ssh.run({"a": "echo A", "b": "echo B; kill -9 $PPID", "c": "echo C"})
    reason = "the SSH session ended unexpectedly (exit status 255)"
    assert results["a"] == CommandResult("A", "", 0)
    assert (results["b"].stdout, results["b"].returncode, results["b"].reason) == ("B", None, reason)
    # ssh's message (after sh's own "Killed" notice) follows the BEGIN marker
    # of "b", so it counts as output of that command, not as session detail.
    assert last_line(results["b"].stderr) == "Connection to host closed by remote host."
    assert results["c"] == CommandResult("", "", None, reason)


@needs_timeout
def test_end_to_end_runner_timeout_is_the_command_budget():
    # The runner's timeout becomes the commands' shared budget; the session
    # itself may take SESSION_GRACE more, so the results still arrive.
    start_early_in_a_second()
    start = time.monotonic()
    results = local_runner(timeout=1).run({"slow": "echo started; sleep 5", "skipped": "echo never"})
    assert time.monotonic() - start < 3.5
    assert results == {
        "slow": CommandResult("started", "", TIMEOUT_EXIT_CODE),
        "skipped": CommandResult("", "", TIMEOUT_EXIT_CODE),
    }
