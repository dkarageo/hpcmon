"""Execution of remote commands over SSH.

All the commands needed for a cluster are bundled into a single shell script
that is fed to ``bash -s`` on the remote side through one SSH session.  Every
command's output is wrapped in unique markers (on both stdout and stderr), so
the output of each command can be separated again locally, and so that any
banner printed by the remote login shell is ignored.
"""

from __future__ import annotations

import math
import re
import shlex
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import IO, Dict, List, Mapping, Optional, Sequence, Set

#: Prefix of the markers that delimit the output of every remote command.
MARKER = "@@HPCMON@@"

_KEY_RE = re.compile(r"[A-Za-z0-9_.-]+")
_BEGIN_RE = re.compile(r"^" + re.escape(MARKER) + r"BEGIN@@(?P<key>[A-Za-z0-9_.-]+)@@$")
_END_RE = re.compile(r"^" + re.escape(MARKER) + r"END@@(?P<key>[A-Za-z0-9_.-]+)@@(?P<rc>-?\d+)@@$")

#: Exit status reported by ``timeout`` when it kills a command.
TIMEOUT_EXIT_CODE = 124

#: Exit status used by OpenSSH for connection errors.
SSH_ERROR_EXIT_CODE = 255

#: Extra time, in seconds, granted to an SSH session on top of the time limit
#: of its commands, for connecting and loading the login profile.
SESSION_GRACE = 15.0

#: Time, in seconds, given to ssh to exit cleanly after SIGTERM before SIGKILL.
#: OpenSSH restores the terminal (e.g. echo after a password prompt) on SIGTERM.
TERMINATE_GRACE = 2.0

_active_processes: Set["subprocess.Popen[bytes]"] = set()
_active_lock = threading.Lock()


class RemoteError(Exception):
    """Raised when the remote host could not be queried at all."""


@dataclass
class CommandResult:
    """The outcome of a single remote command."""

    stdout: str = ""
    stderr: str = ""
    #: The exit status of the command, or ``None`` when it never finished
    #: (e.g. because the whole SSH session timed out).
    returncode: Optional[int] = None
    #: Why the command did not finish, when ``returncode`` is ``None``.
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Whether the command finished with exit status zero."""
        return self.returncode == 0

    @property
    def timed_out(self) -> bool:
        """Whether the command was killed because it took too long."""
        return self.returncode is None or self.returncode == TIMEOUT_EXIT_CODE


def build_script(commands: Mapping[str, str], command_timeout: Optional[float] = None) -> str:
    """Build the bash script that runs all ``commands`` and marks their output.

    The commands run one after the other, in the order of ``commands``.  With
    ``command_timeout``, they share that time budget: each command is stopped
    (exit status 124) when the budget is exhausted, and the remaining commands
    are skipped with the same status, so that the output of the commands that
    did complete is always delivered.

    :param commands: Mapping from a unique key to a shell command line.
    :param command_timeout: Optional limit, in seconds, for all the commands.
    """
    if command_timeout is None or not math.isfinite(command_timeout):
        lines: List[str] = ["__hpcmon_run() {", '  bash -c "$1"', "}"]
    else:
        lines = [
            f"__hpcmon_deadline=$((SECONDS + {max(1, int(command_timeout))}))",
            "__hpcmon_run() {",
            "  __hpcmon_left=$((__hpcmon_deadline - SECONDS))",
            '  if [ "$__hpcmon_left" -le 0 ]; then',
            f"    return {TIMEOUT_EXIT_CODE}",
            "  fi",
            "  if command -v timeout >/dev/null 2>&1; then",
            '    timeout "$__hpcmon_left" bash -c "$1"',
            "  else",
            '    bash -c "$1"',
            "  fi",
            "}",
        ]
    for key, command in commands.items():
        if not _KEY_RE.fullmatch(key):
            raise ValueError(f"invalid command key: {key!r}")
        begin = f"{MARKER}BEGIN@@{key}@@"
        end = f"{MARKER}END@@{key}@@"
        # Markers start on a new line, even if a login banner did not end one.
        lines.extend(
            [
                f"printf '\\n%s\\n' '{begin}'",
                f"printf '\\n%s\\n' '{begin}' >&2",
                f"__hpcmon_run {shlex.quote(command)} </dev/null",
                "__hpcmon_rc=$?",
                f"printf '\\n%s%s@@\\n' '{end}' \"$__hpcmon_rc\"",
                f"printf '\\n%s%s@@\\n' '{end}' \"$__hpcmon_rc\" >&2",
            ]
        )
    lines.append("exit 0")
    return "\n".join(lines) + "\n"


def split_output(text: str) -> Dict[str, "tuple[str, Optional[int]]"]:
    """Split the combined output of a script built by :func:`build_script`.

    :return: Mapping from command key to ``(output, returncode)``. The return
        code is ``None`` when the end marker of a command was never seen.
    """
    sections: Dict[str, "tuple[str, Optional[int]]"] = {}
    current: Optional[str] = None
    buffer: List[str] = []
    for line in text.splitlines():
        begin = _BEGIN_RE.match(line)
        if begin:
            if current is not None:
                sections[current] = ("\n".join(buffer), None)
            current, buffer = begin.group("key"), []
            continue
        end = _END_RE.match(line)
        if end and end.group("key") == current:
            # The end marker is always preceded by an extra newline that
            # guards against output without a trailing newline.
            if buffer and buffer[-1] == "":
                buffer.pop()
            sections[current] = ("\n".join(buffer), int(end.group("rc")))
            current, buffer = None, []
            continue
        if current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = ("\n".join(buffer), None)
    return sections


def outside_output(text: str) -> str:
    """Return the lines of ``text`` that are outside of any command's markers."""
    lines: List[str] = []
    inside = False
    for line in text.splitlines():
        if _BEGIN_RE.match(line):
            inside = True
        elif _END_RE.match(line):
            inside = False
        elif not inside:
            lines.append(line)
    return "\n".join(lines)


def last_line(*texts: str) -> Optional[str]:
    """The last non-blank line of the first of ``texts`` that has one."""
    for text in texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return None


def terminate_all() -> None:
    """Stop every SSH session that is still running (e.g. on Ctrl-C)."""
    with _active_lock:
        processes = list(_active_processes)
    for process in processes:
        _stop(process)


def _stop(process: "subprocess.Popen[bytes]") -> None:
    process.terminate()
    try:
        process.wait(timeout=TERMINATE_GRACE)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _read(handle: IO[bytes]) -> str:
    handle.seek(0)
    return handle.read().decode("utf-8", errors="replace")


class SSHRunner:
    """Run shell commands on a remote host through a user-defined SSH command.

    :param ssh_command: The command used to connect to the host, e.g.
        ``["ssh", "lumi"]``.  The remote command ``bash -l -s`` (or
        ``bash -s``) is appended to it, and the script is sent on stdin.
    :param timeout: Limit, in seconds, for running all the commands.  The
        SSH session as a whole may take :data:`SESSION_GRACE` seconds more.
    :param login_shell: Whether to run the script in a login shell, so that
        the site's profile (and thus its ``PATH``) is loaded.
    """

    def __init__(self, ssh_command: Sequence[str], timeout: float = 120.0, login_shell: bool = True) -> None:
        if not ssh_command:
            raise ValueError("the SSH command must not be empty")
        self.ssh_command = list(ssh_command)
        self.timeout = timeout
        self.login_shell = login_shell

    @property
    def argv(self) -> List[str]:
        """The full local command line that is executed."""
        remote = "bash -l -s" if self.login_shell else "bash -s"
        return self.ssh_command + [remote]

    def run(self, commands: Mapping[str, str]) -> Dict[str, CommandResult]:
        """Run ``commands`` remotely and return their results by key.

        :raises RemoteError: if SSH could not be started or failed before any
            command produced output.
        """
        if not commands:
            return {}
        script = build_script(commands, command_timeout=self.timeout)
        session_limit = self.timeout + SESSION_GRACE
        # Output is captured through temporary files rather than pipes: an SSH
        # ControlMaster spawned by this very command (``ControlPersist``) keeps
        # the inherited descriptors open, so reading pipes until EOF would hang.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                process = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=out, stderr=err)
            except OSError as exc:
                raise RemoteError(f"cannot run {shlex.join(self.argv)!r}: {exc}") from exc
            with _active_lock:
                _active_processes.add(process)
            timed_out = False
            try:
                process.communicate(input=script.encode("utf-8"), timeout=session_limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop(process)
            except BaseException:
                # E.g. KeyboardInterrupt: don't leave the session running.
                _stop(process)
                raise
            finally:
                with _active_lock:
                    _active_processes.discard(process)
            stdout, stderr = _read(out), _read(err)

        out_sections = split_output(stdout)
        err_sections = split_output(stderr)
        if not out_sections:
            if timed_out:
                raise RemoteError(f"no response within {session_limit:g} s")
            if process.returncode == SSH_ERROR_EXIT_CODE:
                raise RemoteError(f"SSH connection failed: {last_line(stderr) or 'exit status 255'}")
            detail = last_line(stderr, stdout)
            raise RemoteError(
                f"the remote shell exited with status {process.returncode} before running any command"
                + (f": {detail}" if detail else "")
            )

        if timed_out:
            reason = f"the SSH session timed out after {session_limit:g} s"
        else:
            detail = last_line(outside_output(stderr)) or f"exit status {process.returncode}"
            reason = f"the SSH session ended unexpectedly ({detail})"
        results: Dict[str, CommandResult] = {}
        for key in commands:
            text, returncode = out_sections.get(key, ("", None))
            err_text, _ = err_sections.get(key, ("", None))
            results[key] = CommandResult(
                stdout=text,
                stderr=err_text,
                returncode=returncode,
                reason=reason if returncode is None else None,
            )
        return results
