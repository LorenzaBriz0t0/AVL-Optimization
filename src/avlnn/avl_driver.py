"""Drives the AVL binary as a subprocess against a single design's geometry/mass/run-case files.

AVL has no non-interactive batch flag; it is scripted by piping a sequence of menu keystrokes
to its stdin. The sequence in `_command_script` was verified against AVL 3.52 on 2026-07-15
(and against session2.txt + src/amode.f from the 3.52 source tree). Two ordering rules that
matter, learned the hard way:

- MSET must come AFTER CASE: MSET copies the mass file's mass/inertia/CG into the stored run
  cases, and amode.f refuses to compute eigenmodes if the run case has Ixx<=0 -- which is what
  you get if CASE (re)loads run-case parameters after MSET already ran.
- MODE is a top-level menu, not an OPER subcommand: an extra <return> is needed to leave OPER
  first, otherwise "MODE" is rejected and the following "N" is misread as OPER's "Name case".
- MODE's N command unconditionally draws an X11 root-locus plot (PLEMAP in aplotmd.f has no
  graphics-enable guard, so PLOP->G does not help): with no display AVL segfaults (exit -11)
  before W can write the .eig file. On headless machines AVL therefore runs under `xvfb-run`
  (a virtual framebuffer X server) whenever DISPLAY is unset and xvfb-run is installed.

Eigenvalues are written to a file (MODE's W command) rather than scraped from stdout: the
.eig file is a stable three-column format, while the N command's screen dump interleaves
eigenvectors with menu redraws.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
from pathlib import Path

AVL_BINARY_ENV_VAR = "AVL_BIN"
DEFAULT_AVL_BINARY = "avl"
EIG_FILENAME = "aircraft.eig"


def _avl_invocation() -> list[str]:
    """The AVL argv, wrapped in xvfb-run on headless machines (see module docstring).
    xvfb-run -a picks a free virtual display per call, so parallel workers don't collide."""
    argv = [avl_binary_path()]
    if not os.environ.get("DISPLAY"):
        xvfb_run = shutil.which("xvfb-run")
        if xvfb_run:
            argv = [xvfb_run, "-a"] + argv
    return argv


class AvlExecutionError(RuntimeError):
    """Raised when the AVL subprocess fails to run or exits with a non-zero code."""


@dataclasses.dataclass(frozen=True)
class AvlRunResult:
    run_dir: Path
    stdout: str
    returncode: int
    eig_text: str | None    # contents of the MODE-written .eig file, if it was produced


def avl_binary_path() -> str:
    return os.environ.get(AVL_BINARY_ENV_VAR, DEFAULT_AVL_BINARY)


def _command_script(geom_file: str, mass_file: str, run_file: str) -> str:
    return "\n".join([
        "PLOP",                 # plotting options menu...
        "G",                    #   ...toggle graphics off: MODE's N draws a root-locus plot,
        "",                     #   which segfaults on a headless server with no X display
        f"LOAD {geom_file}",
        f"MASS {mass_file}",
        f"CASE {run_file}",
        "MSET",                 # apply mass file's mass/inertia/CG to the loaded run case(s);
        "0",                    #   0 = all cases; MUST follow CASE (see module docstring)
        "OPER",
        "X",                    # execute/trim run case 1
        "ST",                   # dump stability-axis derivatives + neutral point
        "",                     # <return> = print to stdout instead of a file
        "",                     # <return> = leave OPER, back to top-level menu
        "MODE",                 # eigenvalue analysis menu (top-level only)
        "N",                    # new eigenmode calculation at the trimmed operating point
        "W",                    # write eigenvalues to a file...
        EIG_FILENAME,           #   ...with this explicit name (fresh scratch dir, no clash)
        "",                     # <return> back to top-level menu
        "QUIT",
        "",
    ]) + "\n"


def run_avl(
    run_dir: Path,
    geometry_text: str,
    mass_text: str,
    run_case_text: str,
    timeout_s: float = 30.0,
) -> AvlRunResult:
    """Writes the three AVL input files into `run_dir` and executes AVL against them."""
    run_dir.mkdir(parents=True, exist_ok=True)
    geom_path = run_dir / "aircraft.avl"
    mass_path = run_dir / "aircraft.mass"
    run_path = run_dir / "aircraft.run"
    geom_path.write_text(geometry_text, encoding="utf-8")
    mass_path.write_text(mass_text, encoding="utf-8")
    run_path.write_text(run_case_text, encoding="utf-8")

    script = _command_script(geom_path.name, mass_path.name, run_path.name)

    try:
        proc = subprocess.run(
            _avl_invocation(), cwd=run_dir, input=script,
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AvlExecutionError(f"failed to run AVL in {run_dir}: {exc}") from exc

    eig_path = run_dir / EIG_FILENAME
    eig_text = eig_path.read_text(encoding="utf-8") if eig_path.exists() else None
    return AvlRunResult(
        run_dir=run_dir, stdout=proc.stdout, returncode=proc.returncode, eig_text=eig_text,
    )
