# cadquery_debugger.py
from __future__ import annotations
import ast
import io
import os
import re
import sys
import time
import traceback
import textwrap
import contextlib
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import resource  # POSIX only, used for CPU/mem limits
except Exception:
    resource = None
# --------------------------
# Utilities & data models
# --------------------------
@dataclass
class CodeFrame:
    lineno: int
    col_offset: int
    line: str
    pointer_line: str

@dataclass
class Issue:
    kind: str                 # "syntax", "runtime", "warning", "info"
    message: str
    lineno: Optional[int] = None
    col_offset: Optional[int] = None
    frame: Optional[CodeFrame] = None
    suggestion: Optional[str] = None
    exception_type: Optional[str] = None
    traceback: Optional[str] = None

@dataclass
class DebugResult:
    ok: bool
    issues: List[Issue]
    ran: bool
    runtime_seconds: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    extra: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        # Convert dataclasses into JSON-serializable dict
        d = asdict(self)
        # Convert CodeFrame dataclasses inside issues
        for i, issue in enumerate(d["issues"]):
            if issue["frame"] is not None:
                issue["frame"] = asdict(issue["frame"])
        return d


# --------------------------
# Pretty code frame
# --------------------------

def _code_frame(src: str, lineno: int, col: int) -> CodeFrame:
    lines = src.splitlines()
    idx = max(0, lineno - 1)
    line = lines[idx] if 0 <= idx < len(lines) else ""
    # Create caret pointer (1-indexed columns in SyntaxError)
    pointer = " " * max(0, (col - 1)) + "^"
    return CodeFrame(lineno=lineno, col_offset=col, line=line, pointer_line=pointer)


# --------------------------
# Static / syntax checks
# --------------------------

def _syntax_check(src: str) -> Tuple[Optional[Issue], Optional[ast.AST]]:
    try:
        tree = ast.parse(src, filename="<agent_code>", mode="exec")
        return None, tree
    except SyntaxError as e:
        frame = _code_frame(src, e.lineno or 1, (e.offset or 1))
        msg = e.msg or "Syntax error"
        sug = _syntax_suggestion(msg)
        return Issue(
            kind="syntax",
            message=msg,
            lineno=e.lineno,
            col_offset=e.offset,
            frame=frame,
            suggestion=sug,
            exception_type="SyntaxError",
        ), None

def _syntax_suggestion(msg: str) -> Optional[str]:
    # Minimal heuristics for quick tips
    lower = msg.lower()
    if "unexpected indent" in lower:
        return "Fix indentation. Mixed tabs/spaces or a stray indent is likely."
    if "expected ':'" in lower:
        return "Missing ':' — after 'def', 'class', 'if', 'for', etc."
    if "eol while scanning string literal" in lower or "unterminated string" in lower:
        return "Unterminated string — check quotes match and escape backslashes."
    if "invalid syntax" in lower:
        return "Double-check parentheses, commas, and colons around the reported spot."
    if "cannot assign to function call" in lower:
        return "You used '=' instead of '==' in a comparison or tried to assign to a call."
    if "name error" in lower:
        return "You referenced a name before defining it."
    return None


# --------------------------
# CadQuery awareness (linty)
# --------------------------

_CQ_IMPORT_PATTERN = re.compile(r"^\s*import\s+cadquery\s+as\s+cq\s*$|^\s*from\s+cadquery\s+import\s+.*", re.MULTILINE)

def _cadquery_checks(src: str) -> List[Issue]:
    issues: List[Issue] = []

    if "cadquery" in src or "cq." in src or "Workplane" in src:
        if not _CQ_IMPORT_PATTERN.search(src):
            issues.append(Issue(
                kind="warning",
                message="CadQuery used but no import found.",
                suggestion="Add `import cadquery as cq` (recommended) or appropriate CadQuery imports."
            ))

    # Gentle nudges on common pitfalls
    if re.search(r"\bcq\.Workplane\([^)]*\)\s*(\n|\r|.)*?\.val\(\)", src, re.DOTALL):
        issues.append(Issue(
            kind="info",
            message="Using `.val()` returns a single object—downstream ops expecting a Workplane will fail.",
            suggestion="Use `.objects` when you need a list, or avoid `.val()` if you want to keep chaining."
        ))

    if re.search(r"\bselect\(|\bfaces\(|\bedges\(|\bvertices\(", src) and ".size(" in src:
        issues.append(Issue(
            kind="info",
            message="Selection followed by `.size(...)` can be `0` and cause downstream operations to explode.",
            suggestion="Check selections aren’t empty before relying on them, or assert non-empty."
        ))

    if "show_object(" in src and "import cadquery" not in src:
        # show_object can exist in CQ-editor contexts; keep as info
        issues.append(Issue(
            kind="info",
            message="`show_object(...)` is typically for CQ-Editor/GUI contexts.",
            suggestion="If running headless, export to STEP/ STL instead (e.g., `cq.exporters.export(...)`)."
        ))

    return issues


# --------------------------
# Runtime sandbox
# --------------------------

def _runner(code: str, q: mp.Queue, time_limit: float, preload_cq: bool) -> None:
    """
    Executes code safely-ish in a subprocess.
    Captures stdout/stderr and returns (ok, stdout, stderr, traceback).
    """
    # Try to restrict resources on POSIX
    if resource is not None:
        try:
            # CPU seconds
            resource.setrlimit(resource.RLIMIT_CPU, (int(time_limit) + 1, int(time_limit) + 1))
            # Address space / memory (512MB cap)
            soft = 512 * 1024 * 1024
            hard = 1024 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        except Exception:
            pass

    stdout = io.StringIO()
    stderr = io.StringIO()
    tb = ""
    ok = True

    # Isolate globals; optionally preload cadquery
    g: Dict[str, Any] = {"__name__": "__main__"}
    if preload_cq:
        try:
            import cadquery as cq  # type: ignore
            g["cq"] = cq
        except Exception:
            # If CQ isn't installed, we still run; errors will be captured
            pass

    try:
        compiled = compile(code, "<agent_code>", "exec")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compiled, g, None)
    except Exception:
        ok = False
        tb = traceback.format_exc()

    q.put((ok, stdout.getvalue(), stderr.getvalue(), tb))


def run_code_safely(code: str, timeout_sec: float = 10.0, preload_cq: bool = True) -> Tuple[bool, str, str, str]:
    """
    Run the user code in a subprocess with a timeout. Returns (ok, stdout, stderr, traceback_str).
    """
    ctx = mp.get_context("spawn") if sys.platform == "win32" else mp.get_context("fork")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_runner, args=(code, q, timeout_sec, preload_cq))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        return False, "", "", "TimeoutError: code execution exceeded {0:.2f}s".format(timeout_sec)

    try:
        ok, out, err, tb = q.get_nowait()
    except Exception:
        ok, out, err, tb = False, "", "", "ExecutionError: no result from subprocess."
    return ok, out, err, tb


# --------------------------
# Public Debugger API
# --------------------------

class CadQueryDebugger:
    """
    Minimal, dependency-free debugging tool for AI-generated CadQuery code.

    Typical use:
        dbg = CadQueryDebugger()
        result = dbg.debug(code, run=True, timeout_sec=8)
        print(result.to_dict())  # structured, agent-friendly

    CLI:
        python cadquery_debugger.py path/to/script.py --run
    """

    def __init__(self, preload_cadquery: bool = True) -> None:
        self.preload_cadquery = preload_cadquery

    def debug(self, src: str, run: bool = False, timeout_sec: float = 10.0) -> DebugResult:
        issues: List[Issue] = []

        # 1) Syntax
        syntax_issue, tree = _syntax_check(src)
        if syntax_issue:
            issues.append(syntax_issue)
            # CadQuery hints still useful even if syntax fails (simple text scan)
            issues.extend(_cadquery_checks(src))
            return DebugResult(ok=False, issues=issues, ran=False, stdout="", stderr="", extra={})

        # 2) CadQuery linty hints (best-effort)
        issues.extend(_cadquery_checks(src))

        # 3) Optional runtime
        ran = False
        ok = True
        out = err = tb = ""
        t0 = time.perf_counter()
        runtime_seconds: Optional[float] = None

        if run:
            ran = True
            ok, out, err, tb = run_code_safely(src, timeout_sec=timeout_sec, preload_cq=self.preload_cadquery)
            runtime_seconds = max(0.0, time.perf_counter() - t0)

            if not ok:
                # Promote runtime error to an issue with a trimmed code frame if we can guess a line
                lineno, col = _extract_tb_location(tb)
                frame = _code_frame(src, lineno, col) if lineno else None
                issues.append(Issue(
                    kind="runtime",
                    message="Runtime error while executing code.",
                    lineno=lineno,
                    col_offset=col,
                    frame=frame,
                    suggestion=_runtime_suggestion(tb),
                    exception_type=_extract_exc_type(tb),
                    traceback=tb,
                ))
                # Still return ok=False overall
                return DebugResult(ok=False, issues=issues, ran=True,
                                   runtime_seconds=runtime_seconds, stdout=out, stderr=err, extra={})

        # 4) All good
        return DebugResult(ok=True, issues=issues, ran=ran,
                           runtime_seconds=runtime_seconds, stdout=out, stderr=err, extra={})


# --------------------------
# Traceback helpers
# --------------------------

_TB_FILELINE_RE = re.compile(r'File "<agent_code>", line (\d+)(?:, in .*)?')
_EXC_TYPE_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_\.]*)\:', re.MULTILINE)

def _extract_tb_location(tb: str) -> Tuple[Optional[int], int]:
    if not tb:
        return None, 1
    m = _TB_FILELINE_RE.search(tb)
    if not m:
        return None, 1
    try:
        return int(m.group(1)), 1
    except Exception:
        return None, 1

def _extract_exc_type(tb: str) -> Optional[str]:
    if not tb:
        return None
    lines = tb.strip().splitlines()
    if not lines:
        return None
    # Last non-empty line generally has "TypeError: message"
    for line in reversed(lines):
        if ":" in line:
            return line.split(":", 1)[0].strip()
    # Fallback
    m = _EXC_TYPE_RE.search(tb)
    return m.group(1) if m else None

def _runtime_suggestion(tb: str) -> Optional[str]:
    if "ModuleNotFoundError" in tb and "cadquery" in tb.lower():
        return "CadQuery isn’t installed in this environment. `pip install cadquery` (or conda/mamba equivalent)."
    if "AttributeError" in tb and "Workplane" in tb:
        return "Invalid CadQuery chain or attribute. Check the API for the method you’re calling on Workplane."
    if "ValueError" in tb and ("empty" in tb.lower() or "no objects" in tb.lower()):
        return "Selection produced no geometry. Validate selectors or add preconditions before downstream ops."
    if "TimeoutError" in tb:
        return "Your code took too long. Consider simplifying geometry or raising the timeout."
    return None


# --------------------------
# Simple CLI
# --------------------------

def _read_file(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def _print_human(result: DebugResult) -> None:
    print("\n=== Result:", "OK" if result.ok else "NOT OK", "===")
    if result.ran:
        print(f"Ran in: {result.runtime_seconds:.3f}s")
    if result.stdout:
        print("\n--- stdout ---\n" + result.stdout)
    if result.stderr:
        print("\n--- stderr ---\n" + result.stderr)
    for issue in result.issues:
        print(f"\n[{issue.kind.upper()}] {issue.message}")
        if issue.exception_type:
            print(f"Type: {issue.exception_type}")
        if issue.lineno:
            print(f"Line: {issue.lineno}, Col: {issue.col_offset or 1}")
        if issue.frame:
            print(f"\n{issue.frame.lineno:>5} | {issue.frame.line}")
            print("      | " + issue.frame.pointer_line)
        if issue.suggestion:
            print("Suggestion:", issue.suggestion)
        if issue.traceback:
            print("\nTraceback (most recent call last):\n" + issue.traceback)

def main(argv: List[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Debug AI-generated CadQuery code.")
    parser.add_argument("path", help="Path to .py file to debug")
    parser.add_argument("--run", action="store_true", help="Execute the code in a sandboxed subprocess")
    parser.add_argument("--timeout", type=float, default=10.0, help="Runtime timeout in seconds")
    parser.add_argument("--no-preload-cq", action="store_true", help="Don’t import cadquery into the sandbox by default")
    parser.add_argument("--json", action="store_true", help="Emit JSON result instead of human print")
    args = parser.parse_args(argv)

    code = _read_file(args.path)
    dbg = CadQueryDebugger(preload_cadquery=not args.no_preload_cq)
    result = dbg.debug(code, run=args.run, timeout_sec=args.timeout)

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_human(result)

if __name__ == "__main__":
    main(sys.argv[1:])
