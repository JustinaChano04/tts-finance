"""Executes generated Python solutions in a restricted subprocess.

Security model (prototype-level, NOT a hard sandbox):
  * A static AST check rejects `import`, dunder attribute access, and calls
    to `open`/`eval`/`exec`/`compile`/`__import__` before anything runs.
  * The code executes in a *separate subprocess* with `python -I -S`
    (isolated mode: no site-packages, no user site, no reading PYTHONPATH),
    an empty environment, a restricted `__builtins__` mapping (no `open`,
    no `__import__`), and a wall-clock timeout.
  * On POSIX, CPU time and address-space are additionally capped via
    `resource.setrlimit` in the child before exec.

This is defense-in-depth for a research prototype, not a security boundary
suitable for untrusted multi-tenant use. It does not prevent all forms of
resource exhaustion (e.g. it will not stop a fork bomb on platforms where
RLIMIT_NPROC isn't enforced) and relies on the AST check to catch dangerous
calls rather than blocking them at the OS level. Do not run this against
adversarial input without further hardening (e.g. a real container/VM).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tts_finance.tasks.base import FinancialQuestion
from tts_finance.verifiers.base import VerificationResult, Verifier

_DISALLOWED_CALL_NAMES = {"open", "eval", "exec", "compile", "__import__", "input"}

_WRAPPER_TEMPLATE = '''
import json
import math

SAFE_BUILTINS = {{
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "float": float, "int": int, "str": str, "bool": bool,
    "pow": pow, "range": range, "list": list, "dict": dict, "tuple": tuple,
    "set": set, "enumerate": enumerate, "zip": zip, "sorted": sorted,
    "map": map, "filter": filter, "True": True, "False": False, "None": None,
    "print": print, "isinstance": isinstance,
}}

safe_globals = {{"__builtins__": SAFE_BUILTINS, "math": math}}

code = {code!r}

try:
    exec(compile(code, "<solution>", "exec"), safe_globals)
    answer = safe_globals.get("answer")
    result = {{"ok": True, "answer": answer, "error": None}}
except Exception as e:
    result = {{"ok": False, "answer": None, "error": f"{{type(e).__name__}}: {{e}}"}}

print(json.dumps(result, default=str))
'''


def _preexec_limits():  # pragma: no cover - POSIX-only, exercised at runtime
    import resource

    for limit, value in (
        (resource.RLIMIT_CPU, (5, 5)),
        (resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024)),
    ):
        try:
            resource.setrlimit(limit, value)
        except (ValueError, OSError):
            pass


def check_code_safety(code: str) -> tuple[bool, str | None]:
    """Static AST check. Returns (is_safe, reason_if_not)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "import statements are not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"dunder attribute access is not allowed: {node.attr}"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return False, f"dunder name access is not allowed: {node.id}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DISALLOWED_CALL_NAMES:
                return False, f"call to '{node.func.id}' is not allowed"

    return True, None


class PythonExecutorVerifier(Verifier):
    def __init__(self, timeout_seconds: float = 5.0, tolerance: float = 0.01):
        self.timeout_seconds = timeout_seconds
        self.tolerance = tolerance

    def verify(self, code: str, question: FinancialQuestion) -> VerificationResult:
        if not code or not code.strip():
            return VerificationResult(
                success=False, answer=None, correct=False,
                execution_error="no code extracted from response",
            )

        is_safe, reason = check_code_safety(code)
        if not is_safe:
            return VerificationResult(
                success=False, answer=None, correct=False,
                execution_error=f"rejected by safety check: {reason}",
            )

        script = _WRAPPER_TEMPLATE.format(code=code)

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "solution.py"
            script_path.write_text(script)

            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-S", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=tmp_dir,
                    env={},
                    preexec_fn=_preexec_limits if sys.platform != "win32" else None,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    success=False, answer=None, correct=False,
                    execution_error=f"execution timed out after {self.timeout_seconds}s",
                )

        if proc.returncode != 0:
            return VerificationResult(
                success=False, answer=None, correct=False,
                execution_error=proc.stderr.strip()[-2000:] or "non-zero exit code",
                stdout=proc.stdout,
            )

        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return VerificationResult(
                success=False, answer=None, correct=False,
                execution_error="could not parse execution output",
                stdout=proc.stdout,
            )

        if not payload["ok"]:
            return VerificationResult(
                success=False, answer=None, correct=False,
                execution_error=payload["error"], stdout=proc.stdout,
            )

        answer = payload["answer"]
        if answer is None:
            return VerificationResult(
                success=False, answer=None, correct=False,
                execution_error="code did not assign a variable named `answer`",
                stdout=proc.stdout,
            )

        try:
            answer = float(answer)
        except (TypeError, ValueError):
            return VerificationResult(
                success=False, answer=None, correct=False,
                execution_error=f"`answer` is not numeric: {answer!r}",
                stdout=proc.stdout,
            )

        correct = self._matches_gold(answer, question.gold_answer)
        return VerificationResult(success=True, answer=answer, correct=correct, stdout=proc.stdout)

    def _matches_gold(self, answer: float, gold: float | None) -> bool:
        if gold is None:
            return False
        return abs(answer - gold) <= self.tolerance * max(abs(gold), 1.0)
