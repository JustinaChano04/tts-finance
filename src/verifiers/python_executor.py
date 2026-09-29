"""Executes generated Python solutions in a restricted subprocess.

Security model (prototype-level, NOT a hard sandbox):
  * A static AST check rejects imports of anything but `math`, dunder
    attribute access, dunder names other than `__name__`, and calls to
    `open`/`eval`/`exec`/`compile`/`__import__`/`input` before anything runs.
  * The code executes in a *separate subprocess* with `python -I -S`
    (isolated mode: no site-packages, no user site, no reading PYTHONPATH),
    an empty environment, a restricted `__builtins__` mapping (no `open`,
    and an `__import__` that only returns `math`), and a wall-clock timeout.
  * On POSIX, CPU time and address-space are additionally capped via
    `resource.setrlimit` in the child before exec.

This is defense-in-depth for a research prototype, not a security boundary
suitable for untrusted multi-tenant use. It does not prevent all forms of
resource exhaustion (e.g. it will not stop a fork bomb on platforms where
RLIMIT_NPROC isn't enforced) and relies on the AST check to catch dangerous
calls rather than blocking them at the OS level. Do not run this against
adversarial input without further hardening (e.g. a real container/VM).

Error attribution: every failure carries an `error_type`. The allowed subset
is deliberately broad (common builtins, exception classes, `import math`,
`if __name__ == "__main__":`) so that valid code a model would reasonably
write is not reported as a model failure. The result is returned through a
file rather than stdout, so the model's own `print` calls cannot corrupt it.
"""

from __future__ import annotations

import ast
import json
import math
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from tasks.base import FinancialQuestion
from verifiers.base import VerificationResult, Verifier
from verifiers.matching import MatchMode, matches_gold

_DISALLOWED_CALL_NAMES = {"open", "eval", "exec", "compile", "__import__", "input"}
_ALLOWED_IMPORTS = {"math"}
_ALLOWED_DUNDER_NAMES = {"__name__"}
_SIGXCPU = getattr(signal, "SIGXCPU", None)  # absent on Windows

_WRAPPER_TEMPLATE = """
import json
import math

_ALLOWED_MODULES = {{"math": math}}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name in _ALLOWED_MODULES:
        return _ALLOWED_MODULES[name]
    raise ImportError(f"import of {{name!r}} is not allowed")


SAFE_BUILTINS = {{
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "float": float, "int": int, "str": str, "bool": bool,
    "pow": pow, "range": range, "list": list, "dict": dict, "tuple": tuple,
    "set": set, "enumerate": enumerate, "zip": zip, "sorted": sorted,
    "map": map, "filter": filter, "True": True, "False": False, "None": None,
    "print": print, "isinstance": isinstance, "all": all, "any": any,
    "reversed": reversed, "divmod": divmod, "next": next, "iter": iter,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError, "ArithmeticError": ArithmeticError,
    "__import__": _safe_import, "__build_class__": __build_class__,
}}

safe_globals = {{"__builtins__": SAFE_BUILTINS, "__name__": "__main__", "math": math}}

code = {code!r}

try:
    exec(compile(code, "<solution>", "exec"), safe_globals)
    result = {{
        "ok": True,
        "assigned": "answer" in safe_globals,
        "answer": safe_globals.get("answer"),
        "error": None,
    }}
except Exception as e:
    result = {{"ok": False, "assigned": False, "answer": None, "error": f"{{type(e).__name__}}: {{e}}"}}

with open({result_path!r}, "w") as f:
    json.dump(result, f, default=str)
"""


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
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    return False, f"import of '{alias.name}' is not allowed"
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in _ALLOWED_IMPORTS:
                return False, f"import from '{node.module}' is not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"dunder attribute access is not allowed: {node.attr}"
        if (
            isinstance(node, ast.Name)
            and node.id.startswith("__")
            and node.id not in _ALLOWED_DUNDER_NAMES
        ):
            return False, f"dunder name access is not allowed: {node.id}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DISALLOWED_CALL_NAMES:
                return False, f"call to '{node.func.id}' is not allowed"

    return True, None


class PythonExecutorVerifier(Verifier):
    def __init__(
        self,
        timeout_seconds: float = 5.0,
        tolerance: float = 0.01,
        match_mode: MatchMode = "strict",
    ):
        self.timeout_seconds = timeout_seconds
        self.tolerance = tolerance
        self.match_mode = match_mode

    def matches(self, answer: float | None, question: FinancialQuestion) -> bool:
        return matches_gold(answer, question.gold_answer, self.tolerance, self.match_mode)

    def verify(self, code: str, question: FinancialQuestion) -> VerificationResult:
        if not code or not code.strip():
            return VerificationResult.failure("no_code", "no code extracted from response")

        try:
            ast.parse(code)
        except SyntaxError as e:
            return VerificationResult.failure("syntax_error", f"SyntaxError: {e}")

        is_safe, reason = check_code_safety(code)
        if not is_safe:
            return VerificationResult.failure(
                "safety_reject", f"rejected by safety check: {reason}"
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "solution.py"
            result_path = Path(tmp_dir) / "result.json"
            script_path.write_text(
                _WRAPPER_TEMPLATE.format(code=code, result_path=str(result_path))
            )

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
                return VerificationResult.failure(
                    "timeout", f"execution timed out after {self.timeout_seconds}s"
                )

            raw_result = result_path.read_text() if result_path.exists() else None

        if _SIGXCPU is not None and proc.returncode == -_SIGXCPU:
            return VerificationResult.failure(
                "timeout", "killed by CPU time limit", stdout=proc.stdout
            )

        if proc.returncode != 0 or raw_result is None:
            # The wrapper catches every exception the solution can raise, so a
            # crash or a missing result file means the harness itself failed.
            detail = proc.stderr.strip()[-2000:] or f"exit code {proc.returncode}"
            return VerificationResult.failure(
                "harness_error", f"executor produced no result: {detail}", stdout=proc.stdout
            )

        try:
            payload = json.loads(raw_result)
        except json.JSONDecodeError:
            return VerificationResult.failure(
                "harness_error", "could not parse executor result", stdout=proc.stdout
            )

        if not payload["ok"]:
            return VerificationResult.failure(
                "runtime_error", payload["error"], stdout=proc.stdout
            )

        if not payload["assigned"]:
            return VerificationResult.failure(
                "no_answer_var",
                "code did not assign a variable named `answer`",
                stdout=proc.stdout,
            )

        answer = _to_finite_float(payload["answer"])
        if answer is None:
            return VerificationResult.failure(
                "non_numeric",
                f"`answer` is not a finite number: {payload['answer']!r}",
                stdout=proc.stdout,
            )

        return VerificationResult(
            success=True,
            answer=answer,
            correct=self.matches(answer, question),
            stdout=proc.stdout,
        )


def _to_finite_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
