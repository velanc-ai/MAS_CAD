# executor_agent.py
from __future__ import annotations
import os
import re
import sys
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Literal, Any, Optional
from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

# ========= Shared schema (aligned with planner) =========
class CADOperation(BaseModel):
    model_config = {"extra": "ignore"}
    id: str
    op_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)

class CADExportSpec(BaseModel):
    model_config = {"extra": "ignore"}
    format: Literal["STEP", "IGES", "STL", "Parasolid", "3MF"] = "STEP"
    mesh_tolerance: str = "medium"
    units: Literal["mm", "cm", "in"] = "mm"

class CADSequencePlan(BaseModel):
    model_config = {"extra": "ignore"}
    part_id: str
    design_id: str
    intent: str
    Printable_Volume: str
    units: Literal["mm", "cm", "in"] = "mm"
    global_params: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    materials: Dict[str, Any] = Field(default_factory=dict)
    tolerance: str = "±0.1 mm"
    coordinate_system: Dict[str, Any] = Field(
        default_factory=lambda: {
            "origin": [0, 0, 0],
            "axes": ["+X", "+Y", "+Z"],
            "plane": "XY",
        }
    )
    steps: List[CADOperation] = Field(default_factory=list)
    export: CADExportSpec = Field(default_factory=CADExportSpec)

# ========= Prompts =========
SYSTEM_PROMPT = """
            You are a senior CAD code generator and fixer for CadQuery 2.x.
            
            TASK:
            - Given a validated CADSequencePlan (JSON), OUTPUT A SINGLE, SELF-CONTAINED PYTHON SCRIPT that builds the part.
            - Use minimal imports. Always begin with: `import cadquery as cq`.
            - Respect units, parameters, coordinate system, and steps in dependency order.
            - Final solid must be bound to a variable named `result`.
            - Add a simple STL export function if helpful but keep it headless (no GUI).
            - If later provided a debugger report, FIX the code and output a corrected full script.
            
            STRICT OUTPUT RULE:
            - When asked for code, output ONLY raw Python code (no markdown fences, no prose).
            
            - Below is the example code structure (You should not use the same code without using the json prompt input)
            import cadquery as cq
            import os
            
            # Step 1: Create the base flange cylinder
            base_flange = cq.Workplane("XY").circle(124 / 2).extrude(19)
            
            # Step 2: Create the central cylinder
            central_cylinder = cq.Workplane("XY").circle(90 / 2).extrude(141)
            
            # Combine the base flange and central cylinder
            combined = base_flange.union(central_cylinder)
            
            # Step 3: Create the central hole
            result = combined.faces(">Z").workplane().hole(36, 160)
            
            # Export function for STL
            def export_stl(solid, filename=None):
                if filename is None:
                    # get the current script name without extension
                    script_name = os.path.splitext(os.path.basename(__file__))[0]
                    filename = f"{{script_name}}.stl"
                cq.exporters.export(solid, filename)
                print("Exported to", filename)
            
            # Export the result to STL
            export_stl(result)
            
            
            """.strip()

HUMAN_PROMPT_BUILD = """
Here is the validated CADSequencePlan JSON. Generate the Python script now.

PLAN_JSON:
{plan_json}

Remember: OUTPUT ONLY PYTHON CODE — no markdown fences, no extra text.
""".strip()

HUMAN_PROMPT_FIX = """
You previously wrote this CadQuery script, but the debugger reported issues.

--- ORIGINAL CODE START ---
{old_code}
--- ORIGINAL CODE END ---

--- DEBUGGER REPORT START ---
{debug_report}
--- DEBUGGER REPORT END ---

Please OUTPUT a corrected FULL Python script (replace the entire file). Same rules as before.
Output ONLY raw Python code — no markdown, no prose.
""".strip()

prompt_builder = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", HUMAN_PROMPT_BUILD),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Debug-time prompt (WITH {input})
prompt_builder_debug = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", HUMAN_PROMPT_BUILD),
        ("human", "{input}"),  # <-- this is where your dbg_cmd goes
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# ========= Utilities =========
ALLOWED_IMPORTS = {"cadquery", "math", "pathlib", "typing", "os", "sys", "time", "random", "json", "OCP", "numpy"}

def _extract_imports(pycode: str) -> List[str]:
    """
    Extract top-level imported module names from code.
    - Ignores 'from __future__ import ...'
    - Keeps order, de-duplicates
    """
    mods: List[str] = []
    for ln in (pycode or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("from __future__ import"):
            continue  # ignore future imports
        if s.startswith("import "):
            # handle "import a, b as bb"
            parts = s.replace(",", " ").split()[1:]
            skip_next_as = False
            for p in parts:
                if skip_next_as:
                    skip_next_as = False
                    continue
                if p == "as":
                    skip_next_as = True
                    continue
                mods.append(p.split(".")[0])
        elif s.startswith("from "):
            # handle "from pkg.sub import x, y as z"
            mod = s.split()[1].split(".")[0]
            if mod != "__future__":
                mods.append(mod)

    # dedupe preserving order
    seen = set(); out = []
    for m in mods:
        if m not in seen:
            seen.add(m); out.append(m)
    return out


# def _extract_imports(pycode: str) -> List[str]:
#     mods: List[str] = []
#     for ln in (pycode or "").splitlines():
#         s = ln.strip()
#         if s.startswith("import "):
#             parts = s.replace(",", " ").split()[1:]
#             for p in parts:
#                 if p == "as":
#                     continue
#                 mods.append(p.split(".")[0])
#         elif s.startswith("from "):
#             mod = s.split()[1].split(".")[0]
#             mods.append(mod)
#     # dedupe, preserve order
#     seen = set(); out = []
#     for m in mods:
#         if m not in seen:
#             seen.add(m); out.append(m)
#     return out

def _import_check_ok(pycode: str, allow_extra: Optional[List[str]] = None) -> bool:
    """
    Dynamic mode:
    - Parse imports from `pycode`.
    - If any imports aren't yet in ALLOWED_IMPORTS (+ allow_extra), auto-extend ALLOWED_IMPORTS.
    - Always return True (no hard blocking).
    """
    imports = _extract_imports(pycode)

    allow = set(ALLOWED_IMPORTS)
    if allow_extra:
        allow.update(allow_extra)

    new_imports = sorted(set(imports) - allow)
    if new_imports:
        print(f"[INFO] Extending ALLOWED_IMPORTS with: {new_imports}")
        ALLOWED_IMPORTS.update(new_imports)  # mutate global allowlist

    return True



# def _import_check_ok(pycode: str, allow_extra: Optional[List[str]] = None) -> bool:
#     allow = set(ALLOWED_IMPORTS)
#     if allow_extra:
#         allow.update(allow_extra)
#     return set(_extract_imports(pycode)).issubset(allow)

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```") and s.endswith("```"):
        inner = s[3:-3].lstrip()
        if inner.lower().startswith("python\n"):
            inner = inner[len("python\n"):]
        return inner
    return s

def _ensure_result_variable(code: str) -> str:
    if "result" in code:
        return code
    return code + (
        "\n\n# Safety: ensure `result` exists\n"
        "try:\n"
        "    result\n"
        "except NameError:\n"
        "    import cadquery as cq\n"
        "    result = cq.Workplane('XY')\n"
    )

def _write_version(out_dir: Path, base: str, idx: int, code: str) -> Path:
    path = out_dir / f"{base}_v{idx}.py"
    path.write_text(code, encoding="utf-8")
    return path

def _promote_final(version_path: Path, final_path: Path) -> None:
    try:
        # On Windows, symlink needs admin; fallback to copy
        if final_path.exists():
            final_path.unlink()
        try:
            final_path.symlink_to(version_path)
        except Exception:
            final_path.write_text(version_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        # last resort: copy content
        final_path.write_text(version_path.read_text(encoding="utf-8"), encoding="utf-8")

# ========= Debugger Tool =========
@tool("debug_python")
def debug_python(code_path: str, debugger_script: str, *args: str) -> str:
    """Run an external Python debugger script against a code file.
    Args:
      code_path: Path to the generated Python file to analyze.
      debugger_script: Path to the external debugger script (e.g., tools/debugger.py).
      *args: Optional extra CLI args for the debugger.
    Returns combined stdout/stderr from the debugger with exit code.
    """
    code_path = str(code_path)
    debugger_script = str(debugger_script)
    if not Path(code_path).exists():
        return f"[debugger] ERROR: code file not found: {code_path}"
    if not Path(debugger_script).exists():
        return f"[debugger] ERROR: debugger script not found: {debugger_script}"
    try:
        proc = subprocess.run([sys.executable, debugger_script, code_path, *args], capture_output=True, text=True)
        out = proc.stdout or ""
        err = proc.stderr or ""
        return f"[debugger rc={proc.returncode}]\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    except Exception as e:
        return f"[debugger] Exception: {e}"

# ========= Agent builder =========
def build_executor_agent(model: str = "gpt-4o", temperature: float = 0.0, verbose: bool = False) -> AgentExecutor:
    llm = ChatOpenAI(model=model, temperature=temperature)
    tools = [debug_python]
    agent = create_tool_calling_agent(llm=llm, prompt=prompt_builder or prompt_builder_debug, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose)

# ========= Result model =========
@dataclass
class ExecResult:
    code_path: Path
    debug_report: Optional[str]
    attempts: int

# ========= High-level runner with autonomous debugging =========
def run_executor_from_plan(
    plan_json: str | Dict[str, Any],
    *,
    out_dir: Path,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    verbose: bool = False,
    allow_imports: Optional[List[str]] = None,
    debugger_script: Optional[str] = None,
    file_stem: Optional[str] = None,
    no_timestamp: bool = True,
    run_debugger_via_llm: bool = True,
    max_debug_iters: int = 3,
    pass_predicate: Optional[str] = r"\[debugger rc=0\]",
) -> ExecResult:
    """
    Generate CadQuery code from a validated CADSequencePlan JSON, then autonomously:
    - run the debugger tool,
    - read its report,
    - fix the code if needed,
    - repeat up to `max_debug_iters`.

    Returns ExecResult(final_code_path, last_debug_report, attempts).
    """
    # Normalize JSON -> schema
    if isinstance(plan_json, str):
        try:
            plan_dict = json.loads(plan_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid plan JSON: {e}")
    else:
        plan_dict = plan_json

    try:
        plan = CADSequencePlan(**plan_dict)
    except ValidationError as e:
        raise RuntimeError(f"Plan did not validate against CADSequencePlan: {e}")

    plan_pretty = json.dumps(plan.model_dump(), indent=2)

    # out paths
    out_dir.mkdir(parents=True, exist_ok=True)
    base = file_stem or plan.part_id or plan.design_id or "model"
    base = re.sub(r"[^a-zA-Z0-9_-]+", "-", base)
    final_path = out_dir / f"{base}.py"

    # Agent
    agent = build_executor_agent(model=model, temperature=temperature, verbose=verbose)

    # --- ROUND 0: build initial code ---
    raw = agent.invoke({"chat_history": [], "plan_json": plan_pretty})
    code = _strip_code_fences((raw.get("output") or "").strip())

    if "import cadquery as cq" not in code:
        code = "import cadquery as cq\n" + code
    code = _ensure_result_variable(code)

    if not _import_check_ok(code, allow_extra=allow_imports):
        raise RuntimeError("Generated code imports disallowed modules. If intentional, pass allow_imports.")

    version_path = _write_version(out_dir, base, idx=0, code=code)

    # If no debugger, just promote and return
    if not debugger_script:
        _promote_final(version_path, final_path)
        return ExecResult(code_path=final_path, debug_report=None, attempts=1)

    # --- Autonomous debug-fix loop ---
    last_report: Optional[str] = None
    attempts = 0
    for i in range(max_debug_iters + 1):  # include initial run
        attempts = i + 1

        if run_debugger_via_llm:
            # Ask the agent to USE the tool itself.
            dbg_cmd = (
                "Use the debug_python tool on the current file and return ONLY the debugger output.\n"
                f"code_path='{version_path.as_posix()}', "
                f"debugger_script='{Path(debugger_script).as_posix()}'."
            )
            tool_call = agent.invoke({
                "chat_history": [],
                "plan_json": plan_pretty,
                "agent_scratchpad": [],
                "input": dbg_cmd,
            })
            report = (tool_call.get("output") or "").strip()
        else:
            # Fallback: call tool directly (not autonomous)
            report = debug_python.func(
                code_path=version_path.as_posix(),
                debugger_script=Path(debugger_script).as_posix(),
            )

        last_report = report

        # Check pass condition
        passed = False
        if pass_predicate:
            try:
                passed = bool(re.search(pass_predicate, report))
            except Exception:
                passed = "[debugger rc=0]" in report
        else:
            passed = "[debugger rc=0]" in report

        if passed:
            # Promote this version as final and exit
            _promote_final(version_path, final_path)
            return ExecResult(code_path=final_path, debug_report=last_report, attempts=attempts)

        # If not last round, ask agent to FIX code using the debugger report
        if i < max_debug_iters:
            fix_prompt = ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT_FIX)]
            )
            llm = ChatOpenAI(model=model, temperature=temperature)
            fix_msgs = fix_prompt.format_messages(old_code=code, debug_report=report)
            fix_raw = llm.invoke(fix_msgs)
            new_code = _strip_code_fences((getattr(fix_raw, "content", "") or "").strip())

            if "import cadquery as cq" not in new_code:
                new_code = "import cadquery as cq\n" + new_code
            new_code = _ensure_result_variable(new_code)

            if not _import_check_ok(new_code, allow_extra=allow_imports):
                # If the fix tried to import something shady, keep the old code to avoid regressions
                # and continue to next iter (will likely fail again, but keeps guardrails tight).
                pass
            else:
                code = new_code
                version_path = _write_version(out_dir, base, idx=i+1, code=code)

    # If we got here, it never passed; promote the last attempt anyway so you can inspect it.
    _promote_final(version_path, final_path)
    return ExecResult(code_path=final_path, debug_report=last_report, attempts=attempts)

__all__ = [
    "CADOperation",
    "CADExportSpec",
    "CADSequencePlan",
    "build_executor_agent",
    "run_executor_from_plan",
    "ExecResult",
    "debug_python",
]
