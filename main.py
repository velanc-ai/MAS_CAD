# main.py
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from Agents.planner_agent import run_planner
from Agents.executor_agent import run_executor_from_plan

load_dotenv()


def query() -> str:
    return input("Enter your natural-language CAD query:\n> ").strip()

def save_text(text: str, path: Path, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"\n=== {label} saved ===")
    print(" ->", path.as_posix())

SAFE_STEM_RE = re.compile(r'[^A-Za-z0-9._-]+')

def sanitize_stem(name: str) -> str:
    cleaned = SAFE_STEM_RE.sub('_', name).strip('._-')
    return cleaned or "PART"

def next_available_stem(root: Path, desired: str) -> str:

    safe = sanitize_stem(desired)
    m = re.match(r'^(.*?)(\d+)$', safe)
    if m:
        prefix, n_str = m.groups()
        n, width = int(n_str), len(n_str)
    else:
        prefix, n, width = (safe.rstrip('_') + '_', 1, 3)

    candidate = f"{prefix}{n:0{width}d}"
    while (root / candidate).exists():
        n += 1
        candidate = f"{prefix}{n:0{width}d}"
    return candidate

if __name__ == "__main__":
    # === Paths ===
    OUTPUT_ROOT = Path(r"G:\MAS_CAD\OUTPUT")

    # === Your natural language CAD query ===
    query = query()                     # <-- runtime input
    if not query:
        raise ValueError("Empty query.")
    # === 1) Planner Agent → Plan JSON ===
    raw_json, plan = run_planner(query, save_dir=None, verbose=True)

    final_part_id = next_available_stem(OUTPUT_ROOT, plan.part_id)

    # Keep JSON and plan in sync with the chosen id
    try:
        data = json.loads(raw_json)
        data["part_id"] = final_part_id
        raw_json = json.dumps(data, indent=2)
    except Exception:
        pass
    try:
        plan.part_id = final_part_id
    except Exception:
        pass


    # Part-specific folder
    part_dir = OUTPUT_ROOT / plan.part_id
    part_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON (authoritative)
    json_path = part_dir / f"{plan.part_id}.json"
    save_text(raw_json, json_path, "JSON file")

    # === 2) Executor Agent (autonomous: generates + debugs code) ===
    try:
        exec_result = run_executor_from_plan(
            plan_json=raw_json,
            out_dir=part_dir,         # executor may or may not honor this; we still force-save below
            model="gpt-4o",
            temperature=0.0,
            verbose=True,
            file_stem=plan.part_id,   # predictable name (no timestamp)
            no_timestamp=True,
            max_debug_iters=3,
        )
    except Exception as e:
        print("\n=== Executor failed ===")
        print(repr(e))
        raise

    # === 2a) Force-save the generated .py next to the JSON ===
    final_py_path = part_dir / f"{plan.part_id}.py"

    code_str = getattr(exec_result, "code_str", None)
    reported_code_path = getattr(exec_result, "code_path", None)

    if code_str:
        # Best case: we got the code string directly
        save_text(code_str, final_py_path, "CadQuery .py file (forced)")
    elif reported_code_path:
        # Fallback: executor claims to have saved somewhere; copy into our target
        reported_code_path = Path(str(reported_code_path))
        if reported_code_path.exists():
            save_text(reported_code_path.read_text(encoding="utf-8"), final_py_path, "CadQuery .py file (copied)")
        else:
            print("\n[ERR] Executor returned code_path but file does not exist:")
            print(" ->", reported_code_path.as_posix())
            raise FileNotFoundError("Executor-reported code_path not found on disk.")
    else:
        print("\n[ERR] Executor returned neither code_str nor code_path.")
        # Dump attributes once for quick debugging
        try:
            print("exec_result attrs:", sorted(vars(exec_result).keys()))
        except Exception:
            pass
        raise RuntimeError("No code returned by executor.")

    # === 3) Summary Output ===
    print("\n=== Summary ===")
    print("JSON  ->", json_path.as_posix())
    print("CODE  ->", final_py_path.as_posix())

    # Attempts / debug info
    attempts = getattr(exec_result, "attempts", "n/a")
    print(f"\n=== Attempts === {attempts}")

    debug_report = getattr(exec_result, "debug_report", "")
    if debug_report:
        print("\n=== Debugger Report ===")
        print(debug_report)
    else:
        print("\n(no debugger report)")

    # List folder contents for sanity
    print("\n=== Folder contents ===")
    for p in sorted(part_dir.glob("*")):
        print(" -", p.name)



# if __name__ == "__main__":
#     OUTPUT_ROOT = Path(r"G:\MAS_CAD\OUTPUT")
#     DEBUGGER_SCRIPT = Path(r"G:\MAS_CAD\tools\debugger.py")
#
#     query = "The object is a flange with a diameter of 124mm and a thickness of 19mm, featuring a raised face with a diameter of 90mm and a height of 141mm. It has a bore diameter of 36mm."
#
#     # 1) Planner (don't save to root)
#     raw_json, plan = run_planner(query, save_dir=None, verbose=True)
#
#     # 2) Per-part folder
#     part_dir = OUTPUT_ROOT / plan.part_id
#     part_dir.mkdir(parents=True, exist_ok=True)
#
#     # Save JSON inside part_dir
#     json_path = part_dir / f"{plan.part_id}.json"
#     json_path.write_text(raw_json, encoding="utf-8")
#
#     # 3) Executor → CadQuery .py in same folder
#     try:
#         exec_result = run_executor_from_plan(
#             plan_json=raw_json,
#             out_dir=part_dir,
#             model="gpt-4o",
#             temperature=0.0,
#             verbose=True,
#             debugger_script=str(DEBUGGER_SCRIPT),
#             run_debugger_via_llm=True,
#             file_stem=plan.part_id,   # <- predictable base name
#             no_timestamp=True,        # <- no timestamp, easier to spot
#         )
#     except Exception as e:
#         print("\n=== Executor failed ===")
#         print(repr(e))
#         raise
#
#     # 4) Confirm and list
#     print("\n=== JSON file saved ===")
#     print(json_path.as_posix())
#
#     print("\n=== CadQuery file saved ===")
#     print(exec_result.code_path.as_posix())
#
#     if not exec_result.code_path.exists():
#         print("\n[ERR] Expected .py not found at path above — check executor logs.")
#     else:
#         print("\n=== Folder contents ===")
#         for p in sorted(part_dir.glob("*")):
#             print(" -", p.name)
#
#     if exec_result.debug_report:
#         print("\n=== Debugger Report ===")
#         print(exec_result.debug_report)
#     else:
#         print("\n(no debugger report)")
#


# if __name__ == "__main__":
#     PATH = "G:\\MAS_CAD\\OUTPUT"
#     query = "Design a simple mounting bracket with two holes, length 50 mm, width 20 mm, thickness 5 mm."
#     raw_json, plan = run_planner(query,save_dir = PATH , verbose=True)
#
#     print("\n=== Raw JSON ===")
#     print(raw_json)
#     print("\n=== Parsed Plan ===")
#     print(plan.model_dump_json(indent=2))

