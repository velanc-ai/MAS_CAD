# prompt_generator_openai.py
from __future__ import annotations
import os, uuid, argparse, json
from datetime import datetime
from typing import List, Optional, Literal
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
# pip install -U openai pandas pydantic python-dotenv
from openai import OpenAI, BadRequestError

load_dotenv()

# =========================
# Pydantic Schemas
# =========================
class GeneratedPrompt(BaseModel):
    id: str = Field(description="Unique id for the prompt (uuid).")
    domain: str = Field(description="High-level domain (e.g., 'electronics', 'CAD').")
    subtype: Optional[str] = Field(
        default=None,
        description="Specific subtype/category (e.g., 'arduino_case', 'enclosure', 'mounting_plate').",
    )
    prompt: str = Field(description="MTurk-style natural-language description with mm units.")

class PromptBatch(BaseModel):
    query: str
    count: int
    prompts: List[GeneratedPrompt]

# =========================
# System Instructions
# =========================
SYSTEM_INSTRUCTIONS = """
You are a domain-specific prompt generator.

Goal:
Produce Amazon Mechanical Turk–style technical descriptions for **Arduino cases / electronics enclosures**
that are realistic, dimensioned (mm), and immediately usable for CAD / rapid prototyping.

Rules:
- Return EXACTLY the requested number of prompts.
- One self-contained description per prompt. No bullets, no lists, no commentary.
- Use clear dimensions in millimeters (mm): external L×W×H, wall thickness, standoffs, hole diameters, port cutouts,
  vent slots, lid/fastener style (snap-fit, M3 screws), and tolerances/clearances where relevant.
- Prefer manufacturable, simple geometry (FDM/laser/CNC friendly) when 'rapid prototyping' is implied.
- Vary vocabulary and include realistic variants: Arduino Uno/Nano/Mega, cable strain reliefs, removable lids,
  countersunk holes, captive nuts, heat-set inserts, fillets/chamfers, ribs, living hinges (if appropriate).
- Avoid contradictions (e.g., negative sizes, impossible fits). Keep dimensions plausible.
- Use subtype like 'arduino_case', 'uno_enclosure', 'nano_case', 'mega_enclosure', or 'proto_shield_box' where helpful.
- Follow the number of prompt count required by user very strictly.
Output:
Return ONLY valid JSON matching this schema:

PromptBatch = {
  "query": str,
  "count": int,
  "prompts": [
    {
      "id": str,              // uuid
      "domain": str,          // e.g. "electronics"
      "subtype": str | null,  // e.g. "arduino_case"
      "prompt": str           // the MTurk-style description (mm)
    }, ...
  ]
}
"""

# =========================
# OpenAI Calls
# =========================

def call_openai_for_batch(client: OpenAI, model: str, user_query: str, count: int) -> str:
    req = {
        "model": model,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user",
             "content": (
                 f"Generate exactly {count} unique, realistic Amazon Mechanical Turk-style technical descriptions "
                 f"for {user_query}. Each prompt must be a single paragraph using millimeter dimensions, "
                 f"include plausible manufacturable details (wall thickness, screws, holes, vents, etc.), "
                 f"and avoid any markdown, bullets, lists, or commentary. "
                 f"DO NOT generate fewer than {count} prompts under any circumstances. "
                 f"Return ONLY a valid JSON object matching the PromptBatch schema."
             )}
        ],
        "response_format": {"type": "json_object"},
    }
    try:
        resp = client.chat.completions.create(**req)
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e

    if not resp.choices:
        raise RuntimeError("No response choices returned from OpenAI.")

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Empty response from OpenAI.")

    return content.strip()


def repair_json_if_needed(client: OpenAI, model: str, broken: str) -> str:
    """Ask the model to repair malformed JSON to the target schema."""
    req = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You repair malformed JSON to exactly match the target schema. Output ONLY JSON."},
            {"role": "user", "content": (
                "Target key structure:\n"
                '{"query": str, "count": int, "prompts": [{"id": str, "domain": str, "subtype": str | null, "prompt": str}]}\n\n'
                f"Broken JSON:\n{broken}"
            )}
        ],
    }
    try:
        resp = client.chat.completions.create(**req)
    except Exception as e:
        raise RuntimeError(f"OpenAI API error during JSON repair: {e}") from e

    if not resp.choices:
        raise RuntimeError("No repair response from OpenAI.")

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Empty repair response from OpenAI.")

    return content.strip()


# =========================
# Core Generation
# =========================
def generate_prompts_openai(
    user_query: str,
    count: int = 20,
    model: str = "gpt-4o",
) -> PromptBatch:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    raw_text = call_openai_for_batch(client, model, user_query, count)

    def parse_to_batch(text: str) -> PromptBatch:
        data = json.loads(text)
        # Ensure IDs exist and default domain/subtype if missing
        for p in data.get("prompts", []):
            if not p.get("id"):
                p["id"] = str(uuid.uuid4())
            if not p.get("domain"):
                p["domain"] = "electronics"
            if p.get("subtype") is None:
                p["subtype"] = "arduino_case"
        return PromptBatch(**data)

    try:
        batch = parse_to_batch(raw_text)
    except (json.JSONDecodeError, ValidationError):
        print("Malformed JSON received. Attempting to repair...")
        fixed_text = repair_json_if_needed(client, model, raw_text)
        batch = parse_to_batch(fixed_text)

    # Final guards: enforce exact count
    if len(batch.prompts) != count:
        print(f"Expected {count} prompts, got {len(batch.prompts)}. Adjusting...")
        batch.prompts = batch.prompts[:count]  # Truncate if too many
        while len(batch.prompts) < count:
            batch.prompts.append(
                GeneratedPrompt(
                    id=str(uuid.uuid4()),
                    domain="electronics",
                    subtype="arduino_case",
                    prompt=f"Placeholder prompt #{len(batch.prompts)+1} (generated due to insufficient output).",
                )
            )

    batch.count = count
    if not batch.query.strip():
        batch.query = user_query

    return batch


# =========================
# Export Helpers
# =========================
def export_prompts(
    batch: PromptBatch,
    export: Literal["csv", "xlsx"] = "csv",
    out_path: str | None = None
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_path is None:
        out_path = f"prompts_{ts}.{export}"

    rows = [{
        "id": p.id,
        "domain": p.domain,
        "subtype": p.subtype,
        "prompt": p.prompt,
        "source_query": batch.query,
    } for p in batch.prompts]

    df = pd.DataFrame(rows)
    if export == "csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_excel(out_path, index=False)
    return out_path


# =========================
# CLI
# =========================
def run_once(
    query: str,
    count: int,
    model: str,
    export: Literal["csv", "xlsx"],
    out_path: Optional[str],
    no_save: bool,
) -> None:
    batch = generate_prompts_openai(query, count=count, model=model)

    print(f"\nGenerated {len(batch.prompts)} prompts for: {batch.query}\n")
    for i, p in enumerate(batch.prompts[:min(3, len(batch.prompts))], start=1):
        preview = (p.prompt[:140] + "…") if len(p.prompt) > 140 else p.prompt
        print(f"[{i}] ({p.subtype or p.domain}) {preview}")
    if len(batch.prompts) > 3:
        print("...")

    if not no_save and count > 1:
        path = export_prompts(batch, export=export, out_path=out_path)
        print(f"\nSaved to {path}")
    elif count == 1:
        print("\n📋 Full JSON:\n", batch.model_dump_json(indent=2))


def main():
    parser = argparse.ArgumentParser(description="OpenAI-powered MTurk-style prompt generator for Arduino cases / enclosures")
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural-language request (e.g., 'generate 30 MTurk-style prompts for an Arduino case, rapid prototyping')",
    )
    parser.add_argument("--count", type=int, default= 20, help="How many prompts to generate")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--export", choices=["csv", "xlsx"], default="csv", help="File format (used if count>1)")
    parser.add_argument("--out", type=str, default=None, help="Explicit output filename (e.g., prompts.csv or prompts.xlsx)")
    parser.add_argument("--no_save", action="store_true", help="Do not write a file even if count>1")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive loop to run multiple batches")
    args = parser.parse_args()

    if args.interactive:
        print("Interactive mode: press Enter on query to exit.\n")
        while True:
            q = args.query or input("Enter user query (blank to quit): ").strip()
            if not q:
                break
            try:
                c = args.count or int(input("How many prompts? [default 20]: ") or "10")
            except ValueError:
                c = 20
            exp = (input("Export format [csv/xlsx, default csv]: ").strip().lower() or args.export).replace(".", "")
            if exp not in {"csv", "xlsx"}:
                exp = "csv"
            outp = input("Output filename (optional, include .csv or .xlsx): ").strip() or args.out
            run_once(q, c, args.model, exp, outp, args.no_save)
            print("\n— done —\n")
            args.query = None
    else:
        q = args.query or input("Enter user query: ").strip()
        run_once(q, args.count, args.model, args.export, args.out, args.no_save)


if __name__ == "__main__":
    main()