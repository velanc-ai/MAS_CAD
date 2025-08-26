# Agents/planner_agent.py
from __future__ import annotations
import os
from typing import List, Dict, Literal, Any, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import research  # make sure tools/__init__.py exists and tools/research.py exports the three tools


# ========= Pydantic schema (all fields present, no nulls) =========
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


def _build_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    """
    Builds the chat prompt that forces a strict CAD JSON plan matching CADSequencePlan.
    """
    system_text = """
        Act as a senior CAD sequence planner by converting the user's request into a deterministic, dependency-ordered
        domain-specific 3D CAD generation plan using the DEEPCAD CAD JSON Sequence as a reference.
        
        CRITICAL RESEARCH REQUIREMENT:
        - FIRST, run at least one research tool call (e.g., `search` or `wiki_tool`) to gather:
          • Material properties relevant to the user's intended manufacturing process and printable volume within the 3d printer
            (e.g., Rapid prototyping or 3D Printing using the affordable manufacturing techniques like FDM and so on...)
          • Design for Additive Manufacturing (DfAM) constraints: 
                minimum wall thickness, hole diameters, overhang/bridge rules, lattice/gyroid guidance, 
                recommended tolerances, post-processing allowances, and orientation notes for the selected process.
        - Summarize the findings concisely and persist them by calling `save_text_to_file`.
        - Then generate the CAD plan.
        
        OUTPUT CONTRACT:
        - Be specific with dimensions and parameters when provided; 
        - otherwise choose sensible defaults and record them in global_params.
        - You MUST include EVERY field defined by the schema in the output
          (part_id - example format {{MB_0001}}, design_id - Example format {{FDM_001}}, 
          intent, units, global_params, constraints, materials, tolerance, coordinate_system, steps, export).
        - Do NOT omit any schema keys.
        - Do NOT output null for any field; if unknown, choose a sensible default.
        - The coordinate_system MUST use exactly these keys: origin [x,y,z], axes ['+X','+Y','+Z'], plane 'XY'.
        - Only output JSON matching this schema and nothing else:
        {format_instructions}
        
        - Below is just a format and you should use only the researched data fetched from 
        the research tool you should not use the same data below without doing any research
        
        HOW TO PLACE RESEARCH INSIDE THE SCHEMA (without changing the schema):
        - Put a compact research digest under global_params.research_summary (string).
        - Add a structured object under global_params.manufacturing_constraints with keys like:
          {{
            "process": "SLS",
            "min_wall_thickness": "1.2 mm",
            "min_hole_diameter": "2.0 mm",
            "overhang_guideline": "Self-supporting above 45°; use fillets/chamfers",
            "tolerance_guideline": "±0.2 mm or ±0.2% (whichever is larger)",
            "orientation_notes": "Orient to reduce Z stair-stepping on critical faces"
          }}
        - Add a structured object under global_params.material_properties with keys like:
          {{
            "material": "PA12",
            "tensile_strength": "48 MPa",
            "elongation_at_break": "20%",
            "heat_deflection_temp": "95 °C",
            "density": "1.01 g/cc"
          }}
        - If you used sources, include a short list of URLs/titles under global_params.sources (list of strings).
        
        PRIORITIES:
        - Prioritize the accuracy of the 3D models and material/process details for additive manufacturing.
        - Use part_id as the .json filename (without extension).
        - Ensure it is a properly formatted single JSON object.
        - If the user did not specify a process, infer a plausible AM process 
        and state it in manufacturing_constraints.process.
        
        Below is an example (ONLY format—do not copy values):
        {{
          "part_id": "SRM_0000",
          "design_id": "end_effector_claw_half",
          "intent": "why this design is required",
          "units": "mm",
          "global_params": {{
            "research_summary": "Concise notes from tools about SLS PA12 and wall thickness.",
            "manufacturing_constraints": {{
              "process": "SLS",
              "min_wall_thickness": "1.2 mm",
              "min_hole_diameter": "2.0 mm",
              "overhang_guideline": "Self-supporting above 45°; chamfer 45°",
              "tolerance_guideline": "±0.2 mm or ±0.2%",
              "orientation_notes": "Critical faces aligned to XY"
            }},
            "material_properties": {{
              "material": "PA12",
              "tensile_strength": "48 MPa",
              "elongation_at_break": "20%",
              "heat_deflection_temp": "95 °C",
              "density": "1.01 g/cc"
            }}
          }},
          "constraints": ["Engineering constraints go here..."],
          "materials": {{"default": "PA12 (SLS)"}},
          "tolerance": "±0.1 mm",
          "coordinate_system": {{"origin": [0,0,0], "axes": ["+X","+Y","+Z"], "plane": "XY"}},
          "steps": [],
          "export": {{"format":"STL","mesh_tolerance":"medium","units":"mm"}}
        }}
        
        """.strip()

    return (
        ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                MessagesPlaceholder("chat_history"),
                ("human", "{query}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
    )


def build_agent(
    model: str = "gpt-4o",
    temperature: float = 0.0,
    response_format_json: bool = True,
    verbose: bool = False,
) -> AgentExecutor:
    """
    Construct and return an AgentExecutor that generates CAD plans.
    NOTE: Ensure OPENAI_API_KEY is available in the environment (loaded by your app).
    """
    parser = PydanticOutputParser(pydantic_object=CADSequencePlan)
    prompt = _build_prompt(parser)

    model_kwargs: Dict[str, Any] = {}
    if response_format_json:
        # Ask the model to emit a single JSON object
        model_kwargs["response_format"] = {"type": "json_object"}

    llm = ChatOpenAI(model=model, temperature=temperature, model_kwargs=model_kwargs)

    # Register research tools
    tools = [
        research.search_tool,
        research.wiki_tool,
        research.save_tool,
    ]

    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    planner_agent = AgentExecutor(agent=agent, tools=tools, verbose=verbose)

    # Expose parser to callers (internal convenience)
    planner_agent._parser = parser  # type: ignore[attr-defined]
    return planner_agent


# ========= Utilities =========
def robust_strip_code_fences(text: str) -> str:
    """
    Removes leading/trailing ``` blocks and an optional leading 'json' language tag.
    Safe to run on already-clean JSON strings.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    if s.endswith("```"):
        s = s[:-3].rstrip()
    return s


def run_planner(
    query: str,
    chat_history: Optional[list] = None,
    *,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    response_format_json: bool = True,
    verbose: bool = False,
    save_dir: str = None,
) -> Tuple[str, CADSequencePlan]:
    """
    High-level helper:
      - builds the agent
      - invokes it with the query
      - strips accidental Markdown fences
      - validates into CADSequencePlan (fills defaults)
    Returns: (raw_json_string, parsed_plan)
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Load your .env before calling run_planner()."
        )

    agent = build_agent(
        model=model,
        temperature=temperature,
        response_format_json=response_format_json,
        verbose=verbose,
    )
    parser: PydanticOutputParser = agent._parser  # type: ignore[attr-defined]

    payload = {
        "query": query,
        "chat_history": chat_history or [],
    }
    raw = agent.invoke(payload)
    raw_json = (raw.get("output") or "").strip()
    if not raw_json:
        raise RuntimeError("Model returned an empty response.")

    raw_json = robust_strip_code_fences(raw_json)

    try:
        plan = parser.parse(raw_json)
    except Exception as e:
        snippet = raw_json[:2000] + ("..." if len(raw_json) > 2000 else "")
        raise RuntimeError(
            f"Error parsing model output into CADSequencePlan: {e}\n\nRaw snippet:\n{snippet}"
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{plan.part_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(raw_json)
        if verbose:
            print(f"✅ Saved CAD plan to {file_path}")
    return raw_json, plan


__all__ = [
    "CADOperation",
    "CADExportSpec",
    "CADSequencePlan",
    "build_agent",
    "run_planner",
    "robust_strip_code_fences",
]
