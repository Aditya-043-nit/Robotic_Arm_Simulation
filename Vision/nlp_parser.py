from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langsmith import traceable
from pydantic import BaseModel, Field
from config import GROQ_API_KEY, LLM_MODEL


class RobotTask(BaseModel):
    action     : str = Field(description="The action to perform: put, pick, move, or place")
    object     : str = Field(description="The item to grab, e.g. Green-Apple, red ball")
    pickup_zone: str = Field(description="Where to pick from, e.g. table, shelf")
    drop_zone  : str = Field(description="Where to put it, e.g. fruit-basket, tray")


SYSTEM = """You are a command parser for a robot vision system.
Extract the task details from the operator's order and return ONLY valid JSON.

Return exactly this structure:
{{
  "action"     : "put" | "pick" | "move" | "place",
  "object"     : "<item to grab>",
  "pickup_zone": "<source location>",
  "drop_zone"  : "<destination location>"
}}

No explanation. No markdown. No extra keys. Just the JSON object."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "{order}"),
])


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0,
    max_tokens=150,
)

parser = JsonOutputParser(pydantic_object=RobotTask)


def validate_task(task: dict) -> dict:
    """Extra validation after JSON parsing — ensures no empty fields."""
    required = ["action", "object", "pickup_zone", "drop_zone"]
    missing = [f for f in required if not task.get(f, "").strip()]
    if missing:
        raise ValueError(f"Parsed task is missing fields: {missing}\nGot: {task}")
    return task

chain = prompt | llm | parser | RunnableLambda(validate_task)


@traceable(name="parse_order", tags=["nlp", "robot-vision"])
def parse_order(order: str) -> dict:
    """
    Parse a natural language robot order into a structured task dict.

    Input : "Put the Green-Apple from table to fruit-basket"
    Output: {
        "action"     : "put",
        "object"     : "Green-Apple",
        "pickup_zone": "table",
        "drop_zone"  : "fruit-basket"
    }

    Every call is traced in LangSmith at smith.langchain.com
    under project: robot-vision
    """
    return chain.invoke({"order": order})


# ── 7. Test runner ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_orders = [
        "Put the Green-Apple from table to fruit-basket",
        "Move the red cup from the shelf to the tray",
        "Pick the banana from the counter and place it in the bowl",
        "Take the bottle off the desk and drop it in the bin",
    ]

    print("Running test orders — check smith.langchain.com for traces\n")
    print(f"{'Order':<52} Result")
    print("-" * 80)

    for order in test_orders:
        try:
            result = parse_order(order)
            print(f"{order:<52} OK → object={result['object']!r}, drop={result['drop_zone']!r}")
        except Exception as e:
            print(f"{order:<52} ERROR: {e}")

    print("\nDone. All traces visible in LangSmith under project 'robot-vision'")