import os
import re
import sys
import json
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# ---------- CORS: lets the grader's website call your API ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- The AI client (pointed at AI Pipe) ----------
client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)


# ---------- Data shapes ----------
class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ---------- Part 1: the Tool (runs the code) ----------
def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})  # run the user's code in a fresh, empty workspace
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except BaseException:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout


# ---------- Part 2: the AI Agent (finds the error line) ----------
def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    numbered_code = "\n".join(
        f"{i}: {line}" for i, line in enumerate(code.splitlines(), start=1)
    )
    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) in the USER'S CODE where the error occurred.
Use the line numbers shown at the start of each code line.
Ignore any traceback lines that are not from the user's code.

CODE (with line numbers):
{numbered_code}

TRACEBACK:
{tb}

Return only the line number(s) where the error is located."""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {
                            "type": "array",
                            "items": {"type": "integer"},
                        }
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False,
                },
            },
        },
    )
    result = ErrorAnalysis.model_validate_json(response.choices[0].message.content)
    return result.error_lines


# ---------- Safety net if the AI call ever fails ----------
def fallback_lines(tb: str) -> List[int]:
    matches = re.findall(r'File "<string>", line (\d+)', tb)
    return [int(matches[-1])] if matches else []


# ---------- The endpoint ----------
@app.post("/code-interpreter")
def code_interpreter(req: CodeRequest):
    run = execute_python_code(req.code)

    if run["success"]:
        return {"error": [], "result": run["output"]}

    try:
        lines = analyze_error_with_ai(req.code, run["output"])
        print("AI analysis worked:", lines)
    except Exception as e:
        print("AI FAILED:", e)
        lines = fallback_lines(run["output"])

    return {"error": lines, "result": run["output"]}