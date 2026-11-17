import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  #support to emojis
load_dotenv()

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite"

BASE_DIR = Path(__file__).resolve().parent    #current directory
WORKSPACE = BASE_DIR / "workspace"
TOOL_SCHEMAS = json.loads((BASE_DIR / "tools.json").read_text(encoding="utf-8")) #define what each tool can do
MAX_STEPS = 5


#tools

def list_files() -> str:
    #list the files in the workspace
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    #read a file from workspace
    path = WORKSPACE / filename
    if not path.is_file(): # error in case it doesnt find the file
        return f"error: no file named {filename}"
    return path.read_text(encoding="utf-8")


TOOLS = {"list_files": list_files, "read_file": read_file}

history = [
    {"role": "system", "content": "You are designed to help with anything asked."},
]

def run_tool(call) -> str:
    #run one tool call and return its result as text
    args = json.loads(call.function.arguments or "{}")
    print(f"  [tool] {call.function.name}({args})")
    return TOOLS[call.function.name](**args)

def chat(user_message: str) -> str:
    history.append({"role": "user", "content": user_message})

    #agentic loop: keep calling the model until it stops asking for tools
    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=history, tools=TOOL_SCHEMAS
        )
        message = response.choices[0].message
        history.append(message)

        #no tools requested
        if not message.tool_calls:
            return message.content or ""

        #run every tool it asked for and send the results back
        for call in message.tool_calls:
            history.append(
                {"role": "tool", "tool_call_id": call.id, "content": run_tool(call)}
            )

    return "too many steps"

#allow the user to make multiple questions
if __name__ == "__main__":
    print(f"assistant ({MODEL}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nyou: ")
            print("\nassistant:", chat(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
