#!/usr/bin/env python3
# Harness: planning -- keeping the model on course without scripting the route.
"""
s03_todo_write.py - Universal Todo Write

The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets. This version integrates 
Multi-Provider support (Anthropic, DeepSeek, Gemini) and API logging.

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
"""

import os
import json
import uuid
import subprocess
import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# =========================================================================
# 全局配置 & 系统状态
# =========================================================================
WORKDIR = Path.cwd()
LOG_FILE = os.path.abspath("llm_api_trace.log")

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""

# -- TodoManager: structured state the LLM writes to --
class TodoManager:
    def __init__(self):
        self.items =[]

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated =[]
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

TODO = TodoManager()


# =========================================================================
# 本地工具执行函数 (Local Tool Functions)
# =========================================================================
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous =["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] +[f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
}


# =========================================================================
# 日志拦截器
# =========================================================================
def log_api_traffic(provider_name, request_data, response_data):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(obj):
        if hasattr(obj, "model_dump"): return obj.model_dump()
        if hasattr(obj, "to_dict"): return obj.to_dict()
        if hasattr(obj, "__dict__"): return obj.__dict__
        return str(obj)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"[{timestamp}] API CALL: {provider_name}\n")
        f.write(f"{'-'*70}\n")
        f.write(">>>[REQUEST PUSHED TO LLM]\n")
        f.write(json.dumps(request_data, indent=2, ensure_ascii=False, default=to_dict))
        f.write(f"\n\n<<<[RAW RESPONSE FROM LLM]\n")
        f.write(json.dumps(response_data, indent=2, ensure_ascii=False, default=to_dict))
        f.write("\n")


# =========================================================================
# LLM Provider Abstractions (Anthropic, DeepSeek, Gemini)
# =========================================================================
TODO_INPUT_SCHEMA_UNIVERSAL = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                },
                "required":["id", "text", "status"]
            }
        }
    },
    "required": ["items"]
}

class AnthropicProvider:
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
        self.model = os.environ.get("MODEL_ID", "claude-3-5-sonnet-20241022")
        self.tools =[
            {"name": "bash", "description": "Run a shell command.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required":["path"]}},
            {"name": "write_file", "description": "Write content to file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required":["path", "old_text", "new_text"]}},
            {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.", "input_schema": TODO_INPUT_SCHEMA_UNIVERSAL},
        ]

    def chat(self, messages: list):
        anthropic_msgs =[]
        for m in messages:
            if m["role"] == "user":
                anthropic_msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                content =[]
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
                anthropic_msgs.append({"role": "assistant", "content": content or "OK"})
            elif m["role"] == "tool":
                content =[]
                # Anthropic 允许在 tool 结果前塞入一个 system nag reminder text block
                if m.get("reminder"):
                    content.append({"type": "text", "text": m["reminder"]})
                for res in m["content"]:
                    content.append({"type": "tool_result", "tool_use_id": res["tool_call_id"], "content": res["content"]})
                anthropic_msgs.append({"role": "user", "content": content})

        req_payload = {"model": self.model, "system": SYSTEM, "messages": anthropic_msgs, "tools": self.tools, "max_tokens": 8000}
        response = self.client.messages.create(**req_payload)
        log_api_traffic("Anthropic", req_payload, response)

        text = ""
        tool_calls =[]
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                class TC:
                    id = block.id
                    name = block.name
                    arguments = block.input
                tool_calls.append(TC())
        return text, tool_calls


class DeepSeekProvider:
    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.environ.get("MODEL_ID", "deepseek-chat")
        self.tools =[
            {"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "read_file", "description": "Read file contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write content to file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required":["path", "content"]}}},
            {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required":["path", "old_text", "new_text"]}}},
            {"type": "function", "function": {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.", "parameters": TODO_INPUT_SCHEMA_UNIVERSAL}},
        ]

    def chat(self, messages: list):
        openai_msgs = [{"role": "system", "content": SYSTEM}]
        for m in messages:
            if m["role"] == "user":
                openai_msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}} for tc in m["tool_calls"]]
                openai_msgs.append(msg)
            elif m["role"] == "tool":
                for res in m["content"]:
                    openai_msgs.append({"role": "tool", "tool_call_id": res["tool_call_id"], "name": res["name"], "content": res["content"]})
                # OpenAI 规范下，追加一个单独的 user message 作为 Nag Reminder
                if m.get("reminder"):
                    openai_msgs.append({"role": "user", "content": m["reminder"]})

        req_payload = {"model": self.model, "messages": openai_msgs, "tools": self.tools}
        response = self.client.chat.completions.create(**req_payload)
        log_api_traffic("DeepSeek/OpenAI", req_payload, response)

        msg = response.choices[0].message
        text = msg.content or ""
        tool_calls =[]
        if msg.tool_calls:
            for tc in msg.tool_calls:
                class TC:
                    id = tc.id
                    name = tc.function.name
                    arguments = json.loads(tc.function.arguments)
                tool_calls.append(TC())
        return text, tool_calls


class GeminiProvider:
    def __init__(self):
        from google import genai
        from google.genai import types
        self.client = genai.Client()
        self.model = os.environ.get("MODEL_ID", "gemini-3.1-pro-preview")
        self.tools =[types.Tool(
            function_declarations=[
                types.FunctionDeclaration(name="bash", description="Run a shell command.", parameters=types.Schema(type=types.Type.OBJECT, properties={"command": types.Schema(type=types.Type.STRING)}, required=["command"])),
                types.FunctionDeclaration(name="read_file", description="Read file contents.", parameters=types.Schema(type=types.Type.OBJECT, properties={"path": types.Schema(type=types.Type.STRING), "limit": types.Schema(type=types.Type.INTEGER)}, required=["path"])),
                types.FunctionDeclaration(name="write_file", description="Write content to file.", parameters=types.Schema(type=types.Type.OBJECT, properties={"path": types.Schema(type=types.Type.STRING), "content": types.Schema(type=types.Type.STRING)}, required=["path", "content"])),
                types.FunctionDeclaration(name="edit_file", description="Replace exact text in file.", parameters=types.Schema(type=types.Type.OBJECT, properties={"path": types.Schema(type=types.Type.STRING), "old_text": types.Schema(type=types.Type.STRING), "new_text": types.Schema(type=types.Type.STRING)}, required=["path", "old_text", "new_text"])),
                types.FunctionDeclaration(
                    name="todo",
                    description="Update task list. Track progress on multi-step tasks.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "items": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(
                                    type=types.Type.OBJECT,
                                    properties={
                                        "id": types.Schema(type=types.Type.STRING),
                                        "text": types.Schema(type=types.Type.STRING),
                                        "status": types.Schema(type=types.Type.STRING)
                                    },
                                    required=["id", "text", "status"]
                                )
                            )
                        },
                        required=["items"]
                    )
                )
            ]
        )]
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM, tools=self.tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

    def chat(self, messages: list):
        from google.genai import types
        gemini_msgs =[]
        for m in messages:
            if m["role"] == "user":
                gemini_msgs.append(types.Content(role="user", parts=[types.Part.from_text(text=m["content"])]))
            elif m["role"] == "assistant":
                parts =[]
                if m.get("content"):
                    parts.append(types.Part.from_text(text=m["content"]))
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        if tc.get("raw_part"):
                            parts.append(tc["raw_part"])
                        else:
                            parts.append(types.Part.from_function_call(name=tc["name"], args=tc["arguments"]))
                gemini_msgs.append(types.Content(role="model", parts=parts))
            elif m["role"] == "tool":
                parts =[]
                # Gemini 允许在 function_response 同一轮的 user parts 里追加 text 作为 Reminder
                if m.get("reminder"):
                    parts.append(types.Part.from_text(text=m["reminder"]))
                for res in m["content"]:
                    part = types.Part.from_function_response(name=res["name"], response={"result": res["content"]})
                    if hasattr(part, "function_response") and hasattr(part.function_response, "id"):
                        part.function_response.id = res["tool_call_id"]
                    parts.append(part)
                gemini_msgs.append(types.Content(role="user", parts=parts))

        req_payload = {"model": self.model, "contents": gemini_msgs, "config": self.config}
        response = self.client.models.generate_content(model=self.model, contents=gemini_msgs, config=self.config)
        log_api_traffic("Gemini", req_payload, response)

        text = ""
        tool_calls =[]
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text += part.text
                elif part.function_call:
                    class TC:
                        id = getattr(part.function_call, "id", uuid.uuid4().hex)
                        name = part.function_call.name
                        arguments = dict(part.function_call.args) if part.function_call.args else {}
                        raw_part = part 
                    tool_calls.append(TC())
        return text, tool_calls


def get_provider():
    provider_name = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider_name == "deepseek":
        return DeepSeekProvider()
    elif provider_name == "gemini":
        return GeminiProvider()
    else:
        return AnthropicProvider()


# =========================================================================
# The Core Pattern: Universal Agent Loop with Nag Reminder
# =========================================================================

def agent_loop(messages: list, provider):
    rounds_since_todo = 0
    while True:
        text, tool_calls = provider.chat(messages)
        
        messages.append({
            "role": "assistant",
            "content": text,
            "tool_calls":[{
                "id": tc.id, 
                "name": tc.name, 
                "arguments": tc.arguments,
                "raw_part": getattr(tc, "raw_part", None)
            } for tc in tool_calls] if tool_calls else None
        })

        if text:
            print(f"\033[32mAgent:\033[0m {text}")

        if not tool_calls:
            return

        results = []
        used_todo = False
        
        for tc in tool_calls:
            if tc.name == "todo":
                used_todo = True
                
            handler = TOOL_HANDLERS.get(tc.name)
            if handler:
                try:
                    output = handler(**tc.arguments)
                    print(f"\033[33m> {tc.name}:\033[0m {str(output)[:200]}")
                except Exception as e:
                    output = f"Error executing {tc.name}: {str(e)}"
                    print(f"\033[31m> {tc.name} Error:\033[0m {output}")
            else:
                output = f"Unknown tool: {tc.name}"
                print(f"\033[31m> Error:\033[0m {output}")
            
            results.append({
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": str(output)
            })
            
        # --- Nag Reminder 核心防遗忘机制 ---
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        reminder = "<reminder>Update your todos.</reminder>" if rounds_since_todo >= 3 else None
        
        # 将 reminder 作为一个专门的字段传入，各家 Provider 会按自己的规范组装
        messages.append({"role": "tool", "content": results, "reminder": reminder})


if __name__ == "__main__":
    provider = get_provider()
    print(f"\033[35m[Using Provider: {provider.__class__.__name__}]\033[0m")
    print(f"Model ID: {provider.model}")
    print(f"\033[90m(API Traffic will be logged to: {LOG_FILE})\033[0m")
    
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
            
        # 每次收到新任务前，清空日志文件
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== New Round Started: {query} ===\n")
            
        history.append({"role": "user", "content": query})
        agent_loop(history, provider)
        
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()