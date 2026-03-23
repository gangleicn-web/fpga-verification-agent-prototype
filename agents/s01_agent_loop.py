#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Universal Agent Loop

The entire secret of an AI coding agent in one pattern:

    while True:
        text, tool_calls = llm.chat(messages, tools)
        if not tool_calls:
            break
        execute tools
        append results to messages

This version abstracts the LLM API to support Anthropic, DeepSeek, and Gemini,
demonstrating how different providers handle tool use (Function Calling).
Specially patched to preserve Gemini's `thought_signature` during message replays.
"""

import os
import json
import uuid
import subprocess
from dotenv import load_dotenv

load_dotenv(override=True)

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

def run_bash(command: str) -> str:
    dangerous =["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# =========================================================================
# LLM Provider Abstractions (Anthropic, DeepSeek, Gemini)
# =========================================================================

class AnthropicProvider:
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
        self.model = os.environ.get("MODEL_ID", "claude-3-5-sonnet-20241022")
        self.tools =[{
            "name": "bash",
            "description": "Run a shell command.",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }]

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
                        content.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["arguments"]
                        })
                anthropic_msgs.append({"role": "assistant", "content": content or "OK"})
            elif m["role"] == "tool":
                content =[]
                for res in m["content"]:
                    content.append({
                        "type": "tool_result",
                        "tool_use_id": res["tool_call_id"],
                        "content": res["content"]
                    })
                anthropic_msgs.append({"role": "user", "content": content})

        response = self.client.messages.create(
            model=self.model, system=SYSTEM,
            messages=anthropic_msgs, tools=self.tools, max_tokens=8000,
        )

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
        self.tools =[{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                }
            }
        }]

    def chat(self, messages: list):
        openai_msgs = [{"role": "system", "content": SYSTEM}]
        for m in messages:
            if m["role"] == "user":
                openai_msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] =[{
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}
                    } for tc in m["tool_calls"]]
                openai_msgs.append(msg)
            elif m["role"] == "tool":
                for res in m["content"]:
                    openai_msgs.append({
                        "role": "tool",
                        "tool_call_id": res["tool_call_id"],
                        "name": res["name"],
                        "content": res["content"]
                    })

        response = self.client.chat.completions.create(
            model=self.model, messages=openai_msgs, tools=self.tools
        )
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
        self.tools = [types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="bash",
                    description="Run a shell command.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"command": types.Schema(type=types.Type.STRING)},
                        required=["command"]
                    )
                )
            ]
        )]
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=self.tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

    def chat(self, messages: list):
        from google.genai import types
        gemini_msgs = []
        for m in messages:
            if m["role"] == "user":
                gemini_msgs.append(types.Content(role="user", parts=[types.Part.from_text(text=m["content"])]))
            elif m["role"] == "assistant":
                parts =[]
                if m.get("content"):
                    parts.append(types.Part.from_text(text=m["content"]))
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        # 【修复核心 1】：如果有保留下来的原始对象(包含 thought_signature)，原样塞回去
                        if tc.get("raw_part"):
                            parts.append(tc["raw_part"])
                        else:
                            parts.append(types.Part.from_function_call(
                                name=tc["name"],
                                args=tc["arguments"]
                            ))
                gemini_msgs.append(types.Content(role="model", parts=parts))
            elif m["role"] == "tool":
                parts = []
                for res in m["content"]:
                    part = types.Part.from_function_response(
                        name=res["name"],
                        response={"result": res["content"]}
                    )
                    if hasattr(part, "function_response") and hasattr(part.function_response, "id"):
                        part.function_response.id = res["tool_call_id"]
                    parts.append(part)
                gemini_msgs.append(types.Content(role="user", parts=parts))

        response = self.client.models.generate_content(
            model=self.model, contents=gemini_msgs, config=self.config
        )

        text = ""
        tool_calls = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text += part.text
                elif part.function_call:
                    class TC:
                        id = getattr(part.function_call, "id", uuid.uuid4().hex)
                        name = part.function_call.name
                        arguments = dict(part.function_call.args) if part.function_call.args else {}
                        # 【修复核心 2】：获取到工具调用请求时，将整个原始对象保存下来
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
# The Core Pattern: Universal Agent Loop
# =========================================================================

def agent_loop(messages: list, provider):
    while True:
        text, tool_calls = provider.chat(messages)
        
        # 【修复核心 3】：在通用字典中加入 "raw_part" 字段，以便将底层对象传递给下一轮循环
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
        for tc in tool_calls:
            print(f"\033[33m$ {tc.arguments.get('command', '')}\033[0m")
            output = run_bash(tc.arguments.get("command", ""))
            print(output[:200] + ("..." if len(output) > 200 else ""))
            
            results.append({
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": output
            })
            
        messages.append({"role": "tool", "content": results})


if __name__ == "__main__":
    # 1. 初始化 provider (通过环境变量自动决定是 Gemini, DeepSeek 还是 Anthropic)
    provider = get_provider()
    print(f"\033[35m[Using Provider: {provider.__class__.__name__}]\033[0m")
    print(f"Model ID: {provider.model}")
    
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
            
        history.append({"role": "user", "content": query})
        
        # 2. 调用时必须传入 provider 参数！
        agent_loop(history, provider)
        
        print()