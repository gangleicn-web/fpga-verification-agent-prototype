#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.

import os
import json
import uuid
import subprocess
import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

# =========================================================================
# 全局配置
# =========================================================================
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."
# 统一定义全局日志文件路径
LOG_FILE = os.path.abspath("llm_api_trace.log")

# =========================================================================
# 本地工具执行函数
# =========================================================================
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def install_python_package(package_name: str) -> str:
    print(f"\033[34m[System] 正在安全安装 Python 包: {package_name}...\033[0m")
    if any(c in package_name for c in [";", "&", "|", ">", "<"]):
        return "Error: Invalid package name. Injection detected."
    try:
        command = f"python3 -m pip install {package_name}"
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            return f"Success: {package_name} installed.\n{r.stdout[-500:]}"
        else:
            return f"Failed to install {package_name}.\nError: {r.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

# =========================================================================
# 日志拦截器
# =========================================================================
def log_api_traffic(provider_name, request_data, response_data):
    """记录 API 请求与返回的原始数据到文件。使用 'a' 追加模式。"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(obj):
        if hasattr(obj, "model_dump"): return obj.model_dump()
        if hasattr(obj, "to_dict"): return obj.to_dict()
        if hasattr(obj, "__dict__"): return obj.__dict__
        return str(obj)

    # 追加模式 ('a')：如果文件不存在会自动创建
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"[{timestamp}] API CALL: {provider_name}\n")
        f.write(f"{'-'*70}\n")
        f.write(">>>[REQUEST PUSHED TO LLM]\n")
        f.write(json.dumps(request_data, indent=2, ensure_ascii=False, default=to_dict))
        f.write(f"\n\n<<< [RAW RESPONSE FROM LLM]\n")
        f.write(json.dumps(response_data, indent=2, ensure_ascii=False, default=to_dict))
        f.write("\n")

# =========================================================================
# LLM Provider Abstractions
# =========================================================================

class AnthropicProvider:
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
        self.model = os.environ.get("MODEL_ID", "claude-3-5-sonnet-20241022")
        self.tools =[
            {
                "name": "bash",
                "description": "Run a shell command.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
            {
                "name": "install_python_package",
                "description": "专门用于安装 Python 第三方依赖包（通过 pip）。当你遇到 ModuleNotFoundError 或需要使用新库前，请调用此工具。",
                "input_schema": {
                    "type": "object",
                    "properties": {"package_name": {"type": "string"}},
                    "required": ["package_name"],
                },
            }
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

        req_payload = {
            "model": self.model, "system": SYSTEM,
            "messages": anthropic_msgs, "tools": self.tools, "max_tokens": 8000
        }
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
            {
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
            },
            {
                "type": "function",
                "function": {
                    "name": "install_python_package",
                    "description": "专门用于安装 Python 第三方依赖包（通过 pip）。当你遇到 ModuleNotFoundError 或需要使用新库前，请调用此工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {"package_name": {"type": "string"}},
                        "required": ["package_name"],
                    }
                }
            }
        ]

    def chat(self, messages: list):
        openai_msgs =[{"role": "system", "content": SYSTEM}]
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
                types.FunctionDeclaration(
                    name="bash",
                    description="Run a shell command.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"command": types.Schema(type=types.Type.STRING)},
                        required=["command"]
                    )
                ),
                types.FunctionDeclaration(
                    name="install_python_package",
                    description="专门用于安装 Python 第三方依赖包（通过 pip）。当你遇到 ModuleNotFoundError 或需要使用新库前，请调用此工具。",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"package_name": types.Schema(type=types.Type.STRING)},
                        required=["package_name"]
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
                            parts.append(types.Part.from_function_call(
                                name=tc["name"], args=tc["arguments"]
                            ))
                gemini_msgs.append(types.Content(role="model", parts=parts))
            elif m["role"] == "tool":
                parts =[]
                for res in m["content"]:
                    part = types.Part.from_function_response(
                        name=res["name"],
                        response={"result": res["content"]}
                    )
                    if hasattr(part, "function_response") and hasattr(part.function_response, "id"):
                        part.function_response.id = res["tool_call_id"]
                    parts.append(part)
                gemini_msgs.append(types.Content(role="user", parts=parts))

        req_payload = {"model": self.model, "contents": gemini_msgs, "config": self.config}
        response = self.client.models.generate_content(
            model=self.model, contents=gemini_msgs, config=self.config
        )
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
# The Core Pattern: Universal Agent Loop
# =========================================================================

def agent_loop(messages: list, provider):
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
        for tc in tool_calls:
            # ========== 工具路由 (Tool Router) ==========
            if tc.name == "bash":
                print(f"\033[33m$ {tc.arguments.get('command', '')}\033[0m")
                output = run_bash(tc.arguments.get("command", ""))
                print(output[:2000] + ("..." if len(output) > 2000 else ""))
            elif tc.name == "install_python_package":
                pkg = tc.arguments.get('package_name', '')
                output = install_python_package(pkg)
                print(output)
            else:
                output = f"Error: Unknown tool {tc.name}"
            # ============================================
            
            results.append({
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": output
            })
            
        messages.append({"role": "tool", "content": results})


if __name__ == "__main__":
    provider = get_provider()
    print(f"\033[35m[Using Provider: {provider.__class__.__name__}]\033[0m")
    print(f"Model ID: {provider.model}")
    print(f"\033[90m(API Traffic will be logged to: {LOG_FILE})\033[0m")
    
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
            
        # =========================================================
        # [修改点] 每次输入新任务时，清空之前的日志文件。
        # 'w' 模式如果文件不存在会自动创建，如果存在则将其截断清空。
        # =========================================================
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== New Round Started: {query} ===\n")
            
        history.append({"role": "user", "content": query})
        agent_loop(history, provider)
        print()