"""MCP Inspector and Host Integration verification script over STDIO JSON-RPC 2.0."""

import json
import subprocess
import sys
from pathlib import Path

def run_inspector_test() -> None:
    """Send standard MCP JSON-RPC 2.0 messages to server over STDIO and assert responses."""
    print("================================================================================")
    print("RUNNING MCP PROTOCOL & INSPECTOR INTEGRATION VERIFICATION OVER STDIO")
    print("================================================================================")

    server_process = subprocess.Popen(
        [sys.executable, "-u", "server.py", "--transport", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # 1. Initialize Request
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "MCP-Inspector-Test-Client", "version": "1.0.0"},
            },
        }

        server_process.stdin.write(json.dumps(init_req) + "\n")
        server_process.stdin.flush()
        line = server_process.stdout.readline()
        assert line.strip(), "No response received for initialize"
        init_resp = json.loads(line)
        print("[PASS] JSON-RPC 2.0 Initialize Response:", init_resp.get("result", {}).get("serverInfo"))

        # Send notifications/initialized
        init_notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        server_process.stdin.write(json.dumps(init_notif) + "\n")
        server_process.stdin.flush()

        # 2. List Tools Request
        tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        server_process.stdin.write(json.dumps(tools_req) + "\n")
        server_process.stdin.flush()
        line = server_process.stdout.readline()
        tools_resp = json.loads(line)
        tools = tools_resp.get("result", {}).get("tools", [])
        tool_names = [t["name"] for t in tools]
        print(f"[PASS] Tools Discovered ({len(tool_names)}):", tool_names)
        assert len(tool_names) == 12, f"Expected 12 tools, got {len(tool_names)}"

        # 3. Call Tool Request (search_books)
        call_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_books",
                "arguments": {"query": "Love Hypothesis"},
            },
        }
        server_process.stdin.write(json.dumps(call_req) + "\n")
        server_process.stdin.flush()
        line = server_process.stdout.readline()
        call_resp = json.loads(line)
        content = call_resp.get("result", {}).get("content", [])
        print(f"[PASS] tools/call 'search_books' returned {len(content)} content blocks.")
        assert len(content) > 0

        # 4. List Resources Request
        res_req = {"jsonrpc": "2.0", "id": 4, "method": "resources/list", "params": {}}
        server_process.stdin.write(json.dumps(res_req) + "\n")
        server_process.stdin.flush()
        line = server_process.stdout.readline()
        res_resp = json.loads(line)
        resources = res_resp.get("result", {}).get("resources", [])
        res_uris = [r["uri"] for r in resources]
        print(f"[PASS] Resources Discovered ({len(res_uris)}):", res_uris)
        assert len(res_uris) == 5

        # 5. List Prompts Request
        prompts_req = {"jsonrpc": "2.0", "id": 5, "method": "prompts/list", "params": {}}
        server_process.stdin.write(json.dumps(prompts_req) + "\n")
        server_process.stdin.flush()
        line = server_process.stdout.readline()
        prompts_resp = json.loads(line)
        prompts = prompts_resp.get("result", {}).get("prompts", [])
        prompt_names = [p["name"] for p in prompts]
        print(f"[PASS] Prompts Discovered ({len(prompt_names)}):", prompt_names)
        assert len(prompt_names) == 5

        print("================================================================================")
        print("ALL MCP INSPECTOR PROTOCOL CHECKS SUCCEEDED WITH ZERO STDOUT CORRUPTION!")
        print("================================================================================")

    finally:
        server_process.terminate()
        try:
            server_process.wait(timeout=2.0)
        except Exception:
            server_process.kill()

if __name__ == "__main__":
    run_inspector_test()
