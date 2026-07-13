# Using predictive-maintenance-mcp with Ollama

This guide explains how to use the Predictive Maintenance MCP Server with **[Ollama](https://ollama.com/)** for fully local, air-gapped vibration analysis.

> **Why Ollama?** All signal data stays on your machine. No API keys needed. Full privacy for sensitive industrial data.

---

## Prerequisites

- **Ollama** installed and running ([download](https://ollama.com/download))
- **predictive-maintenance-mcp** installed (see [INSTALL.md](../INSTALL.md))
- A model with tool-calling support (see below)

---

## Step 1: Install a Compatible Model

MCP requires **tool calling** (function calling) support. Not all Ollama models support this. Recommended models:

| Model | Size | Tool Calling | Notes |
|-------|------|:------------:|-------|
| `qwen2.5:14b` | ~9 GB | ✅ | Best balance of quality and size |
| `qwen2.5:7b` | ~4.7 GB | ✅ | Lighter, good for screening |
| `qwen2.5:32b` | ~20 GB | ✅ | Best quality for complex diagnosis |
| `llama3.1:8b` | ~4.7 GB | ✅ | Good general purpose |
| `llama3.1:70b` | ~40 GB | ✅ | High quality, needs >48 GB RAM |
| `mistral-nemo:12b` | ~7.1 GB | ✅ | Strong reasoning |

```bash
# Pull a model (example: Qwen 2.5 14B)
ollama pull qwen2.5:14b
```

Verify it's running:
```bash
ollama list
```

---

## Step 2: Connect via an MCP Client

Ollama itself does not natively speak the MCP protocol. You need an **MCP client** that can bridge Ollama's API with MCP servers. Several options exist:

### Option A: Open WebUI (Recommended)

[Open WebUI](https://github.com/open-webui/open-webui) supports both Ollama backends and MCP tool servers.

1. Install Open WebUI:
   ```bash
   pip install open-webui
   open-webui serve
   ```

2. In the Open WebUI settings, add the MCP server:
   - Go to **Settings → Tools → MCP Servers**
   - Add a new server with command:
     ```
     /path/to/predictive-maintenance-mcp/.venv/bin/python -m predictive_maintenance_mcp
     ```
   - On Windows:
     ```
     C:\path\to\predictive-maintenance-mcp\.venv\Scripts\python.exe -m predictive_maintenance_mcp
     ```

3. Select your Ollama model and start analyzing.

### Option B: Claude Code (CLI) with Ollama Backend

If you use [Claude Code](https://docs.anthropic.com/en/docs/claude-code) with an Ollama-compatible API proxy, configure the MCP server in your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "predictive-maintenance": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "predictive_maintenance_mcp"]
    }
  }
}
```

### Option C: Custom Python Client

For programmatic access, use the MCP Python SDK directly with Ollama's OpenAI-compatible API:

```python
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:14b"

async def main():
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "predictive_maintenance_mcp"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            
            # Convert MCP tools to Ollama format
            ollama_tools = []
            for tool in tools.tools:
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            # Chat with Ollama
            messages = [
                {"role": "user", "content": "List available vibration signals"}
            ]
            
            response = httpx.post(OLLAMA_URL, json={
                "model": MODEL,
                "messages": messages,
                "tools": ollama_tools,
                "stream": False
            })
            
            result = response.json()
            
            # Handle tool calls from Ollama
            if result["message"].get("tool_calls"):
                for call in result["message"]["tool_calls"]:
                    tool_result = await session.call_tool(
                        call["function"]["name"],
                        arguments=call["function"]["arguments"]
                    )
                    print(f"Tool: {call['function']['name']}")
                    print(f"Result: {tool_result.content}")

asyncio.run(main())
```

---

## Step 3: Verify the Connection

Once connected, ask the model to list available signals:

```
List available vibration signals
```

Expected: The model calls `list_signals()` and returns a list of `.csv` files from `data/signals/`.

Then try a quick screening:

```
Analyze statistics for real_train/baseline_1.csv
```

---

## Performance Considerations

| Aspect | Recommendation |
|--------|---------------|
| **Model size** | 14B+ for reliable tool calling; 7B works for simple queries |
| **RAM** | Model size × 1.2 + 4 GB for the MCP server and data |
| **GPU** | Strongly recommended for models ≥14B |
| **Quantization** | Q4_K_M is a good balance (default in Ollama) |
| **Context window** | 8K minimum; 32K+ recommended for multi-step diagnosis |

### Limitations with Local Models

- **Tool calling reliability**: Smaller models (<7B) may format tool calls incorrectly or hallucinate tool names. Use 14B+ for production diagnostics.
- **Multi-step workflows**: Complex workflows like `diagnose_bearing` (8 steps) work best with 32B+ models. Consider breaking into individual tool calls for smaller models.
- **Evidence-based inference**: Local models may not follow the evidence-based inference policy as strictly as Claude. Always verify diagnostic conclusions against the raw tool outputs.

---

## Troubleshooting

### Model doesn't call tools
- Ensure your model supports tool calling (`ollama show <model>` to check capabilities)
- Try a larger model (14B+)
- Some models need explicit prompting: "Use the `analyze_statistics` tool to..."

### Connection refused
- Verify Ollama is running: `ollama serve`
- Check the port: default is `http://localhost:11434`

### MCP server not starting
- Use absolute paths in the configuration
- Verify the venv: `.venv/Scripts/python.exe -c "import mcp; print('ok')"`
- Check [INSTALL.md](../INSTALL.md) troubleshooting section

### Out of memory
- Use a smaller model or increase swap
- Close other applications  
- Try quantized versions: `ollama pull qwen2.5:14b-q4_0`

---

## Security Note

Running everything locally (Ollama + MCP server) means:
- **Zero data leaves your network** — ideal for proprietary industrial data
- **No API keys** — no cloud dependency
- **Full air-gap capable** — works offline after initial model download

See [SECURITY.md](../SECURITY.md) for the full privacy architecture.
