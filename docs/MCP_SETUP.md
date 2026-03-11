# MCP Setup — Minimal

Register the MCP stdio server for Copilot Chat (one-liner):
```bash
code --add-mcp '{"name":"knowledge-mcp-server","command":"uv","args":["run","python","-m","knowledge_mcp_server.mcp_stdio_server"],"cwd":"'$(pwd)'"}'"}
code --add-mcp '{"name":"knowledge-mcp-server","command":"uv","args":["run","python","-m","knowledge_mcp_server.mcp_stdio_server"],"cwd":"'$(pwd)'"}'
```

Tools exposed
- `#mcp_knowledge-mcp-server_query_knowledge`
- `#mcp_knowledge-mcp-server_index_url`
- `#mcp_knowledge-mcp-server_add_document`
- `#mcp_knowledge-mcp-server_list_documents`

Quick troubleshooting
- Server fails to start: `uv run python -m knowledge_mcp_server.mcp_stdio_server`
- Check logs: `tail -f /tmp/mcp_server.log`
- Reinstall deps: `uv sync`

Notes
- Use stdio transport; keep logs off stdout/stderr.
- Prefer absolute cwd when registering.

That's it — minimal steps to register and debug the MCP server.
