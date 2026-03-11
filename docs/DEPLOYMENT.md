# HTTPS Deployment Guide

Deploy the Predictive Maintenance MCP Server as an **HTTPS endpoint** for enterprise MCP clients such as **Microsoft Copilot Studio**, remote Claude Desktop instances, or any MCP client that requires a network-accessible server.

---

## Architecture

```
┌──────────────────────┐
│  MCP Client          │
│  (Copilot Studio,    │   HTTPS (port 443)
│   Claude Desktop,    │ ──────────────────► ┌─────────────────┐
│   VS Code, etc.)     │                     │  Caddy / nginx  │
└──────────────────────┘                     │  (TLS termination)
                                             └────────┬────────┘
                                                      │ HTTP :8000
                                             ┌────────▼────────┐
                                             │  MCP Server      │
                                             │  (SSE transport)  │
                                             └─────────────────┘
```

The server supports three transports:

| Transport | Protocol | Use case |
|-----------|----------|----------|
| `stdio` (default) | Standard I/O | Local: Claude Desktop, VS Code |
| `sse` | HTTP + Server-Sent Events | Remote: Copilot Studio, networked clients |
| `streamable-http` | HTTP (MCP 2025-03-26 spec) | Remote: newer MCP clients |

---

## Quick Start — Local Testing (no TLS)

```bash
# Run SSE server locally
predictive-maintenance-mcp --transport sse --host 0.0.0.0 --port 8080

# Or with environment variables
MCP_TRANSPORT=sse MCP_HOST=0.0.0.0 MCP_PORT=8080 predictive-maintenance-mcp
```

Test the endpoint:

```bash
curl http://localhost:8080/sse
```

---

## Option A — Docker Compose + Caddy (Recommended)

Caddy provides automatic HTTPS with Let's Encrypt certificates, zero configuration.

### Prerequisites

- Docker and Docker Compose installed
- A domain name pointing to your server (e.g. `mcp.example.com`)
- Ports 80 and 443 open on the firewall

### Steps

1. **Clone the repository** on your server:

   ```bash
   git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git
   cd predictive-maintenance-mcp
   ```

2. **Edit the Caddyfile** — replace `mcp.example.com` with your domain:

   ```
   mcp.yourdomain.com {
       reverse_proxy mcp-server:8000
   }
   ```

3. **Uncomment the Caddy service** in `docker-compose.yml`:

   ```yaml
   caddy:
     image: caddy:2-alpine
     restart: unless-stopped
     ports:
       - "80:80"
       - "443:443"
     volumes:
       - ./Caddyfile:/etc/caddy/Caddyfile:ro
       - caddy_data:/data
       - caddy_config:/config
     depends_on:
       - mcp-server
   ```

   Also uncomment the `volumes:` section at the bottom, and remove the `ports: - "8000:8000"` from `mcp-server`.

4. **Start the stack**:

   ```bash
   docker compose up -d
   ```

5. **Verify**:

   ```bash
   curl https://mcp.yourdomain.com/sse
   ```

Caddy will automatically obtain and renew TLS certificates from Let's Encrypt.

---

## Option B — Behind nginx (Existing Infrastructure)

If you already have nginx with TLS certificates:

```nginx
server {
    listen 443 ssl;
    server_name mcp.yourdomain.com;

    ssl_certificate     /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;

        # Required for SSE (Server-Sent Events)
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE connections stay open
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
}
```

Then run the server:

```bash
predictive-maintenance-mcp --transport sse --host 127.0.0.1 --port 8000
```

---

## Option C — Azure / Cloud (for Copilot Studio)

For enterprises using Azure:

1. **Azure Container Instances** or **Azure App Service** to host the Docker container
2. **Azure Application Gateway** or **Azure Front Door** for TLS termination
3. Set environment variables:

   ```
   MCP_TRANSPORT=sse
   MCP_HOST=0.0.0.0
   MCP_PORT=8000
   ```

4. Configure the MCP endpoint in Copilot Studio as `https://your-app.azurewebsites.net/sse`

---

## Connecting MCP Clients

### Microsoft Copilot Studio

In the Copilot Studio MCP server configuration:

- **URL**: `https://mcp.yourdomain.com/sse`
- **Transport**: SSE

### Claude Desktop (Remote)

```json
{
  "mcpServers": {
    "predictive-maintenance": {
      "url": "https://mcp.yourdomain.com/sse"
    }
  }
}
```

### Streamable HTTP

For clients supporting the newer MCP streamable-http transport:

```bash
predictive-maintenance-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

Endpoint: `https://mcp.yourdomain.com/mcp`

---

## Security Considerations

- **Always use HTTPS** in production — never expose HTTP MCP endpoints directly to the internet
- Use a **reverse proxy** (Caddy, nginx, Azure Front Door) for TLS termination
- Consider adding **authentication** (API key, OAuth) at the reverse proxy level for production deployments
- The server binds to `127.0.0.1` by default — use `--host 0.0.0.0` only behind a reverse proxy or firewall
- If deploying in a corporate environment, coordinate with your IT security team for network policies

---

## CLI Reference

```
predictive-maintenance-mcp [OPTIONS]

Options:
  --transport, -t  {stdio,sse,streamable-http}  Transport protocol (default: stdio)
  --host           HOST                          Bind address (default: 127.0.0.1)
  --port, -p       PORT                          Port number (default: 8000)

Environment Variables:
  MCP_TRANSPORT    Same as --transport
  MCP_HOST         Same as --host
  MCP_PORT         Same as --port
```

CLI arguments take precedence over environment variables.
