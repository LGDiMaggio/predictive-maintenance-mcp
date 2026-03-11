# Deployment Guide — Copilot Studio & Enterprise HTTPS

Deploy the Predictive Maintenance MCP Server as an **HTTPS endpoint** for **Microsoft Copilot Studio**, remote Claude Desktop instances, or any MCP client that requires a network-accessible server.

> **No Docker?** No problem. Options A and B below work with just Python + a reverse proxy — no containers required.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Option A — Direct Python + Reverse Proxy (No Docker)](#option-a--direct-python--reverse-proxy-no-docker)
3. [Option B — Windows Service (Corporate IT)](#option-b--windows-service-corporate-it)
4. [Option C — Docker Compose + Caddy](#option-c--docker-compose--caddy)
5. [Option D — Azure App Service (Cloud)](#option-d--azure-app-service-cloud)
6. [Connecting Microsoft Copilot Studio (Step-by-Step)](#connecting-microsoft-copilot-studio-step-by-step)
7. [Connecting Other MCP Clients](#connecting-other-mcp-clients)
8. [Authentication](#authentication)
9. [Security Considerations](#security-considerations)
10. [CLI Reference](#cli-reference)
11. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌──────────────────────┐
│  MCP Client          │
│  (Copilot Studio,    │   HTTPS (port 443)
│   Claude Desktop,    │ ──────────────────► ┌─────────────────┐
│   VS Code, etc.)     │                     │  Reverse Proxy   │
└──────────────────────┘                     │  (Caddy / nginx / │
                                             │   IIS / Azure)    │
                                             └────────┬────────┘
                                                      │ HTTP :8000
                                             ┌────────▼────────┐
                                             │  MCP Server      │
                                             │  (SSE transport)  │
                                             └─────────────────┘
```

| Transport | Protocol | Use case |
|-----------|----------|----------|
| `stdio` (default) | Standard I/O | Local: Claude Desktop, VS Code |
| `sse` | HTTP + Server-Sent Events | **Remote: Copilot Studio, networked clients** |
| `streamable-http` | HTTP (MCP 2025-03-26 spec) | Remote: newer MCP clients |

---

## Option A — Direct Python + Reverse Proxy (No Docker)

**Best for**: Corporate environments where Docker is not available.

### 1. Install the server

```bash
# On the server machine
pip install predictive-maintenance-mcp[sse]

# Or from source
git clone https://github.com/LGDiMaggio/predictive-maintenance-mcp.git
cd predictive-maintenance-mcp
pip install -e .[sse]
```

### 2. Start the SSE server

```bash
# Linux / macOS
predictive-maintenance-mcp --transport sse --host 127.0.0.1 --port 8000

# Windows (PowerShell)
predictive-maintenance-mcp --transport sse --host 127.0.0.1 --port 8000

# Or with environment variables
set MCP_TRANSPORT=sse
set MCP_HOST=127.0.0.1
set MCP_PORT=8000
predictive-maintenance-mcp
```

**Verify** the server is running:

```bash
curl http://127.0.0.1:8000/sse
# Should return an SSE event stream (text/event-stream)
```

### 3. Add a reverse proxy for HTTPS

You need a reverse proxy to terminate TLS (HTTPS). Choose one:

#### Option A1: Caddy (simplest — auto TLS)

Install Caddy: https://caddyserver.com/docs/install

Create a `Caddyfile`:

```
mcp.yourdomain.com {
    reverse_proxy 127.0.0.1:8000
}
```

Run Caddy:

```bash
caddy run
```

Caddy automatically obtains Let's Encrypt TLS certificates. Your server is now at `https://mcp.yourdomain.com/sse`.

#### Option A2: nginx

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

#### Option A3: IIS with URL Rewrite + ARR (Windows Server)

For Windows Server environments with IIS:

1. Install **URL Rewrite** and **Application Request Routing (ARR)** from the IIS module gallery
2. Enable ARR proxy: IIS Manager → Server → Application Request Routing → Server Proxy Settings → Enable proxy
3. Add a reverse proxy rule in `web.config`:

```xml
<configuration>
  <system.webServer>
    <rewrite>
      <rules>
        <rule name="MCP Reverse Proxy" stopProcessing="true">
          <match url="(.*)" />
          <action type="Rewrite" url="http://127.0.0.1:8000/{R:1}" />
        </rule>
      </rules>
    </rewrite>
    <!-- Disable response buffering for SSE -->
    <httpProtocol>
      <customHeaders>
        <add name="X-Content-Type-Options" value="nosniff" />
      </customHeaders>
    </httpProtocol>
  </system.webServer>
</configuration>
```

4. Bind HTTPS to your site with your corporate TLS certificate.

> **Important for IIS**: Disable response buffering and set `responseBufferLimit="0"` on the handler to allow SSE streaming.

---

## Option B — Windows Service (Corporate IT)

Run the server as a persistent Windows service using [NSSM](https://nssm.cc/) (Non-Sucking Service Manager):

```powershell
# 1. Install NSSM (or use winget)
winget install nssm

# 2. Install the service
nssm install PredictiveMaintenanceMCP "C:\path\to\.venv\Scripts\predictive-maintenance-mcp.exe"
nssm set PredictiveMaintenanceMCP AppParameters "--transport sse --host 127.0.0.1 --port 8000"
nssm set PredictiveMaintenanceMCP AppDirectory "C:\path\to\predictive-maintenance-mcp"
nssm set PredictiveMaintenanceMCP DisplayName "Predictive Maintenance MCP Server"
nssm set PredictiveMaintenanceMCP Description "MCP server for industrial machinery diagnostics"
nssm set PredictiveMaintenanceMCP Start SERVICE_AUTO_START

# 3. Start the service
nssm start PredictiveMaintenanceMCP
```

Then configure IIS, Caddy, or nginx as the reverse proxy (see Option A above).

---

## Option C — Docker Compose + Caddy

> **Requires** Docker and Docker Compose. Skip this if Docker is not available in your environment.

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

3. **Uncomment the Caddy service** in `docker-compose.yml` and remove the `ports: - "8000:8000"` from `mcp-server`.

4. **Start the stack**:

   ```bash
   docker compose up -d
   ```

5. **Verify**:

   ```bash
   curl https://mcp.yourdomain.com/sse
   ```

---

## Option D — Azure App Service (Cloud)

For enterprises using Azure (no Docker needed):

### Using Azure App Service with Python

1. Create an **Azure App Service** (Python 3.11 runtime)

2. Deploy the code via Git or ZIP:

   ```bash
   az webapp up --name your-mcp-server --runtime "PYTHON:3.11" --sku B1
   ```

3. Set **Application Settings** (environment variables):

   | Setting | Value |
   |---------|-------|
   | `MCP_TRANSPORT` | `sse` |
   | `MCP_HOST` | `0.0.0.0` |
   | `MCP_PORT` | `8000` |

4. Set the **Startup command**:

   ```
   predictive-maintenance-mcp
   ```

5. Your endpoint: `https://your-mcp-server.azurewebsites.net/sse`

### Using Azure Container Instances (if Docker is allowed)

```bash
az container create \
  --resource-group myResourceGroup \
  --name mcp-server \
  --image ghcr.io/lgdimaggio/predictive-maintenance-mcp:latest \
  --ports 8000 \
  --environment-variables MCP_TRANSPORT=sse MCP_HOST=0.0.0.0 MCP_PORT=8000
```

Add **Azure Application Gateway** or **Azure Front Door** for TLS termination.

---

## Connecting Microsoft Copilot Studio (Step-by-Step)

### Prerequisites

- The MCP server is running and reachable over HTTPS (see options above)
- You have access to [Microsoft Copilot Studio](https://copilotstudio.microsoft.com/)

### Step 1: Verify your server is reachable

From a browser or Copilot Studio's network, confirm you can reach the endpoint:

```
https://mcp.yourdomain.com/sse
```

You should see an SSE event stream or a connection that stays open.

### Step 2: Add the MCP connector in Copilot Studio

In Copilot Studio, navigate to the **connectors** or **MCP server** configuration section.

Fill in the required fields:

| Field | Value |
|-------|-------|
| **Server name** | `Predictive Maintenance` |
| **Server description** | `Industrial machinery diagnostics: vibration analysis, bearing fault detection, ISO 20816 assessment, and predictive maintenance workflows` |
| **Server URL** | `https://mcp.yourdomain.com/sse` |
| **Authentication** | See the [Authentication section](#authentication) below |

### Step 3: Choose authentication

Copilot Studio offers three authentication options:

| Option | When to use |
|--------|-------------|
| **None** | Internal/private networks, testing |
| **API Key** | Simple token-based access control |
| **OAuth 2.0** | Azure AD / Entra ID corporate SSO |

See the detailed [Authentication section](#authentication) below for configuration.

### Step 4: Test the connection

After saving the configuration, Copilot Studio should discover the server's tools automatically. Try a prompt like:

> "List all available vibration signals"

or

> "Analyze the FFT spectrum of baseline_1.csv"

---

## Connecting Other MCP Clients

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

### Streamable HTTP (MCP 2025-03-26 spec)

For newer clients that support the `streamable-http` transport:

```bash
predictive-maintenance-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

Endpoint: `https://mcp.yourdomain.com/mcp`

---

## Authentication

The MCP server itself does not enforce authentication — instead, authentication is handled at the **reverse proxy layer**. This is the standard pattern for MCP servers and gives you full control over the auth mechanism.

### Option 1: None (Internal Networks)

If the server is only accessible within your corporate network or VPN, no authentication may be needed. Select **None** in Copilot Studio.

### Option 2: API Key (via Reverse Proxy)

Add API key validation at the reverse proxy. The client sends a key in the `Authorization` header; the proxy validates it before forwarding to the MCP server.

#### Caddy with API Key

```
mcp.yourdomain.com {
    @no_api_key {
        not header Authorization "Bearer YOUR-SECRET-API-KEY-HERE"
    }
    respond @no_api_key 401 {
        body "Unauthorized"
        close
    }
    reverse_proxy 127.0.0.1:8000
}
```

#### nginx with API Key

```nginx
server {
    listen 443 ssl;
    server_name mcp.yourdomain.com;

    ssl_certificate     /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;

    # API Key validation
    if ($http_authorization != "Bearer YOUR-SECRET-API-KEY-HERE") {
        return 401 "Unauthorized";
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
    }
}
```

**In Copilot Studio**: Select **API Key** authentication, then enter your key.

### Option 3: OAuth 2.0 (Azure AD / Entra ID)

For corporate SSO with Azure AD (Entra ID):

#### 1. Register an application in Azure AD

1. Go to [Azure Portal](https://portal.azure.com) → **Azure Active Directory** → **App registrations** → **New registration**
2. Name: `Predictive Maintenance MCP Server`
3. Supported account types: **Accounts in this organizational directory only**
4. Register, then note the:
   - **Application (client) ID**
   - **Directory (tenant) ID**
5. Go to **Certificates & secrets** → **New client secret** → copy the value
6. Go to **Expose an API** → **Add a scope**:
   - Scope name: `mcp.access`
   - Who can consent: Admins and users
   - Display name: `Access MCP Server`
7. Go to **API permissions** → **Add a permission** → **My APIs** → select your app → add `mcp.access`

#### 2. Configure the reverse proxy to validate OAuth tokens

##### Caddy with OAuth (via caddy-security plugin)

```
{
    order authenticate before respond
    security {
        oauth identity provider azure {
            driver azure
            tenant_id YOUR_TENANT_ID
            client_id YOUR_CLIENT_ID
            client_secret YOUR_CLIENT_SECRET
            scopes openid profile email
        }
        authentication portal myportal {
            enable identity provider azure
        }
    }
}

mcp.yourdomain.com {
    authenticate with myportal
    reverse_proxy 127.0.0.1:8000
}
```

##### nginx with OAuth (via oauth2-proxy)

Use [oauth2-proxy](https://oauth2-proxy.github.io/oauth2-proxy/) as a sidecar:

```bash
oauth2-proxy \
  --provider=oidc \
  --oidc-issuer-url=https://login.microsoftonline.com/YOUR_TENANT_ID/v2.0 \
  --client-id=YOUR_CLIENT_ID \
  --client-secret=YOUR_CLIENT_SECRET \
  --upstream=http://127.0.0.1:8000 \
  --http-address=0.0.0.0:4180 \
  --email-domain=yourdomain.com
```

Then point nginx at `http://127.0.0.1:4180`.

**In Copilot Studio**: Select **OAuth 2.0** authentication, then provide:

| Field | Value |
|-------|-------|
| **Client ID** | Your Azure AD Application (client) ID |
| **Client secret** | The client secret value |
| **Authorization URL** | `https://login.microsoftonline.com/YOUR_TENANT_ID/oauth2/v2.0/authorize` |
| **Token URL** | `https://login.microsoftonline.com/YOUR_TENANT_ID/oauth2/v2.0/token` |
| **Scope** | `api://YOUR_CLIENT_ID/mcp.access` |

Replace `YOUR_TENANT_ID` and `YOUR_CLIENT_ID` with your actual Azure AD values.

---

## Security Considerations

- **Always use HTTPS** in production — never expose HTTP MCP endpoints directly to the internet
- Use a **reverse proxy** (Caddy, nginx, IIS, Azure Front Door) for TLS termination
- Add **authentication** (API key or OAuth) for any server exposed outside your private network
- The server binds to `127.0.0.1` by default — use `--host 0.0.0.0` only behind a reverse proxy or firewall
- In corporate environments, coordinate with your IT security team for network policies and firewall rules
- Rotate API keys periodically; use OAuth 2.0 for production multi-user environments

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

---

## Troubleshooting

### "Connection refused" from Copilot Studio

1. **Is the server running?** Check with `curl http://127.0.0.1:8000/sse` from the server itself
2. **Is the reverse proxy forwarding?** Check with `curl https://mcp.yourdomain.com/sse` from an external machine
3. **Firewall**: Ensure ports 443 (HTTPS) and 80 (HTTP, for Let's Encrypt) are open
4. **DNS**: Verify `mcp.yourdomain.com` resolves to your server's IP

### SSE connection drops / timeouts

- Ensure your reverse proxy has **buffering disabled** and a long read timeout (see nginx config above)
- For IIS: Set `responseBufferLimit="0"` and increase connection timeout
- Corporate firewalls may terminate idle connections — the MCP SSE protocol includes periodic keepalive events

### Copilot Studio does not discover tools

- Verify the URL ends in `/sse` (not just the domain)
- Check that the server started with `--transport sse` (not `stdio`)
- Check the Copilot Studio logs for error details

### "401 Unauthorized"

- Verify the API key or OAuth token is correct
- Check that the `Authorization` header format matches what the reverse proxy expects (`Bearer <token>`)
- For OAuth: Verify the scope matches what was configured in Azure AD
