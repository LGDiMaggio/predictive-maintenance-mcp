"""Where a tool call comes from, as far as this server can tell.

This server does not authenticate its clients. A tool that reads local files
therefore asks, per call, whether the call came directly from this machine:
over stdio or an in-process client, or over HTTP on a server started through
its command line on a loopback address, from a loopback peer.

The recorded bind address and the socket's peer address are facts the server
observes. Forwarding headers and ``Host`` come from the client, so they can
only turn the answer into "no": a client cannot talk its way into "yes" with
them. A proxy on this machine that adds no forwarding header and presents a
loopback ``Host`` cannot be told from a direct call.
"""

import ipaddress
from collections.abc import Mapping
from typing import Any, Optional

#: Network transports (they open a socket; stdio does not).
NETWORK_TRANSPORTS: tuple[str, ...] = ("sse", "streamable-http")

#: Headers a reverse proxy adds: a call carrying one did not come from this
#: machine directly.
_FORWARDING_HEADERS: tuple[str, ...] = (
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-real-ip",
)

#: The transport and bind address ``server.main`` records before the server
#: starts; both stay None when the server was started any other way.
_SERVER: dict[str, Optional[str]] = {"transport": None, "host": None}


def is_loopback_host(host: Optional[str]) -> bool:
    """Whether binding *host* keeps the socket on this machine.

    ``localhost`` (any letter case) and every loopback address (127.0.0.0/8,
    ``::1``, an IPv4-mapped loopback, with or without brackets) count. An
    empty host, a wildcard (``0.0.0.0``, ``::``), any other address and any
    other name do not: a name could resolve anywhere.
    """
    if host is None:
        return False
    text = host.strip()
    if text.lower() == "localhost":
        return True
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def is_remote_transport(transport: str, host: Optional[str]) -> bool:
    """A network transport bound to an address other clients can reach."""
    return transport in NETWORK_TRANSPORTS and not is_loopback_host(host)


def configure_server_transport(transport: str, host: Optional[str]) -> None:
    """Record the transport the server is about to run on (``server.main``)."""
    _SERVER["transport"] = transport
    _SERVER["host"] = host


def _request_headers(ctx: Any) -> Optional[Mapping[str, str]]:
    """The HTTP headers of the call, or None for a call that did not arrive
    over HTTP (stdio, an in-process client, a direct Python call)."""
    if ctx is None:
        return None
    try:
        headers = ctx.headers
    except (AttributeError, LookupError, ValueError, RuntimeError):
        return None
    return headers if isinstance(headers, Mapping) else None


def _host_name(host_header: str) -> str:
    """The name or address of a Host header, without its port."""
    text = host_header.strip()
    if text.startswith("["):
        return text[1 : text.find("]")] if "]" in text else text
    if text.count(":") == 1:
        return text.split(":", 1)[0]
    return text


def _peer_address(ctx: Any) -> Optional[str]:
    """The address of the socket's other end, or None when it is unknown."""
    try:
        client = ctx.request_context.request.client
    except (AttributeError, LookupError, ValueError, RuntimeError):
        return None
    host = getattr(client, "host", None)
    return host if isinstance(host, str) else None


def is_direct_local_call(ctx: Any) -> bool:
    """Whether this tool call came directly from this machine.

    A call that did not arrive over HTTP is local unless the server recorded
    a bind other machines can reach. A call over HTTP is local only when the
    server was started through its command line on a loopback address, the
    peer address is a loopback address, the call carries no proxy forwarding
    header and its ``Host`` is a loopback name or address; a server started
    any other way answers "no" for every HTTP call.
    """
    transport, host = _SERVER["transport"], _SERVER["host"]
    if transport is not None and is_remote_transport(transport, host):
        return False
    headers = _request_headers(ctx)
    if headers is None:
        return True
    return (
        transport in NETWORK_TRANSPORTS
        and is_loopback_host(host)
        and is_loopback_host(_peer_address(ctx))
        and not any(name in headers for name in _FORWARDING_HEADERS)
        and is_loopback_host(_host_name(headers.get("host", "")))
    )
