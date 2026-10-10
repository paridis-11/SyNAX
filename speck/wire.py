import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import socket
import urllib.parse
import urllib.request

import torch

MAX_BODY = 128 * 1024 * 1024


def pack(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    body = stream.getvalue()
    if len(body) > MAX_BODY:
        raise ValueError("Message exceeds 128 MiB")
    return body


def unpack(body):
    if len(body) > MAX_BODY:
        raise ValueError("Message exceeds 128 MiB")
    return torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def private_address(address):
    ip = ipaddress.ip_address(address)
    return (ip.is_loopback or ip in ipaddress.ip_network("100.64.0.0/10")
            or ip in ipaddress.ip_network("10.0.0.0/8")
            or ip in ipaddress.ip_network("172.16.0.0/12")
            or ip in ipaddress.ip_network("192.168.0.0/16")
            or ip in ipaddress.ip_network("fd7a:115c:a1e0::/48"))


class Transport:
    def __init__(self, server, token, timeout=30, proxy=None):
        parsed = urllib.parse.urlsplit(server)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise ValueError("Use an http(s)://host:port server address")
        if parsed.scheme == "http":
            addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 80)
            if not addresses or not all(private_address(item[4][0]) for item in addresses):
                raise ValueError("Unencrypted HTTP is restricted to LAN/Tailscale/loopback")
        if len(token) < 32:
            raise ValueError("Use a generated client token of at least 32 characters")
        self.server = server.rstrip("/")
        self.token = token
        self.timeout = timeout
        proxies = {}
        if proxy:
            target = urllib.parse.urlsplit(proxy)
            if (target.scheme != "http" or not target.hostname or not target.port
                    or target.username or target.password or target.query or target.fragment
                    or target.path not in {"", "/"}
                    or not ipaddress.ip_address(target.hostname).is_loopback):
                raise ValueError("Proxy must be an explicit loopback HTTP proxy, e.g. http://127.0.0.1:1055")
            proxies = {"http": proxy, "https": proxy}
        # Ignore ambient proxy variables; only an explicitly selected local proxy is used.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))

    def request(self, path, payload=None):
        body = None if payload is None else pack(payload)
        request = urllib.request.Request(
            self.server + path, data=body,
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/x-synax-tensors"})
        with self.opener.open(request, timeout=self.timeout) as response:
            content = response.read(MAX_BODY + 1)
            if response.headers.get("Content-Type") == "application/json":
                return json.loads(content)
            return unpack(content)
