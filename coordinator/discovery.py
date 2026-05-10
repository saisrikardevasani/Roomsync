"""LAN peer discovery via mDNS / zeroconf.

Each RoomSync instance advertises itself under the service type
  _roomsync._udp.local.
with TXT records carrying the room code and coordinator role.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from zeroconf import ServiceInfo, ServiceBrowser, Zeroconf
from zeroconf.asyncio import AsyncZeroconf

log = logging.getLogger(__name__)

SERVICE_TYPE = "_roomsync._udp.local."
_DEFAULT_TTL_S = 60


@dataclass(frozen=True)
class Peer:
    hostname: str
    ip: str
    port: int
    room_code: str
    role: str  # "host" | "peer"


@dataclass
class DiscoveryService:
    room_code: str
    role: str  # "host" | "peer"
    port: int
    on_peer_added: Callable[[Peer], None] = field(default=lambda p: None)
    on_peer_removed: Callable[[str], None] = field(default=lambda name: None)

    _azc: Optional[AsyncZeroconf] = field(init=False, default=None, repr=False)
    _info: Optional[ServiceInfo]  = field(init=False, default=None, repr=False)
    _peers: Dict[str, Peer]       = field(init=False, default_factory=dict, repr=False)

    async def start(self) -> None:
        self._azc = AsyncZeroconf()
        local_ip  = _local_ip()
        name      = f"roomsync-{socket.gethostname()}.{SERVICE_TYPE}"

        self._info = ServiceInfo(
            SERVICE_TYPE,
            name,
            addresses=[socket.inet_aton(local_ip)],
            port=self.port,
            properties={
                b"room":  self.room_code.encode(),
                b"role":  self.role.encode(),
            },
            server=f"{socket.gethostname()}.local.",
        )
        await self._azc.async_register_service(self._info)
        log.info("Registered mDNS service %s on %s:%d", name, local_ip, self.port)

        # Browse for other instances
        ServiceBrowser(self._azc.zeroconf, SERVICE_TYPE, handlers=[self._on_change])

    async def stop(self) -> None:
        if self._azc and self._info:
            await self._azc.async_unregister_service(self._info)
            await self._azc.async_close()

    def peers(self) -> list[Peer]:
        return list(self._peers.values())

    # ── zeroconf callbacks (called from zeroconf thread) ─────────────────────
    def _on_change(self, zc: Zeroconf, service_type: str,
                   name: str, state_change) -> None:
        from zeroconf import ServiceStateChange
        if state_change is ServiceStateChange.Added:
            self._add_peer(zc, service_type, name)
        elif state_change is ServiceStateChange.Removed:
            self._remove_peer(name)

    def _add_peer(self, zc: Zeroconf, service_type: str, name: str) -> None:
        info = ServiceInfo(service_type, name)
        info.request(zc, timeout=3000)
        if not info.addresses:
            return

        props     = info.decoded_properties
        room_code = props.get("room", "")
        if room_code != self.room_code:
            return  # different room

        ip   = socket.inet_ntoa(info.addresses[0])
        peer = Peer(
            hostname  = name,
            ip        = ip,
            port      = info.port,
            room_code = room_code,
            role      = props.get("role", "peer"),
        )
        self._peers[name] = peer
        log.info("Peer joined: %s (%s:%d)", name, ip, info.port)
        self.on_peer_added(peer)

    def _remove_peer(self, name: str) -> None:
        peer = self._peers.pop(name, None)
        if peer:
            log.info("Peer left: %s", name)
            self.on_peer_removed(name)


def _local_ip() -> str:
    """Return the machine's outbound LAN IP without sending a packet."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
