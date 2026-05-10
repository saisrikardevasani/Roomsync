"""TURN relay fallback for when peers are not on the same LAN.

Uses aiohttp to connect to a zero-knowledge relay server. The relay forwards
encrypted audio frames but never holds decryption keys.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass, field
from typing import Callable, Optional

import aiohttp

log = logging.getLogger(__name__)

RELAY_WS_PATH = "/relay/v1"

# Wire frame over WebSocket: | seq(4B) | peer_id(16B) | payload |
_HDR_FMT  = "!I16s"
_HDR_SIZE = struct.calcsize(_HDR_FMT)


@dataclass
class RelayClient:
    relay_url: str        # wss://relay.roomsync.example/relay/v1
    room_code: str
    peer_id: bytes        # 16-byte random peer identifier
    on_frame: Callable[[bytes, bytes], None] = field(default=lambda pid, f: None)

    _ws: Optional[aiohttp.ClientWebSocketResponse] = field(init=False, default=None)
    _session: Optional[aiohttp.ClientSession]      = field(init=False, default=None)
    _seq: int                                      = field(init=False, default=0)
    _running: bool                                 = field(init=False, default=False)

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        url = f"{self.relay_url}?room={self.room_code}"
        self._ws      = await self._session.ws_connect(url, heartbeat=10.0)
        self._running = True
        asyncio.get_event_loop().create_task(self._recv_loop())
        log.info("Relay connected to %s room=%s", self.relay_url, self.room_code)

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:     await self._ws.close()
        if self._session: await self._session.close()

    async def send_frame(self, encrypted_frame: bytes) -> None:
        if not self._ws or self._ws.closed:
            return
        header = struct.pack(_HDR_FMT, self._seq, self.peer_id)
        self._seq += 1
        await self._ws.send_bytes(header + encrypted_frame)

    async def _recv_loop(self) -> None:
        while self._running and self._ws and not self._ws.closed:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.BINARY:
                data = msg.data
                if len(data) < _HDR_SIZE:
                    continue
                seq, peer_id = struct.unpack_from(_HDR_FMT, data)
                payload      = data[_HDR_SIZE:]
                self.on_frame(bytes(peer_id), payload)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                log.warning("Relay WebSocket closed/error — will reconnect")
                break
