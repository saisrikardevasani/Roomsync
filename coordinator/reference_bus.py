"""Shared audio reference bus.

Streams each laptop's speaker output to all peers in the room via UDP multicast.
The C++ pipeline consumes the combined reference stream for multi-laptop AEC.

Spec:
  - 8kHz mono, Opus-compressed, ~16kbps
  - 20ms jitter buffer
  - LAN path: UDP multicast, ~1ms latency
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import opuslib  # pip install opuslib

log = logging.getLogger(__name__)

MCAST_GROUP   = "239.255.42.99"
MCAST_PORT    = 45321
FRAME_MS      = 20          # Opus frame size
SAMPLE_RATE   = 8000
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 160 samples
JITTER_FRAMES = 2           # 40ms jitter buffer

# Wire header: | seq (4B) | timestamp_ms (8B) | payload_len (2B) |
HDR_FMT  = "!IQH"
HDR_SIZE = struct.calcsize(HDR_FMT)


@dataclass
class ReferenceBus:
    on_frame: Callable[[bytes], None] = field(default=lambda f: None)

    _encoder: Optional[opuslib.Encoder]  = field(init=False, default=None)
    _decoder: Optional[opuslib.Decoder]  = field(init=False, default=None)
    _send_sock: Optional[socket.socket]  = field(init=False, default=None)
    _recv_sock: Optional[socket.socket]  = field(init=False, default=None)
    _seq: int                            = field(init=False, default=0)
    _jitter_buf: deque                   = field(init=False, default_factory=lambda: deque(maxlen=JITTER_FRAMES + 4))
    _running: bool                       = field(init=False, default=False)

    async def start(self) -> None:
        self._encoder = opuslib.Encoder(SAMPLE_RATE, 1, opuslib.APPLICATION_VOIP)
        self._encoder.bitrate = 16000  # 16kbps
        self._decoder = opuslib.Decoder(SAMPLE_RATE, 1)

        self._send_sock = _multicast_sender()
        self._recv_sock = _multicast_receiver()
        self._running   = True

        asyncio.get_running_loop().create_task(self._recv_loop())
        log.info("Reference bus started on %s:%d", MCAST_GROUP, MCAST_PORT)

    async def stop(self) -> None:
        self._running = False
        if self._send_sock: self._send_sock.close()
        if self._recv_sock: self._recv_sock.close()

    def send_frame(self, pcm_bytes: bytes) -> None:
        """Encode and multicast one 20ms PCM frame (8kHz, 16-bit, mono)."""
        if not self._send_sock or not self._encoder:
            return
        encoded  = self._encoder.encode(pcm_bytes, FRAME_SAMPLES)
        ts_ms    = int(time.monotonic() * 1000)
        header   = struct.pack(HDR_FMT, self._seq, ts_ms, len(encoded))
        self._seq += 1
        self._send_sock.sendto(header + encoded, (MCAST_GROUP, MCAST_PORT))

    async def _recv_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                data, _ = await loop.run_in_executor(
                    None, lambda: self._recv_sock.recvfrom(4096))
                if len(data) < HDR_SIZE:
                    continue
                seq, ts_ms, plen = struct.unpack_from(HDR_FMT, data)
                payload  = data[HDR_SIZE: HDR_SIZE + plen]
                pcm      = self._decoder.decode(payload, FRAME_SAMPLES)
                self._jitter_buf.append((seq, ts_ms, pcm))

                if len(self._jitter_buf) >= JITTER_FRAMES:
                    _, _, frame_pcm = self._jitter_buf.popleft()
                    self.on_frame(frame_pcm)
            except OSError as exc:
                log.debug("Reference bus recv stopped: %s", exc)
                break


def _multicast_sender() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)  # LAN only
    return s


def _multicast_receiver() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", MCAST_PORT))
    mreq = struct.pack("4sL", socket.inet_aton(MCAST_GROUP), socket.INADDR_ANY)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.setblocking(False)
    return s
