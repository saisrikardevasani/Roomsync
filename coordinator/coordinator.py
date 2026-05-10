"""Room coordinator entry point.

Usage:
    python coordinator.py --room-code ABC123 --role host [--relay-url wss://...]
    python coordinator.py --room-code ABC123 --role peer  [--relay-url wss://...]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets

from discovery    import DiscoveryService, Peer
from reference_bus import ReferenceBus
from relay        import RelayClient

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("coordinator")

COORD_PORT = int(os.getenv("ROOMSYNC_PORT", "45320"))


async def run(args: argparse.Namespace) -> None:
    peer_id    = secrets.token_bytes(16)
    ref_bus    = ReferenceBus()
    peers: list[Peer] = []

    def on_peer_added(p: Peer) -> None:
        peers.append(p)
        log.info("Peer joined the room: %s at %s:%d", p.hostname, p.ip, p.port)

    def on_peer_removed(name: str) -> None:
        nonlocal peers
        peers = [p for p in peers if p.hostname != name]
        log.info("Peer left: %s", name)

    discovery = DiscoveryService(
        room_code      = args.room_code,
        role           = args.role,
        port           = COORD_PORT,
        on_peer_added  = on_peer_added,
        on_peer_removed= on_peer_removed,
    )

    relay: RelayClient | None = None
    if args.relay_url:
        relay = RelayClient(
            relay_url = args.relay_url,
            room_code = args.room_code,
            peer_id   = peer_id,
        )
        await relay.connect()

    await discovery.start()
    await ref_bus.start()

    log.info("RoomSync coordinator running | room=%s role=%s port=%d",
             args.room_code, args.role, COORD_PORT)

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await discovery.stop()
        await ref_bus.stop()
        if relay: await relay.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="RoomSync room coordinator")
    parser.add_argument("--room-code", required=True)
    parser.add_argument("--role",      choices=["host", "peer"], default="peer")
    parser.add_argument("--relay-url", default=None,
                        help="WebSocket URL of the TURN relay (optional)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
