"""1-on-1 E2E encryption using the Signal Protocol (Double Ratchet + X3DH).

Each 10ms audio frame is encrypted with AES-256-GCM using a per-frame key
derived from the current Double Ratchet state. Fixed 256-byte output size
(padded) prevents length leakage.

Dependencies:
    pip install cryptography
    pip install git+https://github.com/signalapp/libsignal#subdirectory=python
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# Frame wire format: | nonce(12B) | ciphertext | tag(16B) | padding |
# Total fixed size: 256 bytes
FRAME_WIRE_SIZE = 256
NONCE_SIZE      = 12
TAG_SIZE        = 16
MAX_PLAINTEXT   = FRAME_WIRE_SIZE - NONCE_SIZE - TAG_SIZE  # 228 bytes


@dataclass
class DoubleRatchetSession:
    """Minimal Double Ratchet implementation for per-frame key derivation.

    For production replace with the full libsignal Double Ratchet bindings.
    This implements the symmetric-key ratchet portion (chain key → message key).
    """

    chain_key: bytes = field(default_factory=lambda: os.urandom(32))
    _msg_idx: int    = field(init=False, default=0)

    def next_message_key(self) -> bytes:
        """Derive next per-frame AES-256 key and advance chain key."""
        # CK_i+1 = HKDF(CK_i, 0x01, "chain")
        # MK_i   = HKDF(CK_i, 0x02, "msg")
        self.chain_key = _hkdf(self.chain_key, b"\x01", b"chain", 32)
        msg_key        = _hkdf(self.chain_key, b"\x02", b"msg",   32)
        self._msg_idx += 1
        return msg_key


@dataclass
class SignalSession:
    """Encrypt/decrypt audio frames for a 1-on-1 session."""

    local_ratchet:  DoubleRatchetSession = field(default_factory=DoubleRatchetSession)
    remote_ratchet: DoubleRatchetSession = field(default_factory=DoubleRatchetSession)

    @classmethod
    def from_x3dh(cls, shared_secret: bytes) -> "SignalSession":
        """Bootstrap ratchets from an X3DH shared secret."""
        send_ck = _hkdf(shared_secret, b"send", b"rootkey", 32)
        recv_ck = _hkdf(shared_secret, b"recv", b"rootkey", 32)
        return cls(
            local_ratchet  = DoubleRatchetSession(chain_key=send_ck),
            remote_ratchet = DoubleRatchetSession(chain_key=recv_ck),
        )

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt ≤227 bytes of plaintext → 256-byte fixed-size ciphertext.

        Wire layout: nonce(12) | length(1) | padded_plaintext(227) | tag(16)
        The 1-byte length prefix makes unpadding exact and safe for plaintexts
        that legitimately end with zero bytes.
        """
        if len(plaintext) > MAX_PLAINTEXT - 1:
            plaintext = plaintext[:MAX_PLAINTEXT - 1]
        length_prefix = bytes([len(plaintext)])
        padded    = length_prefix + plaintext + b"\x00" * (MAX_PLAINTEXT - 1 - len(plaintext))
        key       = self.local_ratchet.next_message_key()
        nonce     = os.urandom(NONCE_SIZE)
        ct        = AESGCM(key).encrypt(nonce, padded, None)  # includes 16B tag
        wire = nonce + ct
        if len(wire) != FRAME_WIRE_SIZE:
            raise RuntimeError(f"Invariant violated: wire size {len(wire)} != {FRAME_WIRE_SIZE}")
        return wire

    def decrypt(self, wire_frame: bytes) -> bytes:
        """Decrypt a 256-byte wire frame → original plaintext."""
        if len(wire_frame) != FRAME_WIRE_SIZE:
            raise ValueError(f"Invalid frame size: {len(wire_frame)}")
        nonce  = wire_frame[:NONCE_SIZE]
        ct     = wire_frame[NONCE_SIZE:]
        key    = self.remote_ratchet.next_message_key()
        padded = AESGCM(key).decrypt(nonce, ct, None)
        length = padded[0]
        return padded[1: 1 + length]


def _hkdf(key_material: bytes, info_suffix: bytes, salt_str: bytes, length: int) -> bytes:
    return HKDF(
        algorithm = hashes.SHA256(),
        length    = length,
        salt      = salt_str,
        info      = info_suffix,
    ).derive(key_material)
