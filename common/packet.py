"""UDP wire format shared by edge sender and server receiver.

Each packet on the wire = HEADER (32 B) + PAYLOAD (≤ MAX_PAYLOAD_BYTES).
A logical patch is split into one or more chunks, each carried by exactly
one UDP packet. Loss of any chunk drops that chunk only — reassembly
delivers the patch as "incomplete" so the recovery layer (server side,
Week 2) can decide what to do.

Why these fields ride on EVERY chunk (not just the first one)
-------------------------------------------------------------
UDP gives no ordering or delivery guarantees. If the first chunk of a
patch is lost, we still want the server to know which (frame_id, det_id)
the surviving chunks belong to, where to paste them, and at what JPEG
quality they were encoded. The 32-byte fixed overhead per chunk is the
price of that robustness. With ~1368-byte payloads, header overhead is
~2.3% — easily worth it given that "first chunk loss" is a real failure
mode in lossy UDP.

Endianness
----------
Network byte order (big-endian, '!' in struct) throughout. This matters
when the edge and server happen to be different architectures (Jetson is
ARM, laptop is x86) — both are little-endian today, but '!' future-proofs
against any mixed deployment.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List

# ---------- Wire-format constants ----------

# Magic bytes at the start of every packet. Lets the receiver discard
# unrelated UDP traffic (e.g. mDNS, stray broadcasts) without trying to
# parse it. "RPEC" = RoI Privacy Edge Camera.
MAGIC = b"RPEC"

# Bump if the header layout changes incompatibly. Receiver will discard
# packets with an unknown version (and ideally log it).
PROTOCOL_VERSION = 1

# Packet types — for now only PATCH_CHUNK; HEARTBEAT / FEEDBACK come
# in Week 2 when the closed-loop controller lands.
PKT_TYPE_PATCH_CHUNK = 0
PKT_TYPE_HEARTBEAT = 1  # reserved
PKT_TYPE_FEEDBACK = 2  # reserved (server → edge)

# Safe UDP payload size on standard Ethernet/Wi-Fi without IP-level
# fragmentation: 1500 MTU - 20 IP header - 8 UDP header = 1472. We
# reserve a little slack for VPN/encapsulation overhead.
MAX_UDP_PAYLOAD = 1400

# After subtracting our 32-byte application header.
MAX_PAYLOAD_BYTES = MAX_UDP_PAYLOAD - 32  # = 1368
assert MAX_PAYLOAD_BYTES > 0

# ---------- Header struct ----------
#
# Layout (big-endian, 32 bytes total):
#
#   offset  field          size  notes
#   ------  -------------  ----  ------------------------------------
#     0     magic           4    b"RPEC"
#     4     version         1    PROTOCOL_VERSION
#     5     pkt_type        1    PKT_TYPE_*
#     6     reserved8       2    must be 0, future flags / padding
#     8     frame_id        4    uint32, monotonic
#    12     det_id          1    uint8, 0..255 patches per frame
#    13     quality         1    uint8, JPEG quality used (1..100)
#    14     chunk_idx       2    uint16, this chunk's index in patch
#    16     chunk_count     2    uint16, total chunks for this patch
#    18     payload_len     2    uint16, bytes of payload following
#    20     bbox_x1         2    int16, original bbox in frame coords
#    22     bbox_y1         2    int16
#    24     bbox_x2         2    int16
#    26     bbox_y2         2    int16
#    28     conf_q14        2    uint16, confidence * 10000 (0..10000)
#    30     reserved16      2    must be 0
#   ------  -------------  ----
#    32

HEADER_FORMAT = "!4sBBH I BB H H H hhhh H H"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 32, f"header size mismatch: {HEADER_SIZE}"


@dataclass
class PacketHeader:
    """Parsed packet header. Mirrors the wire layout above."""

    frame_id: int
    det_id: int
    quality: int
    chunk_idx: int
    chunk_count: int
    payload_len: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in frame coords
    confidence: float  # 0.0..1.0
    pkt_type: int = PKT_TYPE_PATCH_CHUNK
    version: int = PROTOCOL_VERSION

    def pack(self) -> bytes:
        x1, y1, x2, y2 = self.bbox
        conf_q14 = int(round(self.confidence * 10000))
        conf_q14 = max(0, min(10000, conf_q14))
        return struct.pack(
            HEADER_FORMAT,
            MAGIC,
            self.version,
            self.pkt_type,
            0,  # reserved8
            self.frame_id,
            self.det_id,
            self.quality,
            self.chunk_idx,
            self.chunk_count,
            self.payload_len,
            x1, y1, x2, y2,
            conf_q14,
            0,  # reserved16
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "PacketHeader":
        """Parse a header. Raises ValueError if buf is too short or magic/version is wrong."""
        if len(buf) < HEADER_SIZE:
            raise ValueError(
                f"packet too short for header: {len(buf)} < {HEADER_SIZE}"
            )
        (
            magic, version, pkt_type, _r8,
            frame_id, det_id, quality,
            chunk_idx, chunk_count, payload_len,
            x1, y1, x2, y2,
            conf_q14, _r16,
        ) = struct.unpack(HEADER_FORMAT, buf[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        if version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol version: {version} (want {PROTOCOL_VERSION})"
            )
        return cls(
            frame_id=frame_id,
            det_id=det_id,
            quality=quality,
            chunk_idx=chunk_idx,
            chunk_count=chunk_count,
            payload_len=payload_len,
            bbox=(x1, y1, x2, y2),
            confidence=conf_q14 / 10000.0,
            pkt_type=pkt_type,
            version=version,
        )


# ---------- Chunking helpers ----------

def chunk_patch_bytes(
    data: bytes,
    chunk_size: int = MAX_PAYLOAD_BYTES,
) -> List[bytes]:
    """Slice `data` into chunks of at most `chunk_size` bytes.

    Returns at least one chunk (an empty patch would still produce one
    zero-length chunk, but in practice JPEG output is never empty).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not data:
        return [b""]
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def build_packets(
    *,
    frame_id: int,
    det_id: int,
    quality: int,
    bbox: tuple[int, int, int, int],
    confidence: float,
    jpeg_bytes: bytes,
    chunk_size: int = MAX_PAYLOAD_BYTES,
) -> List[bytes]:
    """Turn one encoded patch into a list of ready-to-send UDP packets.

    Each returned bytes object is [HEADER || payload], ≤ MAX_UDP_PAYLOAD.
    """
    chunks = chunk_patch_bytes(jpeg_bytes, chunk_size=chunk_size)
    total = len(chunks)
    if total > 0xFFFF:
        raise ValueError(
            f"patch too large: {len(jpeg_bytes)} B → {total} chunks "
            f"(uint16 chunk_count max {0xFFFF})"
        )
    if not 0 <= det_id <= 0xFF:
        raise ValueError(f"det_id out of uint8 range: {det_id}")
    if not 1 <= quality <= 100:
        raise ValueError(f"quality out of range: {quality}")

    packets: List[bytes] = []
    for idx, chunk in enumerate(chunks):
        hdr = PacketHeader(
            frame_id=frame_id,
            det_id=det_id,
            quality=quality,
            chunk_idx=idx,
            chunk_count=total,
            payload_len=len(chunk),
            bbox=bbox,
            confidence=confidence,
        )
        packets.append(hdr.pack() + chunk)
    return packets
