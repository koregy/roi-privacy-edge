"""UDP wire format shared by edge sender and server receiver.

Each packet on the wire = HEADER (32 B) + PAYLOAD (≤ MAX_PAYLOAD_BYTES).

Packet types:

PKT_TYPE_FRAME_HEADER  — announces "frame F has N patches, full image
                         is W×H". Sent once per frame, before its chunks.
                         Lets the receiver know when a frame is 'complete'
                         (i.e., all expected patches accounted for) vs.
                         still in flight.

PKT_TYPE_PATCH_CHUNK   — one slice of one patch. Multiple chunks per
                         patch, multiple patches per frame.

Why both
--------
Without FRAME_HEADER, the receiver can't tell "this frame is done" from
"more patches are still coming" — it has to fall back on timeouts, which
are coarse and racy. With FRAME_HEADER, we have an authoritative count
to compare against, and the timeout becomes a backstop for lost headers.

Frame headers are themselves UDP, so they can be lost. The receiver
handles this gracefully: if a chunk arrives for a frame_id we've never
seen a header for, we still reassemble that patch, we just can't emit a
"frame complete" event for that frame. The patch-level data still flows
through.

Loss of a single FRAME_HEADER affects exactly one frame's completion
signal; the next frame's header restores normal operation.

PATCH_CHUNK payload layout
--------------------------
Each PATCH_CHUNK packet on the wire = 32B header + payload.

Payload layout depends on chunk_idx:

  chunk_idx == 0:
      [8B patch metadata prefix] + [JPEG bytes, first slice]

  chunk_idx > 0:
      [JPEG bytes]

8B prefix carries expanded_bbox (uint16 × 4: x1, y1, x2, y2). This is
the bbox actually used for cropping on the edge side — the server uses
it to paste the patch back at the correct location.

Why payload prefix (not header field):
  - 32B header has no contiguous 8B slot (reserved bytes too narrow).
  - Redundancy across chunks would be wasted: losing chunk 0 means
    losing the JPEG SOI marker, so the patch is unrecoverable anyway.
    expanded_bbox only matters when the patch is recoverable.
  - original_bbox stays in the header for quick lookup without payload
    parse, and to preserve wire-level backwards readability of bbox.

All chunks (including chunk 0) carry at most JPEG_CHUNK_SIZE bytes of
JPEG data. Chunk 0's total payload is JPEG_CHUNK_SIZE + 8 bytes (still
within MAX_PAYLOAD_BYTES). Uniform JPEG slice size keeps reassembly
logic simple.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List

# ---------- Wire-format constants ----------

MAGIC = b"RPEC"
PROTOCOL_VERSION = 1

PKT_TYPE_PATCH_CHUNK = 0
PKT_TYPE_HEARTBEAT = 1   # reserved (Week 2)
PKT_TYPE_FEEDBACK = 2    # reserved (server → edge, Week 4)
PKT_TYPE_FRAME_HEADER = 3   # Day 6

MAX_UDP_PAYLOAD = 1400
MAX_PAYLOAD_BYTES = MAX_UDP_PAYLOAD - 32
assert MAX_PAYLOAD_BYTES > 0

# ---------- Packet header struct (32 B) ----------
HEADER_FORMAT = "!4sBBH I BB H H H hhhh H H"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 32, f"header size mismatch: {HEADER_SIZE}"

# ---------- Frame header payload struct ----------
# For PKT_TYPE_FRAME_HEADER, payload after the 32-byte header is:
#   n_patches : uint8   (1 B)
#   frame_w   : uint16  (2 B)
#   frame_h   : uint16  (2 B)
# Total payload = 5 bytes. Tiny packet on the wire (32 + 5 = 37 B).
FRAME_HEADER_PAYLOAD_FORMAT = "!B H H"
FRAME_HEADER_PAYLOAD_SIZE = struct.calcsize(FRAME_HEADER_PAYLOAD_FORMAT)
assert FRAME_HEADER_PAYLOAD_SIZE == 5

# ---------- PATCH_CHUNK metadata prefix (chunk 0 only) ----------
# 8 bytes packed before JPEG data in chunk_idx == 0:
#   exp_x1, exp_y1, exp_x2, exp_y2 : uint16 × 4
PATCH_META_PREFIX_FORMAT = "!HHHH"
PATCH_META_PREFIX_SIZE = struct.calcsize(PATCH_META_PREFIX_FORMAT)
assert PATCH_META_PREFIX_SIZE == 8

# JPEG slice size per chunk. Uniform across all chunks (including chunk 0).
# Chunk 0's total payload = JPEG_CHUNK_SIZE + PATCH_META_PREFIX_SIZE = 1368 B,
# which still fits within MAX_PAYLOAD_BYTES (1368). Non-zero chunks carry
# JPEG_CHUNK_SIZE bytes of JPEG, with room to spare in the UDP payload.
#
# A non-uniform layout (chunk 0: 1360 JPEG, others: 1368 JPEG) was considered
# but rejected: it changes chunk count only on rare size-boundary cases, and
# adds reassembly complexity (per-chunk slice-size dependency on chunk_idx).
JPEG_CHUNK_SIZE = MAX_PAYLOAD_BYTES - PATCH_META_PREFIX_SIZE
assert JPEG_CHUNK_SIZE == 1360


@dataclass
class PacketHeader:
    """Parsed packet header.

    The bbox / confidence / chunk_idx / chunk_count / quality fields are
    only meaningful for PKT_TYPE_PATCH_CHUNK packets. For PKT_TYPE_FRAME_
    HEADER, they are zero-filled by the sender and ignored by the
    receiver (real frame metadata is in the 5-byte payload).

    `bbox` here is the ORIGINAL detection bbox (pre-margin-expansion).
    The expanded bbox lives in the chunk 0 payload prefix.
    """

    frame_id: int
    det_id: int
    quality: int
    chunk_idx: int
    chunk_count: int
    payload_len: int
    bbox: tuple[int, int, int, int]
    confidence: float
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
            0,
            self.frame_id,
            self.det_id,
            self.quality,
            self.chunk_idx,
            self.chunk_count,
            self.payload_len,
            x1, y1, x2, y2,
            conf_q14,
            0,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "PacketHeader":
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
            frame_id=frame_id, det_id=det_id, quality=quality,
            chunk_idx=chunk_idx, chunk_count=chunk_count,
            payload_len=payload_len,
            bbox=(x1, y1, x2, y2),
            confidence=conf_q14 / 10000.0,
            pkt_type=pkt_type, version=version,
        )


# ---------- FRAME_HEADER builder / parser ----------

@dataclass
class FrameHeader:
    """Decoded contents of a PKT_TYPE_FRAME_HEADER packet."""

    frame_id: int
    n_patches: int
    frame_w: int
    frame_h: int


def build_frame_header_packet(
    *, frame_id: int, n_patches: int, frame_w: int, frame_h: int,
) -> bytes:
    """Build a wire-ready FRAME_HEADER packet (length = 37 B)."""
    if not 0 <= n_patches <= 0xFF:
        raise ValueError(f"n_patches out of uint8 range: {n_patches}")
    if not 0 <= frame_w <= 0xFFFF:
        raise ValueError(f"frame_w out of uint16 range: {frame_w}")
    if not 0 <= frame_h <= 0xFFFF:
        raise ValueError(f"frame_h out of uint16 range: {frame_h}")

    payload = struct.pack(
        FRAME_HEADER_PAYLOAD_FORMAT, n_patches, frame_w, frame_h
    )
    hdr = PacketHeader(
        frame_id=frame_id,
        det_id=0, quality=0, chunk_idx=0, chunk_count=0,
        payload_len=len(payload),
        bbox=(0, 0, 0, 0),
        confidence=0.0,
        pkt_type=PKT_TYPE_FRAME_HEADER,
    )
    return hdr.pack() + payload


def parse_frame_header_payload(payload: bytes) -> tuple[int, int, int]:
    """Parse 5-byte FRAME_HEADER payload → (n_patches, frame_w, frame_h)."""
    if len(payload) != FRAME_HEADER_PAYLOAD_SIZE:
        raise ValueError(
            f"FRAME_HEADER payload size mismatch: "
            f"{len(payload)} != {FRAME_HEADER_PAYLOAD_SIZE}"
        )
    return struct.unpack(FRAME_HEADER_PAYLOAD_FORMAT, payload)


# ---------- PATCH_CHUNK metadata prefix helpers ----------

def pack_patch_meta_prefix(expanded_bbox: tuple[int, int, int, int]) -> bytes:
    """Pack expanded_bbox into 8-byte prefix for chunk 0 payload."""
    ex1, ey1, ex2, ey2 = expanded_bbox
    for v, name in ((ex1, "ex1"), (ey1, "ey1"), (ex2, "ex2"), (ey2, "ey2")):
        if not 0 <= v <= 0xFFFF:
            raise ValueError(f"expanded_bbox.{name} out of uint16 range: {v}")
    return struct.pack(PATCH_META_PREFIX_FORMAT, ex1, ey1, ex2, ey2)


def unpack_patch_meta_prefix(buf: bytes) -> tuple[int, int, int, int]:
    """Parse 8-byte prefix → expanded_bbox (x1, y1, x2, y2).

    Raises ValueError if buf is shorter than 8 bytes.
    """
    if len(buf) < PATCH_META_PREFIX_SIZE:
        raise ValueError(
            f"patch meta prefix too short: {len(buf)} < {PATCH_META_PREFIX_SIZE}"
        )
    return struct.unpack(PATCH_META_PREFIX_FORMAT, buf[:PATCH_META_PREFIX_SIZE])


# ---------- Chunking helpers ----------

def chunk_patch_bytes(
    data: bytes, chunk_size: int = JPEG_CHUNK_SIZE,
) -> List[bytes]:
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
    expanded_bbox: tuple[int, int, int, int],
    confidence: float,
    jpeg_bytes: bytes,
    chunk_size: int = JPEG_CHUNK_SIZE,
) -> List[bytes]:
    """Build a list of wire-ready PATCH_CHUNK packets for one patch.

    `bbox` is the original (pre-expansion) detection bbox; it goes into
    every chunk header. `expanded_bbox` is the actual crop bbox; it is
    packed as an 8-byte prefix on chunk 0's payload only.
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

    meta_prefix = pack_patch_meta_prefix(expanded_bbox)

    packets: List[bytes] = []
    for idx, chunk in enumerate(chunks):
        # chunk 0 carries the 8B expanded_bbox prefix before JPEG bytes.
        if idx == 0:
            payload = meta_prefix + chunk
        else:
            payload = chunk

        hdr = PacketHeader(
            frame_id=frame_id, det_id=det_id, quality=quality,
            chunk_idx=idx, chunk_count=total,
            payload_len=len(payload),
            bbox=bbox, confidence=confidence,
        )
        packets.append(hdr.pack() + payload)

    return packets