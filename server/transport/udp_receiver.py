"""UDP receiver + chunk reassembly (server node).

Receives chunks, groups them by (frame_id, det_id), and emits
ReceivedPatch objects once a patch is complete OR its TTL expires.

Design notes
------------
- One blocking socket. The caller pumps recv() from its main loop.
  No threads — keeps the data path linear and easy to reason about.
- Reassembly state is bounded by a TTL (default 200 ms). After that,
  any partial patch is yielded with `complete=False`. The recovery
  layer (Week 2) decides what to do — typically zero-order hold from
  the same tracked object's previous frame.
- We never trust the wire. Every header field is validated; bad
  packets are counted and dropped silently. The receiver must not
  crash regardless of what arrives on the socket.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from common.packet import (
    HEADER_SIZE,
    MAX_UDP_PAYLOAD,
    PKT_TYPE_PATCH_CHUNK,
    PacketHeader,
)


# (frame_id, det_id) — uniquely identifies a logical patch in flight
PatchKey = Tuple[int, int]


@dataclass
class _PatchAssembly:
    """In-flight reassembly state for one patch."""

    frame_id: int
    det_id: int
    quality: int
    bbox: tuple[int, int, int, int]
    confidence: float
    chunk_count: int
    # Sparse chunk storage indexed by chunk_idx. None = not yet received.
    chunks: List[Optional[bytes]] = field(default_factory=list)
    received_chunks: int = 0
    first_seen_s: float = 0.0

    def is_complete(self) -> bool:
        return self.received_chunks == self.chunk_count

    def assemble(self) -> bytes:
        """Concatenate received chunks. Missing slots → empty bytes."""
        return b"".join(c if c is not None else b"" for c in self.chunks)


@dataclass
class ReceivedPatch:
    """A patch handed to the next stage (stitching / recovery).

    Carries everything the server needs to either paste this patch into
    the stitched frame (if complete) or hand it off to the recovery
    layer (if incomplete).
    """

    frame_id: int
    det_id: int
    quality: int
    bbox: tuple[int, int, int, int]
    confidence: float
    data: bytes
    complete: bool
    chunks_received: int
    chunks_expected: int

    @property
    def loss_ratio(self) -> float:
        if self.chunks_expected == 0:
            return 0.0
        missing = self.chunks_expected - self.chunks_received
        return missing / self.chunks_expected


@dataclass
class ReceiveStats:
    packets_received: int = 0
    packets_dropped_bad_header: int = 0
    packets_dropped_bad_payload: int = 0
    duplicate_chunks: int = 0
    patches_complete: int = 0
    patches_incomplete_ttl: int = 0
    bytes_received: int = 0


class UDPReceiver:
    """Bind a UDP port, recv chunks, yield ReceivedPatch.

    Typical use:

        with UDPReceiver("0.0.0.0", 9999) as rx:
            while running:
                for patch in rx.poll(timeout_s=0.01):
                    handle(patch)
    """

    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        recv_buffer_bytes: int = 1 << 22,   # 4 MiB
        patch_ttl_s: float = 0.200,
    ) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buffer_bytes
            )
        except OSError:
            pass
        self._sock.bind((bind_host, int(bind_port)))
        self._sock.setblocking(False)  # we use poll() with select-style timeout

        self._ttl_s = float(patch_ttl_s)
        self._inflight: Dict[PatchKey, _PatchAssembly] = {}
        self.stats = ReceiveStats()

    # ---------- Main poll loop ----------

    def poll(self, timeout_s: float = 0.01) -> Iterator[ReceivedPatch]:
        """Drain the socket for up to `timeout_s` seconds.

        Yields completed patches as they finish, plus any inflight
        patches whose TTL expired during this poll. Safe to call
        repeatedly from a tight loop.
        """
        deadline = time.perf_counter() + max(0.0, timeout_s)

        # Drain whatever's queued in the kernel buffer right now.
        while True:
            try:
                buf, _addr = self._sock.recvfrom(MAX_UDP_PAYLOAD)
            except BlockingIOError:
                # Nothing waiting. Sleep a tiny bit if we still have
                # budget; otherwise return.
                if time.perf_counter() >= deadline:
                    break
                # Short sleep keeps CPU low while we wait for the next
                # camera-driven burst (~33 ms apart at 30 FPS).
                time.sleep(0.0005)
                continue

            completed = self._ingest(buf)
            if completed is not None:
                yield completed

            if time.perf_counter() >= deadline:
                break

        # After draining, sweep expired patches.
        yield from self._sweep_expired()

    def flush_expired(self) -> List[ReceivedPatch]:
        """Force-yield any expired in-flight patches. Useful at shutdown."""
        return list(self._sweep_expired(force_all=False))

    # ---------- Internal: per-packet ingest ----------

    def _ingest(self, buf: bytes) -> Optional[ReceivedPatch]:
        self.stats.packets_received += 1
        self.stats.bytes_received += len(buf)

        try:
            hdr = PacketHeader.unpack(buf)
        except ValueError:
            self.stats.packets_dropped_bad_header += 1
            return None

        if hdr.pkt_type != PKT_TYPE_PATCH_CHUNK:
            # Future packet types (heartbeat, feedback) go elsewhere.
            return None

        payload = buf[HEADER_SIZE : HEADER_SIZE + hdr.payload_len]
        if len(payload) != hdr.payload_len:
            self.stats.packets_dropped_bad_payload += 1
            return None
        if hdr.chunk_idx >= hdr.chunk_count or hdr.chunk_count == 0:
            self.stats.packets_dropped_bad_payload += 1
            return None

        key: PatchKey = (hdr.frame_id, hdr.det_id)
        asm = self._inflight.get(key)
        if asm is None:
            asm = _PatchAssembly(
                frame_id=hdr.frame_id,
                det_id=hdr.det_id,
                quality=hdr.quality,
                bbox=hdr.bbox,
                confidence=hdr.confidence,
                chunk_count=hdr.chunk_count,
                chunks=[None] * hdr.chunk_count,
                first_seen_s=time.perf_counter(),
            )
            self._inflight[key] = asm
        elif asm.chunk_count != hdr.chunk_count:
            # Header disagreement across chunks of the same patch =
            # corrupt or stale state. Drop the whole thing.
            self.stats.packets_dropped_bad_payload += 1
            del self._inflight[key]
            return None

        if asm.chunks[hdr.chunk_idx] is not None:
            self.stats.duplicate_chunks += 1
            return None

        asm.chunks[hdr.chunk_idx] = payload
        asm.received_chunks += 1

        if asm.is_complete():
            del self._inflight[key]
            self.stats.patches_complete += 1
            return ReceivedPatch(
                frame_id=asm.frame_id,
                det_id=asm.det_id,
                quality=asm.quality,
                bbox=asm.bbox,
                confidence=asm.confidence,
                data=asm.assemble(),
                complete=True,
                chunks_received=asm.received_chunks,
                chunks_expected=asm.chunk_count,
            )

        return None

    # ---------- Internal: TTL sweep ----------

    def _sweep_expired(self, force_all: bool = False) -> Iterator[ReceivedPatch]:
        if not self._inflight:
            return
        now = time.perf_counter()
        expired_keys = []
        for key, asm in self._inflight.items():
            if force_all or (now - asm.first_seen_s) >= self._ttl_s:
                expired_keys.append(key)
        for key in expired_keys:
            asm = self._inflight.pop(key)
            self.stats.patches_incomplete_ttl += 1
            yield ReceivedPatch(
                frame_id=asm.frame_id,
                det_id=asm.det_id,
                quality=asm.quality,
                bbox=asm.bbox,
                confidence=asm.confidence,
                data=asm.assemble(),
                complete=False,
                chunks_received=asm.received_chunks,
                chunks_expected=asm.chunk_count,
            )

    # ---------- Lifecycle ----------

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "UDPReceiver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
