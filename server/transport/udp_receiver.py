"""UDP receiver + chunk reassembly + frame completion (server node).

Day 5: yielded ReceivedPatch when each patch completed.
Day 6: also yields ReceivedFrame when all expected patches for a
frame_id arrive (per the FRAME_HEADER announcement).
Week 2 prep: ReceivedPatch carries expanded_bbox (from chunk 0 prefix).

Design notes
------------
Two-level state: frame-level (expected count, frame size, set by
FRAME_HEADER) and patch-level (chunk reassembly). If a FRAME_HEADER is
lost, we still reassemble patches; we just can't emit "frame complete"
for that frame_id and it eventually fires via frame_ttl_s.

TTL backstops: patch_ttl_s (default 200 ms) bounds chunk waiting,
frame_ttl_s (default 500 ms) bounds frame waiting. After these expire,
whatever's accumulated is emitted, marked incomplete.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from common.packet import (
    HEADER_SIZE,
    MAX_UDP_PAYLOAD,
    PKT_TYPE_FRAME_HEADER,
    PKT_TYPE_PATCH_CHUNK,
    PacketHeader,
    parse_frame_header_payload,
    unpack_patch_meta_prefix,
    PATCH_META_PREFIX_SIZE,
)


PatchKey = Tuple[int, int]  # (frame_id, det_id)


@dataclass
class _PatchAssembly:
    frame_id: int
    det_id: int
    quality: int
    bbox: tuple[int, int, int, int]
    confidence: float
    chunk_count: int
    chunks: List[Optional[bytes]] = field(default_factory=list)
    received_chunks: int = 0
    first_seen_s: float = 0.0
    expanded_bbox: Optional[tuple[int, int, int, int]] = None

    def is_complete(self) -> bool:
        return self.received_chunks == self.chunk_count

    def assemble(self) -> bytes:
        return b"".join(c if c is not None else b"" for c in self.chunks)


@dataclass
class _FrameAssembly:
    frame_id: int
    expected_patches: int  # -1 if FRAME_HEADER not yet seen
    frame_w: int
    frame_h: int
    first_seen_s: float
    patches: Dict[int, "ReceivedPatch"] = field(default_factory=dict)


@dataclass
class ReceivedPatch:
    frame_id: int
    det_id: int
    quality: int
    bbox: tuple[int, int, int, int]
    expanded_bbox: tuple[int, int, int, int] | None   # from chunk 0 prefix; None if chunk 0 lost
    confidence: float
    data: bytes
    complete: bool
    chunks_received: int
    chunks_expected: int

    @property
    def loss_ratio(self) -> float:
        if self.chunks_expected == 0:
            return 0.0
        return (self.chunks_expected - self.chunks_received) / self.chunks_expected


@dataclass
class ReceivedFrame:
    """All patches received for one frame_id, ready for stitching.

    Emitted when either (a) FRAME_HEADER's expected count is met, or
    (b) frame_ttl_s expires (partial frame).
    """

    frame_id: int
    frame_w: int
    frame_h: int
    expected_patches: int
    patches: List[ReceivedPatch]
    header_seen: bool       # False if FRAME_HEADER never arrived
    complete: bool          # True iff all expected patches present & complete

    @property
    def n_received(self) -> int:
        return len(self.patches)

    @property
    def n_complete_patches(self) -> int:
        return sum(1 for p in self.patches if p.complete)


@dataclass
class ReceiveStats:
    packets_received: int = 0
    packets_dropped_bad_header: int = 0
    packets_dropped_bad_payload: int = 0
    duplicate_chunks: int = 0
    patches_complete: int = 0
    patches_incomplete_ttl: int = 0
    bytes_received: int = 0
    frame_headers_received: int = 0
    frames_complete: int = 0
    frames_partial_ttl: int = 0
    orphan_patches: int = 0   # patches arrived without FRAME_HEADER


class UDPReceiver:
    """Bind UDP port, yield ReceivedPatch and ReceivedFrame events."""

    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        recv_buffer_bytes: int = 1 << 22,
        patch_ttl_s: float = 0.200,
        frame_ttl_s: float = 0.500,
    ) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buffer_bytes
            )
        except OSError:
            pass
        self._sock.bind((bind_host, int(bind_port)))
        self._sock.setblocking(False)

        self._patch_ttl_s = float(patch_ttl_s)
        self._frame_ttl_s = float(frame_ttl_s)
        self._inflight_patches: Dict[PatchKey, _PatchAssembly] = {}
        self._inflight_frames: Dict[int, _FrameAssembly] = {}
        self.stats = ReceiveStats()

    def poll(self, timeout_s: float = 0.01) -> Iterator[object]:
        """Drain socket up to `timeout_s` s. Yields ReceivedPatch and
        ReceivedFrame mixed — caller dispatches by type."""
        deadline = time.perf_counter() + max(0.0, timeout_s)

        while True:
            try:
                buf, _addr = self._sock.recvfrom(MAX_UDP_PAYLOAD)
            except BlockingIOError:
                if time.perf_counter() >= deadline:
                    break
                time.sleep(0.0005)
                continue

            for event in self._ingest(buf):
                yield event

            if time.perf_counter() >= deadline:
                break

        yield from self._sweep_expired_patches()
        yield from self._sweep_expired_frames()

    def flush(self) -> List[object]:
        """Force-emit everything in flight (at shutdown)."""
        out: List[object] = []
        out.extend(self._sweep_expired_patches(force_all=True))
        out.extend(self._sweep_expired_frames(force_all=True))
        return out

    def _ingest(self, buf: bytes) -> Iterator[object]:
        self.stats.packets_received += 1
        self.stats.bytes_received += len(buf)

        try:
            hdr = PacketHeader.unpack(buf)
        except ValueError:
            self.stats.packets_dropped_bad_header += 1
            return

        raw_payload = buf[HEADER_SIZE : HEADER_SIZE + hdr.payload_len]
        if len(raw_payload) != hdr.payload_len:
            self.stats.packets_dropped_bad_payload += 1
            return

        # Dispatch by packet type FIRST. Chunk-0 prefix only applies to PATCH_CHUNK,
        # not FRAME_HEADER (whose payload is the 5-byte n_patches/w/h struct).
        if hdr.pkt_type == PKT_TYPE_FRAME_HEADER:
            self._handle_frame_header(hdr.frame_id, raw_payload)
            yield from self._maybe_emit_frame(hdr.frame_id)
            return
        if hdr.pkt_type != PKT_TYPE_PATCH_CHUNK:
            return

        # PATCH_CHUNK only past this point.
        # Chunk 0 carries an 8-byte expanded_bbox prefix; later chunks carry JPEG only.
        if hdr.chunk_idx == 0:
            try:
                expanded_bbox = unpack_patch_meta_prefix(raw_payload)
            except ValueError:
                self.stats.packets_dropped_bad_payload += 1
                return
            payload = raw_payload[PATCH_META_PREFIX_SIZE:]
        else:
            expanded_bbox = None
            payload = raw_payload

        rp = self._handle_chunk(hdr, payload, expanded_bbox)
        if rp is not None:
            yield rp
            self._attach_to_frame(rp)
            yield from self._maybe_emit_frame(rp.frame_id)

    def _handle_frame_header(self, frame_id: int, payload: bytes) -> None:
        try:
            n_patches, frame_w, frame_h = parse_frame_header_payload(payload)
        except ValueError:
            self.stats.packets_dropped_bad_payload += 1
            return

        self.stats.frame_headers_received += 1
        fr = self._inflight_frames.get(frame_id)
        if fr is None:
            self._inflight_frames[frame_id] = _FrameAssembly(
                frame_id=frame_id,
                expected_patches=n_patches,
                frame_w=frame_w, frame_h=frame_h,
                first_seen_s=time.perf_counter(),
            )
        else:
            # Header arrived after some patches.
            fr.expected_patches = n_patches
            fr.frame_w = frame_w
            fr.frame_h = frame_h

    def _handle_chunk(
        self,
        hdr: PacketHeader,
        payload: bytes,
        expanded_bbox: Optional[tuple[int, int, int, int]],
    ) -> Optional[ReceivedPatch]:
        if hdr.chunk_idx >= hdr.chunk_count or hdr.chunk_count == 0:
            self.stats.packets_dropped_bad_payload += 1
            return None

        key: PatchKey = (hdr.frame_id, hdr.det_id)
        asm = self._inflight_patches.get(key)
        if asm is None:
            asm = _PatchAssembly(
                frame_id=hdr.frame_id, det_id=hdr.det_id,
                quality=hdr.quality, bbox=hdr.bbox,
                confidence=hdr.confidence,
                chunk_count=hdr.chunk_count,
                chunks=[None] * hdr.chunk_count,
                first_seen_s=time.perf_counter(),
            )
            self._inflight_patches[key] = asm
        elif asm.chunk_count != hdr.chunk_count:
            self.stats.packets_dropped_bad_payload += 1
            del self._inflight_patches[key]
            return None

        if asm.chunks[hdr.chunk_idx] is not None:
            self.stats.duplicate_chunks += 1
            return None

        asm.chunks[hdr.chunk_idx] = payload
        if hdr.chunk_idx == 0:
            asm.expanded_bbox = expanded_bbox
        asm.received_chunks += 1

        if asm.is_complete():
            del self._inflight_patches[key]
            self.stats.patches_complete += 1
            return ReceivedPatch(
                frame_id=asm.frame_id, det_id=asm.det_id,
                quality=asm.quality, bbox=asm.bbox,
                expanded_bbox=asm.expanded_bbox,
                confidence=asm.confidence,
                data=asm.assemble(),
                complete=True,
                chunks_received=asm.received_chunks,
                chunks_expected=asm.chunk_count,
            )
        return None

    def _attach_to_frame(self, rp: ReceivedPatch) -> None:
        fr = self._inflight_frames.get(rp.frame_id)
        if fr is None:
            fr = _FrameAssembly(
                frame_id=rp.frame_id, expected_patches=-1,
                frame_w=0, frame_h=0,
                first_seen_s=time.perf_counter(),
            )
            self._inflight_frames[rp.frame_id] = fr
            self.stats.orphan_patches += 1
        fr.patches[rp.det_id] = rp

    def _maybe_emit_frame(self, frame_id: int) -> Iterator[ReceivedFrame]:
        fr = self._inflight_frames.get(frame_id)
        if fr is None:
            return
        if fr.expected_patches < 0:
            return
        if len(fr.patches) < fr.expected_patches:
            return

        del self._inflight_frames[frame_id]
        self.stats.frames_complete += 1
        all_complete = all(p.complete for p in fr.patches.values())
        yield ReceivedFrame(
            frame_id=fr.frame_id,
            frame_w=fr.frame_w, frame_h=fr.frame_h,
            expected_patches=fr.expected_patches,
            patches=sorted(fr.patches.values(), key=lambda p: p.det_id),
            header_seen=True,
            complete=all_complete,
        )

    def _sweep_expired_patches(
        self, force_all: bool = False,
    ) -> Iterator[ReceivedPatch]:
        if not self._inflight_patches:
            return
        now = time.perf_counter()
        expired = []
        for key, asm in self._inflight_patches.items():
            if force_all or (now - asm.first_seen_s) >= self._patch_ttl_s:
                expired.append(key)
        for key in expired:
            asm = self._inflight_patches.pop(key)
            self.stats.patches_incomplete_ttl += 1
            rp = ReceivedPatch(
                frame_id=asm.frame_id, det_id=asm.det_id,
                quality=asm.quality, bbox=asm.bbox,
                expanded_bbox=asm.expanded_bbox,
                confidence=asm.confidence,
                data=asm.assemble(),
                complete=False,
                chunks_received=asm.received_chunks,
                chunks_expected=asm.chunk_count,
            )
            yield rp
            self._attach_to_frame(rp)

    def _sweep_expired_frames(
        self, force_all: bool = False,
    ) -> Iterator[ReceivedFrame]:
        if not self._inflight_frames:
            return
        now = time.perf_counter()
        expired = []
        for fid, fr in self._inflight_frames.items():
            if force_all or (now - fr.first_seen_s) >= self._frame_ttl_s:
                expired.append(fid)
        for fid in expired:
            fr = self._inflight_frames.pop(fid)
            self.stats.frames_partial_ttl += 1
            expected = (
                fr.expected_patches if fr.expected_patches >= 0
                else len(fr.patches)
            )
            all_complete = (
                expected > 0
                and len(fr.patches) == expected
                and all(p.complete for p in fr.patches.values())
            )
            yield ReceivedFrame(
                frame_id=fr.frame_id,
                frame_w=fr.frame_w, frame_h=fr.frame_h,
                expected_patches=expected,
                patches=sorted(fr.patches.values(), key=lambda p: p.det_id),
                header_seen=(fr.expected_patches >= 0),
                complete=all_complete,
            )

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "UDPReceiver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
