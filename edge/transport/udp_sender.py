"""UDP sender for chunked patch packets (edge node).

Day 6: adds FRAME_HEADER packet support. send_frame() now sends one
FRAME_HEADER packet first (announcing n_patches + frame size), then the
per-patch chunks. The receiver uses this to detect frame completion.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Sequence

from common.packet import (
    MAX_UDP_PAYLOAD,
    build_frame_header_packet,
    build_packets,
)
from edge.patch.jpeg_encoder import EncodedPatch


@dataclass
class SendStats:
    packets_sent: int = 0
    bytes_sent: int = 0
    patches_sent: int = 0
    frames_sent: int = 0
    send_errors: int = 0

    def reset(self) -> None:
        self.packets_sent = 0
        self.bytes_sent = 0
        self.patches_sent = 0
        self.frames_sent = 0
        self.send_errors = 0


class UDPSender:
    def __init__(
        self,
        server_host: str,
        server_port: int,
        send_buffer_bytes: int = 1 << 20,
        per_chunk_sleep_us: float = 0.0,
    ) -> None:
        self.server_addr = (server_host, int(server_port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, send_buffer_bytes
            )
        except OSError:
            pass
        self._sleep_s = max(0.0, per_chunk_sleep_us) / 1_000_000.0
        self.stats = SendStats()

    def _sendto(self, packet: bytes) -> bool:
        if len(packet) > MAX_UDP_PAYLOAD:
            self.stats.send_errors += 1
            return False
        try:
            self._sock.sendto(packet, self.server_addr)
        except OSError:
            self.stats.send_errors += 1
            return False
        self.stats.packets_sent += 1
        self.stats.bytes_sent += len(packet)
        return True

    def send_frame_header(
        self, *, frame_id: int, n_patches: int, frame_w: int, frame_h: int,
    ) -> bool:
        """Send a single FRAME_HEADER announcing this frame's expected
        patch count and source image size. Caller must invoke this BEFORE
        send_patch() / send_frame() for the same frame_id."""
        pkt = build_frame_header_packet(
            frame_id=frame_id, n_patches=n_patches,
            frame_w=frame_w, frame_h=frame_h,
        )
        return self._sendto(pkt)

    def send_patch(self, enc: EncodedPatch) -> int:
        """Send one EncodedPatch as N chunks. Returns chunks sent."""
        packets = build_packets(
            frame_id=enc.frame_id, det_id=enc.det_id,
            quality=enc.quality,
            bbox=enc.original_bbox,
            expanded_bbox=enc.expanded_bbox,
            confidence=enc.conf,
            jpeg_bytes=enc.data,
        )
        sent = 0
        for pkt in packets:
            if self._sendto(pkt):
                sent += 1
            if self._sleep_s > 0.0:
                time.sleep(self._sleep_s)
        if sent > 0:
            self.stats.patches_sent += 1
        return sent

    def send_frame(
        self,
        *,
        frame_id: int,
        encoded: Sequence[EncodedPatch],
        frame_w: int,
        frame_h: int,
    ) -> int:
        """Send a full frame: FRAME_HEADER + all patch chunks.

        Returns total UDP packets sent (1 header + N chunks).
        """
        sent = 0
        if self.send_frame_header(
            frame_id=frame_id,
            n_patches=len(encoded),
            frame_w=frame_w, frame_h=frame_h,
        ):
            sent += 1
        for enc in encoded:
            sent += self.send_patch(enc)
        self.stats.frames_sent += 1
        return sent

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "UDPSender":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
