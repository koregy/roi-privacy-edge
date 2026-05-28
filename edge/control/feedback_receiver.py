"""Non-blocking UDP receiver for server→edge FEEDBACK packets.

The edge main loop polls this receiver once per frame (non-blocking).
If a FEEDBACK packet is available, the latest target_quality is
returned to the caller, which then updates the JPEG encoder.

Stale feedback (older frame_id than the latest received) is discarded
to avoid jitter from out-of-order arrivals.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Optional

from common.packet import (
    PacketHeader,
    PKT_TYPE_FEEDBACK,
    HEADER_SIZE,
    parse_feedback_payload,
)


@dataclass
class FeedbackReceiverStats:
    packets_received: int = 0
    bytes_received: int = 0
    bad_header: int = 0
    bad_payload: int = 0
    wrong_type: int = 0
    stale_frame_id: int = 0
    quality_changes: int = 0


class FeedbackReceiver:
    """Non-blocking UDP receiver. Caller polls once per frame."""

    BUFFER_SIZE = 65535   # max UDP datagram

    def __init__(self, bind_host: str, bind_port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind_host, bind_port))
        self._sock.setblocking(False)
        self.bind_host = bind_host
        self.bind_port = bind_port
        self._last_frame_id: int = -1
        self._current_quality: Optional[int] = None
        self.stats = FeedbackReceiverStats()

    @property
    def current_quality(self) -> Optional[int]:
        """Latest target quality received, or None if no feedback yet."""
        return self._current_quality

    def poll(self) -> Optional[int]:
        """Drain pending feedback packets non-blockingly.

        Returns the new target_quality if it changed from the previous
        value, else None. Always processes ALL pending packets so the
        socket buffer doesn't accumulate stale data.
        """
        prev_quality = self._current_quality
        try:
            while True:
                try:
                    buf, _ = self._sock.recvfrom(self.BUFFER_SIZE)
                except BlockingIOError:
                    break  # no more packets pending
                self.stats.packets_received += 1
                self.stats.bytes_received += len(buf)
                self._ingest(buf)
        except OSError:
            # socket-level error; ignore for now (e.g., transient ICMP)
            pass

        if self._current_quality != prev_quality and self._current_quality is not None:
            self.stats.quality_changes += 1
            return self._current_quality
        return None

    def _ingest(self, buf: bytes) -> None:
        # Header
        try:
            hdr = PacketHeader.unpack(buf)
        except ValueError:
            self.stats.bad_header += 1
            return

        if hdr.pkt_type != PKT_TYPE_FEEDBACK:
            self.stats.wrong_type += 1
            return

        # Stale check
        if hdr.frame_id < self._last_frame_id:
            self.stats.stale_frame_id += 1
            return

        # Payload
        raw_payload = buf[HEADER_SIZE : HEADER_SIZE + hdr.payload_len]
        if len(raw_payload) != hdr.payload_len:
            self.stats.bad_payload += 1
            return

        try:
            target_quality = parse_feedback_payload(raw_payload)
        except ValueError:
            self.stats.bad_payload += 1
            return

        # Accept
        self._last_frame_id = hdr.frame_id
        self._current_quality = target_quality

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "FeedbackReceiver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
