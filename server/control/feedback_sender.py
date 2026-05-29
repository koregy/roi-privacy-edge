"""UDP sender for server→edge FEEDBACK packets.

Sends a small (33B) packet with the new target JPEG quality whenever
the adaptive controller's output changes. Used by the closed-loop
adaptive system to push quality decisions to the edge encoder.

Send redundancy >= 2 protects against packet loss on the feedback
channel: identical packets are idempotent on the edge (each one is
just "set quality to X").
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

from common.packet import build_feedback_packet


@dataclass
class FeedbackSenderStats:
    sends: int = 0          # number of send_quality calls
    packets_sent: int = 0   # total UDP packets sent (sends * redundancy)
    bytes_sent: int = 0
    send_errors: int = 0


class FeedbackSender:
    """UDP socket pointed at the edge's feedback receiver."""

    def __init__(
        self,
        edge_host: str,
        edge_port: int,
        redundancy: int = 2,
    ) -> None:
        if redundancy < 1:
            raise ValueError(f"redundancy must be >= 1, got {redundancy}")
        self.edge_host = edge_host
        self.edge_port = edge_port
        self.redundancy = redundancy
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.stats = FeedbackSenderStats()

    def send_quality(self, *, frame_id: int, target_quality: int) -> int:
        """Send target_quality to edge. Returns number of UDP packets sent."""
        try:
            pkt = build_feedback_packet(
                frame_id=frame_id,
                target_quality=target_quality,
            )
        except ValueError:
            self.stats.send_errors += 1
            return 0

        sent = 0
        for _ in range(self.redundancy):
            try:
                self._sock.sendto(pkt, (self.edge_host, self.edge_port))
                sent += 1
                self.stats.packets_sent += 1
                self.stats.bytes_sent += len(pkt)
            except OSError:
                self.stats.send_errors += 1
        if sent > 0:
            self.stats.sends += 1
        return sent

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "FeedbackSender":
        return self

    def __exit__(self, *exc) -> None:
        self.close()