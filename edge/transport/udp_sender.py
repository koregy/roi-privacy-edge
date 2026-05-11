"""UDP sender for chunked patch packets (edge node).

Holds a single connected UDP socket for the lifetime of the capture loop.
For each frame, accepts the list of EncodedPatch and turns them into
UDP packets via common.packet.build_packets, then sendto's each one.

Why one socket
--------------
sendto() does a syscall every call; we don't want to also pay socket()
+ close() per call. Keep it open, reuse it.

Pacing
------
For now we just sendto() back-to-back. If the receiver loses chunks
under bursty load, the right fix is a small per-chunk sleep (≈100 µs);
that's a tuning knob to add when we have empirical loss data. The
calling pipeline already paces itself naturally at the camera rate.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Sequence

from common.packet import build_packets, MAX_UDP_PAYLOAD
from edge.patch.jpeg_encoder import EncodedPatch


@dataclass
class SendStats:
    """Cumulative stats for the sender. The dashboard / logger reads this."""

    packets_sent: int = 0
    bytes_sent: int = 0          # total wire bytes including headers
    patches_sent: int = 0        # number of EncodedPatch successfully shipped
    send_errors: int = 0

    def reset(self) -> None:
        self.packets_sent = 0
        self.bytes_sent = 0
        self.patches_sent = 0
        self.send_errors = 0


class UDPSender:
    """Edge-side UDP transmitter."""

    def __init__(
        self,
        server_host: str,
        server_port: int,
        send_buffer_bytes: int = 1 << 20,  # 1 MiB
        per_chunk_sleep_us: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        server_host, server_port :
            Destination address. Same-LAN Wi-Fi for our demo.
        send_buffer_bytes :
            SO_SNDBUF size hint. The kernel may cap this; that's fine.
            A larger buffer reduces sendto() blocking when chunks burst.
        per_chunk_sleep_us :
            Optional microsecond sleep between chunks. 0 = no pacing.
            Set to ~100 if the receiver starts losing chunks under load.
        """
        self.server_addr = (server_host, int(server_port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, send_buffer_bytes
            )
        except OSError:
            # Some kernels reject large SO_SNDBUF; the default is fine.
            pass

        # Convert to seconds once, avoid the division per chunk.
        self._sleep_s = max(0.0, per_chunk_sleep_us) / 1_000_000.0

        self.stats = SendStats()

    # ---------- Per-patch / per-frame send paths ----------

    def send_patch(self, enc: EncodedPatch) -> int:
        """Send one EncodedPatch as N chunks. Returns chunks sent."""
        packets = build_packets(
            frame_id=enc.frame_id,
            det_id=enc.det_id,
            quality=enc.quality,
            bbox=enc.original_bbox,
            confidence=enc.conf,
            jpeg_bytes=enc.data,
        )
        sent = 0
        for pkt in packets:
            # Defensive — should never trigger, but the wire format
            # depends on packets fitting in one datagram.
            if len(pkt) > MAX_UDP_PAYLOAD:
                self.stats.send_errors += 1
                continue
            try:
                self._sock.sendto(pkt, self.server_addr)
            except OSError:
                self.stats.send_errors += 1
                continue
            self.stats.packets_sent += 1
            self.stats.bytes_sent += len(pkt)
            sent += 1
            if self._sleep_s > 0.0:
                time.sleep(self._sleep_s)
        if sent > 0:
            self.stats.patches_sent += 1
        return sent

    def send_frame(self, encoded: Sequence[EncodedPatch]) -> int:
        """Send all encoded patches for one frame. Returns total chunks sent."""
        total = 0
        for enc in encoded:
            total += self.send_patch(enc)
        return total

    # ---------- Lifecycle ----------

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "UDPSender":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
