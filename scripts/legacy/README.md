# Legacy scripts

Day 5 prototype scripts kept for reference. **Not maintained.**

- `run_edge_send.py`: single-shot edge sender (precursor to `run_edge_video.py`).
  Contains a known TypeError at line 80 (`send_frame` signature change).
- `run_server_recv.py`: single-shot server receiver (precursor to `run_server_stitch.py`).
- `test_inference.py`: standalone YOLOv8TRT smoke test (superseded by
  `test_patch_pipeline.py` and `test_udp_loopback.py`).

For Week 1 entry points, see `scripts/run_edge_video.py` and
`scripts/run_server_stitch.py`.