# roi-privacy-edge — Chat Handoff

이 문서는 새 채팅 세션 시작 시 첫 메시지로 붙여넣어, Claude가 변수명/시그니처를 추측하지 않고 ground truth에서 시작하게 하기 위함. 작업이 진척될 때마다 "현재 상태" 섹션과 "알려진 부채" 섹션 업데이트.

---

## 프로젝트 개요

**제목:** RoI-Based Privacy-Preserving Edge Streaming Pipeline
**일정:** 3주 (May 9 – May 29, 2026)
**마감:** May 29
**계획서:** `ICT_Module4_Proposal_v2.docx`

**구성:**
- Edge: Jetson Orin Nano (192.168.35.97), `~/roi-privacy-edge`
- Server: 노트북 (192.168.35.153), `~/projects/roi-privacy-edge`
- 같은 서브넷 Wi-Fi, SSH 키로 GitHub 연동

**핵심 아이디어:** Edge에서 YOLO로 person detect → person 영역(RoI)만 잘라 JPEG로 압축 → UDP로 전송 → Server에서 받아 빈 배경에 stitching. RoI 외 영역은 송신 안 함 → privacy preservation.

---

## 현재 상태 (마지막 업데이트: Week 2 Day 1 시작 직전)

### Week 1 완료 (Day 1–6)
- Day 1–2: YOLOv8n FP16 TensorRT engine — 8.8MB, GPU compute 4.33ms, e2e 27.6ms
- Day 3–4: patch extractor + JPEG encoder — 84KB/frame @ q=75, 4 persons
- Day 5: UDP transport — 32B header + chunking, loopback 100 frames 0 loss
- Day 6: FRAME_HEADER 패킷 + naive stitcher, 100 frames e2e (frames_complete=100/100)

### Week 2 Day 0 완료 (May 12, 부채 정리)
**부채 2번 (expanded_bbox wire 누락) 해결:**
- D안 적용: chunk 0 payload 앞에 8B `expanded_bbox` prefix
- packet.py / udp_sender.py / udp_receiver.py / stitcher / test_udp_loopback.py 수정
- loopback 3 verify 블록 PASS, e2e 100 frames `bad_pay=0` 회귀 없음
- 베이스라인 mp4 저장: `results/wk2_day0_expanded_bbox_baseline.mp4`

### Week 2 남은 일정
- Day 1–2: constraint simulator (drop/delay/noise + dashboard 슬라이더 preliminary)
- Day 3–4: recovery layer (IoU tracker + zero-order hold + adjacent frame interpolation)
- Day 5: adaptive JPEG quality controller (network state → quality, with hysteresis)
- Day 6: decision logic (Normal/Warning/Emergency) + closed-loop feedback (server → edge)
- Day 7: Stretch prep (ResNet18 dynamic shape ONNX, batch=1/4 검증)

---

## 핵심 코드 ground truth

새 채팅이 추측하지 않게 정확한 시그니처를 박아둠. 변경 시 이 섹션도 업데이트.

### `common/packet.py` — wire format 상수

```python
MAGIC = b"RPEC"
PROTOCOL_VERSION = 1

PKT_TYPE_PATCH_CHUNK = 0
PKT_TYPE_HEARTBEAT = 1   # reserved
PKT_TYPE_FEEDBACK = 2    # reserved (server → edge)
PKT_TYPE_FRAME_HEADER = 3

MAX_UDP_PAYLOAD = 1400
MAX_PAYLOAD_BYTES = 1368             # MAX_UDP_PAYLOAD - 32
HEADER_SIZE = 32
HEADER_FORMAT = "!4sBBH I BB H H H hhhh H H"

# FRAME_HEADER payload: n_patches(u8) + frame_w(u16) + frame_h(u16) = 5B
FRAME_HEADER_PAYLOAD_FORMAT = "!B H H"
FRAME_HEADER_PAYLOAD_SIZE = 5

# PATCH_CHUNK chunk 0 prefix: expanded_bbox uint16 × 4 = 8B
PATCH_META_PREFIX_FORMAT = "!HHHH"
PATCH_META_PREFIX_SIZE = 8
JPEG_CHUNK_SIZE = 1360   # MAX_PAYLOAD_BYTES - PATCH_META_PREFIX_SIZE, uniform across all chunks
```

**PATCH_CHUNK payload layout (중요):**
- `chunk_idx == 0`: [8B expanded_bbox prefix] + [JPEG bytes, first slice]
- `chunk_idx > 0`: [JPEG bytes]
- `original_bbox`는 header의 bbox 필드 (int16 × 4)에 들어감 (모든 chunk에 redundant)
- `expanded_bbox`는 chunk 0 prefix에만 (chunk 0 손실 = JPEG SOI 손실이라 patch 자체 못 살림 → redundancy 불필요)

### `common/packet.py` — 주요 함수

```python
def build_packets(
    *,
    frame_id: int,
    det_id: int,
    quality: int,
    bbox: tuple[int, int, int, int],              # original
    expanded_bbox: tuple[int, int, int, int],     # required
    confidence: float,
    jpeg_bytes: bytes,
    chunk_size: int = JPEG_CHUNK_SIZE,
) -> List[bytes]

def build_frame_header_packet(
    *, frame_id: int, n_patches: int, frame_w: int, frame_h: int,
) -> bytes

def pack_patch_meta_prefix(expanded_bbox) -> bytes        # 8B
def unpack_patch_meta_prefix(buf: bytes) -> tuple[int,int,int,int]
def parse_frame_header_payload(payload) -> tuple[int,int,int]
```

### `edge/detector/yolov8_trt.py`

```python
@dataclass
class Detection:
    x1: int; y1: int; x2: int; y2: int
    confidence: float
    class_id: int

class YOLOv8TRT:
    def __init__(self, engine_path: str): ...
    def detect(self, frame: np.ndarray) -> list[Detection]: ...
```

### `edge/patch/extractor.py`

```python
@dataclass
class Patch:
    frame_id: int
    det_id: int
    image: np.ndarray                             # cropped BGR uint8
    original_bbox: tuple[int, int, int, int]     # detector output, pre-margin
    expanded_bbox: tuple[int, int, int, int]     # after margin + frame clip
    conf: float

def extract_patches(frame, detections, frame_id) -> list[Patch]
```

### `edge/patch/jpeg_encoder.py`

```python
@dataclass
class EncodedPatch:
    frame_id: int
    det_id: int
    data: bytes                                   # JPEG bytes
    original_bbox: tuple[int, int, int, int]
    expanded_bbox: tuple[int, int, int, int]
    quality: int
    conf: float

class PatchJPEGEncoder:
    def __init__(self, default_quality: int = 75): ...
    def encode_many(self, patches: list[Patch]) -> list[EncodedPatch]: ...
```

### `server/transport/udp_receiver.py`

```python
@dataclass
class ReceivedPatch:
    frame_id: int
    det_id: int
    quality: int
    bbox: tuple[int, int, int, int]                          # original (from header)
    expanded_bbox: tuple[int, int, int, int] | None          # from chunk 0 prefix; None if chunk 0 lost
    confidence: float
    data: bytes
    complete: bool
    chunks_received: int
    chunks_expected: int

@dataclass
class ReceivedFrame:
    frame_id: int
    frame_w: int
    frame_h: int
    expected_patches: int
    patches: List[ReceivedPatch]
    header_seen: bool
    complete: bool
    # properties: n_received, n_complete_patches

class UDPReceiver:
    def __init__(
        self, bind_host, bind_port,
        recv_buffer_bytes: int = 1 << 22,
        patch_ttl_s: float = 0.200,
        frame_ttl_s: float = 0.500,
    ): ...
    def poll(self, timeout_s: float = 0.01) -> Iterator[object]: ...    # yields ReceivedPatch | ReceivedFrame
    def flush(self) -> List[object]: ...
    @property stats: ReceiveStats
```

**`ReceiveStats` 필드:**
`packets_received`, `packets_dropped_bad_header`, `packets_dropped_bad_payload`, `duplicate_chunks`, `patches_complete`, `patches_incomplete_ttl`, `bytes_received`, `frame_headers_received`, `frames_complete`, `frames_partial_ttl`, `orphan_patches`

---

## 파일 구조

```
roi-privacy-edge/
├── common/
│   ├── packet.py             # wire format, struct pack/unpack, build_packets, build_frame_header_packet
│   └── config.py             # DEFAULT_JPEG_QUALITY 등
├── edge/
│   ├── detector/
│   │   └── yolov8_trt.py     # YOLOv8TRT class, Detection dataclass
│   ├── patch/
│   │   ├── __init__.py       # exports PatchJPEGEncoder, extract_patches
│   │   ├── extractor.py      # extract_patches(), Patch dataclass
│   │   └── jpeg_encoder.py   # PatchJPEGEncoder, EncodedPatch dataclass
│   └── transport/
│       ├── __init__.py       # exports UDPSender
│       └── udp_sender.py     # UDPSender class, send_frame method
├── server/
│   ├── transport/
│   │   ├── __init__.py       # exports UDPReceiver, ReceivedPatch, ReceivedFrame
│   │   └── udp_receiver.py   # UDPReceiver class, reassembly, TTL sweeps
│   └── stitcher/
│       ├── __init__.py       # exports stitch_frame, StitchResult
│       └── naive.py          # stitch_frame(rf, draw_bbox=False) -> StitchResult
├── scripts/
│   ├── test_patch_pipeline.py     # extractor + encoder smoke test
│   ├── test_udp_loopback.py       # 127.0.0.1 e2e, 3 verify blocks
│   ├── run_edge_video.py          # edge sender (args: --server --port --source --engine --quality --max-frames --target-fps --sleep-us --log-every)
│   └── run_server_stitch.py       # server receiver + stitcher (args: --bind --port --output --png-dir --fps --draw-bbox --idle-timeout --ttl-ms)
├── engines/
│   └── yolov8n_fp16.engine        # TRT engine, 8.8MB, built on Jetson
├── data/
│   ├── test_images/persons.jpg
│   └── videos/person-bicycle-car-detection.mp4   # 768x432 12fps
└── results/
    ├── stitched.mp4                              # latest e2e output (overwritten each run)
    └── wk2_day0_expanded_bbox_baseline.mp4       # baseline (loss=0, normal)
```

---

## 환경 / 컨벤션

- Python 3.10+ (Jetson은 3.10, 노트북도 비슷)
- 들여쓰기: **스페이스 4칸**, 탭 금지
- 네이밍: `snake_case` (함수/변수), `PascalCase` (클래스)
- Type hints: `tuple[int, int, int, int]` (소문자, Python 3.9+ 스타일), `Optional` import 사용 OK
- Import: from-imports 그룹별 정렬 (stdlib → third-party → local)
- 실행: `cd ~/roi-privacy-edge && PYTHONPATH=. python -u scripts/...`
- Jetson에는 TensorRT + cuda Python bindings 있음, 노트북엔 둘 다 없음 → detector 호출은 Jetson에서만 가능

---

## 알려진 부채 / 추후 정리할 것들

1. **SIGINT 처리 (run_server_stitch.py)** — 가끔 [final] 줄 안 찍힘. 현재 코드에 `signal.signal` + `_running` flag + `flush()` + `[final]` 다 있어서 명확한 재현 케이스 없으면 미루는 중. constraint simulator 돌리다가 재현되면 그때 잡기.
2. ~~expanded_bbox wire 누락~~ — **Week 2 Day 0에 해결 완료**.
3. **quality sweep CSV plot 미생성** — Week 3 평가 단계에서 일괄.
4. **MOT17 다운 실패** (motchallenge.net 차단) — person-bicycle-car-detection.mp4로 충분히 굴러가는 중. Week 2 끝나고 데이터셋 다양성 필요해지면 다시.
5. **VideoWriterLazy 자동 백업** — `results/stitched.mp4`가 매번 덮어써짐. 의미 있는 이름으로 명시적 저장하거나 스크립트에 백업 로직 추가 고려.

---

## 새 채팅 시작 시 권장 패턴

1. 이 HANDOFF.md 통째로 첫 메시지에 붙여넣기
2. 그 다음 줄에 작업 요청 (예: "Week 2 Day 1 시작하자. constraint simulator 들어가자.")
3. 작업 중 파일 수정이 필요하면 **해당 파일의 관련 부분을 직접 보여주기** — Claude가 추측하면 변수명 틀리고 시간 낭비됨. grep + sed로 빠르게 발췌:
   ```bash
   grep -n "class\|@dataclass" path/to/file.py
   sed -n '40,60p' path/to/file.py
   ```
4. 변경 후엔 항상 syntax check + 가능하면 단위 테스트로 빠른 검증

---

## TODO before next chat

- [ ] git commit + push (부채 2번 변경)
- [ ] 노트북에 git pull
- [ ] 이 HANDOFF.md를 `docs/HANDOFF.md` 또는 repo root에 저장
- [ ] (선택) HANDOFF.md를 README에서 링크하거나 .git/ 옆에 두기

---

## 검증되지 않은 / 추측 포함 부분

다음 항목들은 이 채팅 컨텍스트에 실제 코드가 안 보였어서 추측 포함. 새 채팅 들어가기 전에 본인이 확인하고 수정 권장:

- **`UDPSender` 클래스의 정확한 메서드 시그니처** — `send_frame(frame_id, encoded, frame_w, frame_h)` 호출 패턴은 확인됨. 내부 메서드는 모름.
- **`stitch_frame()` 시그니처와 `StitchResult` dataclass 필드** — `stitch_frame(rf, draw_bbox=False) -> StitchResult`, `res.image`, `res.n_pasted` 사용 확인됨. 그 외는 모름.
- **`Detection` dataclass 정확한 필드** — `x1,y1,x2,y2,confidence,class_id`로 추정 (Week 1 시작 메시지 기반).
- **`common/config.py` 다른 상수들** — `DEFAULT_JPEG_QUALITY` 외엔 모름.

이 부분 채워서 v2 HANDOFF로 업데이트해두면 다음 채팅이 더 편해짐.