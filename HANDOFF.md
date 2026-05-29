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

## 현재 상태 (마지막 업데이트: Week 2 Day 2 시작 직전 - May 24)

### Week 1 완료 (Day 9–15)
- Day 1–2: YOLOv8n FP16 TensorRT engine — 8.8MB, GPU compute 4.33ms, e2e 27.6ms
- Day 3–4: patch extractor + JPEG encoder — 84KB/frame @ q=75
- Day 5: UDP transport — 32B header + chunking, loopback 100 frames 0 loss
- Day 6: FRAME_HEADER 패킷 + naive stitcher, 100 frames e2e (frames_complete=100/100)

### Week 2 Day 0 (May 12) — expanded_bbox wire prefix 처리

### Week 2 Day 1 (May 13) — 코드 리뷰 + packet_filter seam 추가

### 11일 공백 (May 14–23)

### Week 2 재시작 Day 0 (May 24 오전) — Week 1 e2e 재검증
- 노트북 git pull, baseline 재실행: patches_complete=67, bad=0/0/0 ✅
- 일정 압축: 3주 → 5월 29일까지 6일

### Week 2 Day 1 완료 (May 24 오후) — Constraint Simulator (drop + delay)
- 신규: `server/transport/constraint_sim.py`
- 수정: `server/transport/__init__.py`, `scripts/run_server_stitch.py`
- Noise는 scope cut (적응형 컨트롤러와 묶여서 6일 일정에 부담)
- Test A (회귀, no flags): baseline 동일 ✅
- Test B (drop 30%, seed=42): seen=219, dropped=75, fhdr=58/100, patches_c/i=39/20
- Test C (delay 20-50ms): TTL(200ms) 내, 무손실, avg_delay=34.74ms
- Test D (drop 15% + delay 10-30ms): ChainedFilter 순서 효과 검증 — delay filter의 seen=188 < drop filter의 seen=219, drop된 31 패킷이 sleep 절약

### Week 2 Day 2 완료 (May 24 늦은 오후 ~ 밤) — Recovery + Reorder + Header Redundancy

#### 신규 파일
- `server/recovery/tracker.py` — IoU-based greedy multi-object tracker
- `server/recovery/zoh.py` — Zero-order Hold recovery layer
- `server/recovery/__init__.py`
- `server/transport/reorder.py` — Frame reorder buffer (min-heap, bounded latency)

#### 수정 파일
- `server/transport/udp_receiver.py`
  - `ReceivedPatch.recovered: bool = False` 필드 추가 (stitcher dispatch용)
  - **Dedup 메커니즘 추가:** `_emitted_frame_ids: set[int]` + 4군데 가드
    (handle_frame_header, attach_to_frame, maybe_emit_frame, sweep_expired_frames)
  - 이유: FRAME_HEADER redundancy=3과 결합 시 같은 frame_id가 3번 emit되는 버그 발견 (특히 detection=0인 frame). dedup으로 idempotent 보장.
- `server/stitcher/naive.py` — recovered 패치 빨간 테두리 (BGR red=0,0,255)
- `edge/transport/udp_sender.py` — `send_frame()`에 `header_redundancy: int = 1` 인자 추가, FRAME_HEADER N회 전송 for 루프
- `scripts/run_server_stitch.py` — recovery + reorder wire-up, CLI 인자, stats 출력
- `scripts/run_edge_video.py` — `--header-redundancy` CLI 인자

#### CLI 추가 인자
- 수신측: `--recovery`, `--recovery-iou` (default 0.3), `--recovery-max-age` (default 10), `--reorder`, `--reorder-size` (default 5), `--reorder-wait-ms` (default 500)
- 송신측: `--header-redundancy` (default 1 = no redundancy)

#### 검증 결과 (drop 30%, sim_seed=42, redundancy=3, fps=12)

| Test | 조건 | mp4 frames | 핵심 결과 |
|------|------|------|------|
| A | baseline (default) | 100 | 회귀 통과 |
| E | recovery off, redund=1 | 58 | Day 1 Test B와 동일 |
| F | recovery on, redund=1 | 58 | recovered=20, tracks=7 (ID switch 발생) |
| G | reorder only, redund=1 | 58 | ooo_at_emit=4 |
| H | recovery+reorder, redund=1 | 58 | **tracks=1** (reorder가 ID 일관성 회복) |
| J | drop 0, redund=3 | 100 | dedup 검증 (수정 전 166 → 수정 후 100) |
| **I** | **drop 30, recovery+reorder+redund=3** | **97** | **recovered=24, tracks=1** |
| K | I와 동일, recovery off | 97 | 모든 stats Test I와 동일, [recov] 없음, ZoH 시각 효과 isolation |

#### 핵심 성과 (Test I)
- `mp4 frames written: 97/100` — FRAME_HEADER cascade 해소 (이론값 0.3³ = 0.027 → 97.3 frame 살아남음)
- 시각적으로 부드러운 사람 인식 확인
- ZoH가 24개 손실 patch를 빨간 테두리로 시각화하며 복구

#### 발견된 부채 → 보고서 처리
- **UDP out-of-order**: UDPReceiver는 frame_id 순서 보장 없이 emit. Reorder buffer (5 frame, 500ms timeout)로 완화. 4 frame은 여전히 ooo (size 늘리면 줄지만 latency 증가).
- **FRAME_HEADER cascade**: 단일 packet 손실이 frame 전체 손실로 cascade. redundancy=3로 0.3³ = 0.027 로 감소. 추가 packet 비용은 100 frame당 200 packet (32B × 200 = 6.4KB, 무관).
- **ZoH의 본질적 한계**: 빠른 객체 이동에서 misalignment. Day 5 평가 단계에서 옵션 1 (motion prediction) 적용 여부 결정.
- **빈 frame redundant emit 버그 (수정됨)**: `expected_patches=0`인 frame이 redundancy 횟수만큼 emit되던 문제. `_emitted_frame_ids` set으로 dedup하여 해결.

#### Day 5 데모/평가 권고
- 데모 영상: drop 15%로 시연 (30%는 cascade 영향 줄어도 여전히 frame loss 일부)
- 평가 데이터: drop {0%, 15%, 25%, 30%} × recovery {on, off} × redundancy {1, 3}
- 인코딩: `--fps 12` (송신 fps와 일치)
- RQ4 시연: ZoH 복구 + 빨간 테두리 시각화 + Day 3 PID adaptive 통합

### Week 2 Day 3 완료 (May 25) — Decision Logic + PI Adaptive Controller

#### 신규 파일
- `server/decision/state_machine.py` — Confidence-weighted 3-state machine (Normal/Warning/Emergency) with sliding window + hysteresis
- `server/decision/__init__.py`
- `server/control/adaptive.py` — PI feedback controller with anti-windup (kp=30, ki=5, target_ratio=0.85)
- `server/control/__init__.py`

#### 수정 파일
- `scripts/run_server_stitch.py`
  - `_process_frame` helper로 wire-up 통일 (3군데 호출 경로 — main poll, rx.flush, reorder.flush)
  - CLI: `--decision`, `--decision-window` (10), `--decision-warn` (0.80), `--decision-emerg` (0.50), `--adaptive`, `--adaptive-target` (0.85), `--adaptive-initial-q` (75)
  - State transition log, [state] [ctrl] final stats

#### RQ4 통합 (옵션 2 + 4)
- 옵션 2: Confidence-weighted ratio (검출 confidence로 가중). RQ4의 thresholding/filtering 카테고리 강화.
- 옵션 4: PI feedback control (단순 lookup table 대신). Anti-windup으로 saturation 방지.

#### 검증 결과 (sim_seed=42, redundancy=3, fps=12)

| Test | Drop | mp4/100 | Normal% | Warning% | Emergency% | q_range | q_avg |
|------|------|------|------|------|------|------|------|
| A | 0% | 100 | — | — | — | — | — |
| L | 0% (decision+adaptive) | 100 | 100% | 0 | 0 | [80,95] | 94.5 |
| M | 15% | 100 | 85% | 15% | 0 | [80,95] | 94.5 |
| N | 30% | 97 | 44% | 49% | **7%** | **[30,95]** | **52.3** |
| O | 50% | 89 | 52% | 35% | 13% | [30,95] | 51.9 |

#### 핵심 성과
- Drop 0~50%에서 monotonic state progression
- PI controller가 drop 30%에서 quality 30까지 dynamic 적응
- transitions 6~10회 (hysteresis로 떨림 방지)
- Test N: 데모 영상 후보 (recovery 24 + adaptive quality + 3-state 시연)

#### 발견된 부채 → fix됨
- 4 frame 미스매치 ([state] total < frames_stitched): rx.flush + reorder.flush 두 경로가 `_process_frame` 헬퍼 안 거치고 직접 `_handle_frame` 호출. 두 군데 패치로 해결.

#### Day 4 준비 사항
- Adaptive controller가 quality 계산만 함. 엣지에 전달은 Day 4 (피드백 채널 또는 open-loop fallback).
- Streamlit 셸 시작 (Day 4 오후).

### Week 2 Day 4 완료 (May 25 늦은 밤 ~ 26 새벽) — Closed-Loop + Kalman + 영상 발굴

#### 신규 파일
- `common/packet.py` (수정): FEEDBACK packet format (33B = 32B header + 1B payload)
- `server/control/feedback_sender.py`: UDP sender for server→edge quality feedback
- `edge/control/feedback_receiver.py`: Non-blocking UDP receiver, idempotent dedup
- `edge/control/__init__.py`
- `server/recovery/kalman.py`: 4-state constant-velocity Kalman filter

#### 수정 파일
- `server/recovery/tracker.py`: Track에 KalmanState 필드, predict/correct, kf_predicted_bbox 매칭에 사용
- `server/recovery/zoh.py`: Kalman predicted position으로 expanded_bbox 평행이동
- `scripts/run_server_stitch.py`: `--feedback`, `--feedback-edge-host`, `--feedback-port`, `--feedback-redundancy`, `--adaptive-min-q`, `--adaptive-max-q` 추가
- `scripts/run_edge_video.py`: `--enable-feedback`, `--feedback-bind`, `--feedback-port` 추가. 카메라 입력 지원 (`--source 0` 또는 `/dev/video0`)

#### 핵심 발견
- 폐쇄루프 동작 입증 (Test P: sends=25, packets=50, changes=19, stale=6)
- Kalman 효과 person-bicycle-car-detection.mp4에서 명확 (사람이 슬라이드, 순간이동 없음)
- 다중 사람 영상 (new_vid.mp4, 16명) 21개 트랙 생성 (5 over-counting = ID switch)
- Detection 한계: 작은 객체 깜빡임은 트래커 튜닝으로 해결 불가
- **Demo configuration 발견**: 정적 quality (q=50) + 폐쇄루프 OFF가 시각 일관성 우월. 폐쇄루프는 정량 입증용으로 분리.

#### Demo 후보 영상
- `person-bicycle-car-detection.mp4`: 단일 사람, Kalman 효과 명확 (핵심 시연)
- `new_vid_2x.mp4`: 다중 사람 16명, ID switch 시연 (다중 객체 시연)
- 라이브 카메라 (Logitech C270, 640x480): 실시간 시연

#### 트레이드오프 분석
- Closed-loop PID는 정량적으로 동작이지만 quality 변동이 시각 출렁임 만듦
- Static q=50 + recovery + redundancy 5 = demo 시 가장 부드러움
- Future work: PID output smoothing (EWMA), quality change rate limiting

#### TODO (Day 5 = 내일)
- [ ] Streamlit dashboard (1~2 patterns 우선)
- [ ] Demo 영상 8 scene 측정 + 결합
- [ ] 영상 자막 추가
- [ ] 제출

### 압축 일정 (May 24 → 29, 6일)
- Day 1 (5/24 토): constraint simulator ✅
- Day 2 (5/24 늦은 오후~밤): IoU tracker + ZoH recovery + reorder buffer + FRAME_HEADER redundancy + dedup ✅
- Day 3 (5/25 일): Decision logic + Adaptive controller (옵션 2 confidence-weighted + 옵션 4 PID 통합)
- Day 4 (5/26 월): 피드백 채널 시도 → 안 되면 open-loop 폴백 → Streamlit 셸
- Day 5 (5/27 화): Streamlit 4패널 + 평가 실험
- Day 6 (5/28 수): 평가 마무리 + 보고서 + 데모 영상 + 제출

### Scope cuts (확정)
- ❌ Noise 시뮬레이터
- ❌ Option B (서버 동적 배치, ResNet18)
- ❌ Adjacent-frame interpolation (Zero-order hold만)
- ❌ MOT17 데이터셋
- ⚠️ 폐쇄루프 피드백 — Day 4 정오까지 안 잡히면 open-loop 폴백

---

## 핵심 코드 ground truth

새 채팅이 추측하지 않게 정확한 시그니처를 박아둠. 변경 시 이 섹션도 업데이트.

### `common/packet.py` — wire format 상수

```python
MAGIC = b"RPEC"
PROTOCOL_VERSION = 1

PKT_TYPE_PATCH_CHUNK = 0
PKT_TYPE_HEARTBEAT = 1   # reserved
PKT_TYPE_FEEDBACK = 2    # reserved (server → edge, Week 2 Day 6)
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

### `common/config.py` — 전체 상수

```python
PERSON_CLASS_ID = 0          # COCO class id for "person"
MIN_PATCH_SIZE = 32          # skip patches smaller than this (pixels, either dim)
BBOX_MARGIN = 0.10           # expand bbox by this fraction on each side before crop
DEFAULT_JPEG_QUALITY = 75    # JPEG quality (1-100)
MAX_PATCH_BYTES = 64 * 1024  # warn threshold for oversized patches
```

### `edge/detector/yolov8_trt.py`

```python
@dataclass
class Detection:
    x1: int; y1: int; x2: int; y2: int
    confidence: float
    class_id: int

class YOLOv8TRT:
    def __init__(
        self,
        engine_path: str | Path,
        input_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
    ) -> None: ...
    def detect(self, image_bgr: np.ndarray, person_only: bool = True) -> List[Detection]: ...
```

### `edge/patch/extractor.py`

```python
@dataclass
class Patch:
    frame_id: int
    det_id: int
    image: np.ndarray                             # cropped BGR uint8, numpy view (not copy)
    original_bbox: tuple[int, int, int, int]     # detector output, pre-margin
    expanded_bbox: tuple[int, int, int, int]     # after margin + frame clip
    conf: float
    # property: shape -> tuple[int, int]  (height, width)

def extract_patches(
    frame: np.ndarray,
    detections: Sequence[Detection],
    frame_id: int,
    margin: float = BBOX_MARGIN,
    min_size: int = MIN_PATCH_SIZE,
    person_class_id: int = PERSON_CLASS_ID,
) -> List[Patch]
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
    # property: size_bytes -> int

class PatchJPEGEncoder:
    def __init__(self, default_quality: int = DEFAULT_JPEG_QUALITY) -> None: ...
    def set_quality(self, quality: int) -> None: ...   # adaptive quality seam
    def encode(self, patch: Patch, quality: int | None = None) -> EncodedPatch: ...
    def encode_many(self, patches: Sequence[Patch], quality: int | None = None) -> List[EncodedPatch]: ...
    def reset_stats(self) -> None: ...
    # properties: quality, avg_bytes
    # stats attrs: encoded_count, total_bytes_out, oversized_count
```

### `edge/transport/udp_sender.py`

```python
@dataclass
class SendStats:
    packets_sent: int = 0
    bytes_sent: int = 0
    patches_sent: int = 0
    frames_sent: int = 0
    send_errors: int = 0

class UDPSender:
    def __init__(
        self,
        server_host: str,
        server_port: int,
        send_buffer_bytes: int = 1 << 20,
        per_chunk_sleep_us: float = 0.0,
    ) -> None: ...
    def send_frame(
        self,
        *,
        frame_id: int,
        encoded: Sequence[EncodedPatch],
        frame_w: int,
        frame_h: int,
    ) -> int: ...   # returns total UDP packets sent (1 header + N chunks)
    def send_patch(self, enc: EncodedPatch) -> int: ...   # returns chunks sent
    def send_frame_header(self, *, frame_id, n_patches, frame_w, frame_h) -> bool: ...
    def close(self) -> None: ...
    # context manager: __enter__ / __exit__
    # stats attr: stats: SendStats
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
    # property: loss_ratio -> float

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
    orphan_patches: int = 0
    recv_errors: int = 0      # non-blocking OSError on recvfrom

class UDPReceiver:
    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        recv_buffer_bytes: int = 1 << 22,
        patch_ttl_s: float = 0.200,
        frame_ttl_s: float = 0.500,
        packet_filter: Callable[[bytes], bytes | None] | None = None,
        # packet_filter: (buf) -> buf (pass-through) | None (drop)
        # seam for Week 2 constraint simulator
    ) -> None: ...
    def poll(self, timeout_s: float = 0.01) -> Iterator[object]: ...   # yields ReceivedPatch | ReceivedFrame
    def flush(self) -> List[object]: ...
    def close(self) -> None: ...
    # context manager: __enter__ / __exit__
    # stats attr: stats: ReceiveStats
```

### `server/stitcher/naive.py`

```python
@dataclass
class StitchResult:
    frame_id: int
    image: np.ndarray         # BGR (H, W, 3) uint8
    n_pasted: int             # successfully decoded + pasted patches
    n_skipped_decode: int     # bytes present but cv2.imdecode failed
    n_skipped_incomplete: int # patches marked incomplete (still attempted)

def stitch_frame(
    frame: ReceivedFrame,
    *,
    bg_value: int = DEFAULT_BG_VALUE,       # 128 (mid-grey)
    draw_bbox: bool = False,
    fallback_size: tuple[int, int] = DEFAULT_FALLBACK_SIZE,   # (720, 1280)
) -> StitchResult
```

`_paste` 동작: `expanded_bbox`가 있으면 그것을 사용, None이면 `original_bbox` fallback (chunk 0 손실 케이스). 두 경우 모두 patch_img를 bbox 크기로 resize해서 paste.

### `server/transport/constraint_sim.py` (Week 2 Day 1 추가)

```python
@dataclass
class FilterStats:
    seen: int = 0
    dropped: int = 0
    delayed: int = 0
    total_delay_ms: float = 0.0
    # properties: drop_rate, avg_delay_ms

class RandomDropFilter:
    def __init__(self, p: float, seed: Optional[int] = None) -> None: ...
    def __call__(self, buf: bytes) -> Optional[bytes]: ...  # None = drop
    # attrs: p, stats: FilterStats

class DelayJitterFilter:
    def __init__(self, min_ms: float, max_ms: float, seed: Optional[int] = None) -> None: ...
    def __call__(self, buf: bytes) -> bytes: ...  # never None
    # attrs: min_ms, max_ms, stats: FilterStats

class ChainedFilter:
    def __init__(self, *filters: Callable[[bytes], Optional[bytes]]) -> None: ...
    def __call__(self, buf: bytes) -> Optional[bytes]: ...
    # attrs: filters (tuple)

def build_filter(
    *,
    drop_prob: float = 0.0,
    delay_min_ms: float = 0.0,
    delay_max_ms: float = 0.0,
    seed: Optional[int] = None,
) -> Optional[Callable[[bytes], Optional[bytes]]]: ...
```

CLI 인자 (`run_server_stitch.py`): `--drop-prob`, `--delay-min-ms`, `--delay-max-ms`, `--sim-seed`.
[final] 출력 후 `[sim]` prefix로 filter stats 출력 (filter 비활성 시 무출력).

---

## 파일 구조

```
roi-privacy-edge/
├── common/
│   ├── packet.py             # wire format, struct pack/unpack, build_packets, build_frame_header_packet
│   └── config.py             # PERSON_CLASS_ID, BBOX_MARGIN, MIN_PATCH_SIZE, DEFAULT_JPEG_QUALITY, MAX_PATCH_BYTES
├── edge/
│   ├── detector/
│   │   └── yolov8_trt.py     # YOLOv8TRT class, Detection dataclass
│   ├── patch/
│   │   ├── __init__.py       # exports Patch, extract_patches, EncodedPatch, PatchJPEGEncoder, encode_patch
│   │   ├── extractor.py      # extract_patches(), Patch dataclass
│   │   └── jpeg_encoder.py   # PatchJPEGEncoder, EncodedPatch dataclass, encode_patch
│   └── transport/
│       ├── __init__.py       # exports UDPSender, SendStats
│       └── udp_sender.py     # UDPSender class, send_frame method
├── server/
│   ├── transport/
│   │   ├── __init__.py       # exports UDPReceiver, ReceivedPatch, ReceivedFrame, ReceiveStats
│   │   └── udp_receiver.py   # UDPReceiver class, reassembly, TTL sweeps, packet_filter seam
│   └── stitcher/
│       ├── __init__.py       # exports stitch_frame, StitchResult, DEFAULT_BG_VALUE
│       └── naive.py          # stitch_frame(frame, *, bg_value, draw_bbox, fallback_size) -> StitchResult
├── scripts/
│   ├── run_edge_video.py          # edge sender (args: --server --port --source --engine --quality --max-frames --target-fps --sleep-us --log-every)
│   ├── run_server_stitch.py       # server receiver + stitcher (args: --bind --port --output --png-dir --fps --draw-bbox --idle-timeout --ttl-ms)
│   ├── test_patch_pipeline.py     # extractor + encoder smoke test + quality sweep CSV
│   ├── test_udp_loopback.py       # 127.0.0.1 e2e, 3 verify blocks (JPEG / FRAME_HEADER / bbox roundtrip)
│   └── legacy/                    # Day 5 prototype scripts, not maintained
│       ├── README.md              # "deprecated" 경고 + D1 버그 고지
│       ├── run_edge_send.py       # single-shot edge sender (TypeError at send_frame call — do not use)
│       ├── run_server_recv.py     # single-shot server receiver
│       └── test_inference.py      # YOLOv8TRT standalone smoke test
├── engines/
│   └── yolov8n_fp16.engine        # TRT engine, 8.8MB, built on Jetson
├── data/
│   ├── test_images/persons.jpg
│   └── videos/person-bicycle-car-detection.mp4   # 768x432 12fps
├── results/
│   ├── stitched.mp4                              # latest e2e output (overwritten each run)
│   └── wk2_day0_expanded_bbox_baseline.mp4       # baseline (loss=0, normal)
└── REVIEW.md                                     # Week 1 코드 리뷰 (Phase 1-3)
```

---

## 환경 / 컨벤션

- Python 3.10+ (Jetson은 3.10, 노트북도 비슷)
- 들여쓰기: **스페이스 4칸**, 탭 금지
- 네이밍: `snake_case` (함수/변수), `PascalCase` (클래스)
- Type hints: `tuple[int, int, int, int]` (소문자, Python 3.9+ 스타일), `Optional` import 사용 OK
- Import 순서: stdlib → third-party → local. (`collections.abc` 등 stdlib은 반드시 local import 앞에)
- 실행: `cd ~/roi-privacy-edge && PYTHONPATH=. python -u scripts/...`
- Jetson에는 TensorRT + cuda Python bindings 있음, 노트북엔 둘 다 없음 → detector 호출은 Jetson에서만 가능

---

## 알려진 부채 / 추후 정리할 것들

1. **SIGINT 처리 (run_server_stitch.py)** — 가끔 [final] 줄 안 찍힘. `signal.signal` + `_running` flag + `flush()` 다 있어서 명확한 재현 케이스 없으면 미루는 중. constraint simulator 돌리다가 재현되면 그때 잡기.
2. ~~expanded_bbox wire 누락~~ — Week 2 Day 0에 해결 완료.
3. **quality sweep CSV plot 미생성** — `results/patch_sizes.csv`는 생성됨. matplotlib으로 bytes-vs-quality 커브 1장 추가 필요. Week 3 평가 단계에서 일괄.
4. **MOT17 다운 실패** (motchallenge.net 차단) — person-bicycle-car-detection.mp4로 충분히 굴러가는 중. Week 2 끝나고 데이터셋 다양성 필요해지면 다시.
5. **VideoWriterLazy 자동 백업 없음** — `results/stitched.mp4`가 매번 덮어써짐. 의미 있는 이름으로 명시적 저장하거나 스크립트에 백업 로직 추가 고려.
6. **`test_patch_pipeline.py:60` 로그 오타** — `f"bbox=({d.x1},{d.y2},{d.x2},{d.y2})"` 에서 두 번째 좌표가 `d.y1` 이어야 함. 로그만 틀리고 기능 무영향.
7. **`cv2.imwrite` 반환값 미확인 (`run_server_stitch.py:211`)** — `--png-dir` 사용 시 디스크 풀·경로 오류가 조용히 실패. `ok = cv2.imwrite(...); if not ok: warn` 형태로 수정 권장.
8. **TRT/CUDA 오류에 `assert` 사용 (`yolov8_trt.py`)** — 7개. `python -O` 시 비활성화. `raise RuntimeError(...)` 교체 권장. Jetson 직접 실행 시 `-O` 안 쓰므로 당장 급하진 않음.
9. **delay만으로는 손실 발생 안 함** — TTL(patch=200ms, frame=500ms) 안쪽 delay는 무손실. 평가에서 delay 효과 보려면 100~200ms 범위 필요. Day 5 평가 단계에서 결정.
10. **FRAME_HEADER 손실의 cascade 효과** — FRAME_HEADER가 drop되면 그 프레임 전체가 mp4에서 사라짐 (패치 도착했어도). 평가 시 frame loss 과대평가 가능. 해결책: (a) FRAME_HEADER duplicate 전송 또는 (b) 첫 PATCH_CHUNK에 frame meta prefix 추가. Day 5 평가 시 영향 보고 결정.
11. **`patches: List` vs `Dict` 혼란**: `_FrameAssembly.patches`는 Dict[int, ReceivedPatch] (내부용), `ReceivedFrame.patches`는 List[ReceivedPatch] (외부 노출, det_id 정렬). 헷갈리기 쉬움. Recovery layer 작성 시 처음에 Dict로 가정해 AttributeError 발생.
12. **`expected_patches=0` 빈 frame의 redundancy emit 버그 (해결됨)**: redundancy>1일 때 같은 frame_id가 redundancy 횟수만큼 emit. `_emitted_frame_ids: set` dedup으로 해결.
13. **RQ4 정렬**: 본 시스템은 결정론적 알고리즘 (filtering, thresholding) 사용. 프로젝트 조건의 "AI does not necessarily mean large or complex models" 명시. Day 3에 confidence-weighted decision + PID adaptive로 RQ4 카테고리 강화 예정.

---

## Week 2 인터페이스 설계 메모

### constraint simulator (Day 1-2)
`UDPReceiver.__init__`의 `packet_filter` 파라미터 사용:
```python
def drop_10pct(buf: bytes) -> bytes | None:
    return None if random.random() < 0.10 else buf

rx = UDPReceiver(host, port, packet_filter=drop_10pct)
```
`packet_filter`가 `None`을 반환하면 해당 패킷 drop, bytes를 반환하면 변형된 패킷으로 처리.

### recovery layer (Day 3-4)
`run_server_stitch.py:_handle_frame()`에 1줄 삽입:
```python
def _handle_frame(rf, writer, png_dir, draw_bbox):
    rf = recovery.enhance(rf)      # 여기 추가
    res = stitch_frame(rf, draw_bbox=draw_bbox)
```
`RecoveryLayer.enhance(rf: ReceivedFrame) -> ReceivedFrame` 인터페이스. `rf.patches`가 mutable list이므로 incomplete patch를 이전 frame 데이터로 교체 가능.

### adaptive quality (Day 5)
`PatchJPEGEncoder.set_quality(new_q)` seam이 준비됨. feedback transport(server → edge recv socket) 신설 필요.

### closed-loop feedback (Day 6)
`PKT_TYPE_FEEDBACK = 2` 상수 예약됨. payload format 미정의 → `packet.py`에 `FEEDBACK_PAYLOAD_FORMAT` 추가 필요.

---

## 새 채팅 시작 시 권장 패턴

1. 이 HANDOFF.md 통째로 첫 메시지에 붙여넣기
2. 그 다음 줄에 작업 요청 (예: "Week 2 Day 1 시작하자. constraint simulator 들어가자.")
3. 작업 중 파일 수정이 필요하면 **해당 파일의 관련 부분을 직접 보여주기** — Claude가 추측하면 변수명 틀리고 시간 낭비됨. grep + sed로 빠르게 발췌:
   ```bash
   grep -n "class\|@dataclass\|def " path/to/file.py
   sed -n '40,60p' path/to/file.py
   ```
4. 변경 후엔 항상 syntax check + 가능하면 단위 테스트로 빠른 검증

---

## TODO before next chat
- [x] git commit + push (Day 0 + Day 1 변경사항 일괄)
- [x] 노트북에 git pull
- [x] Week 2 Day 1: constraint simulator
- [x] Week 2 Day 2: recovery + reorder + redundancy + dedup
- [ ] Week 2 Day 3: Decision logic + Adaptive (옵션 2 + 4 통합)
- [ ] Test K (recovery off 비교 영상) — Day 2 마지막 검증
