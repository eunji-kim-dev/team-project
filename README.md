# team-project

Fast-SCNN 3클래스(바닥 / 장애물 / 그 외) 분할 모델을 Jetson Orin Nano 에서 TensorRT 로 돌리고,
RealSense D435 의 RGB-D 와 융합해 로컬 코스트맵을 만드는 프로젝트.

## 폴더 구조

```
team-project/
├─ deploy/      변환·엔진·추론 모듈 (python -m deploy.<이름> 으로 실행)
│   ├─ export_onnx.py     .pth → ONNX (--keep-logits: argmax 없이 로짓 출력)
│   ├─ build_engine.py    ONNX → TensorRT 엔진 (fp32 / fp16 / int8 PTQ / qat)
│   ├─ segmenter.py       추론 모듈. .onnx → onnxruntime, .engine → TensorRT 자동 선택
│   ├─ qat_train.py       QAT INT8 학습 → Q/DQ ONNX   (quant.py, trt_scales.py 가 보조)
│   ├─ inspect_engine.py  엔진 레이어 정보 추출
│   ├─ profile_engine.py  엔진 층별 실행 시간·정밀도 프로파일 (엔진 간 속도 차이 원인 분석)
│   ├─ qat_sensitivity.py 블록별 양자화 진단
│   ├─ preprocess.py      입력 전처리 공통 (리사이즈·채널 순서·정규화)
│   ├─ cuda_util.py       CUDA 메모리·스트림 얇은 래퍼
│   └─ fastscnn_mmseg.py  mmseg 없이 쓰는 Fast-SCNN 정의
├─ eval/        측정 도구 (python -m eval.<이름>)
│   ├─ bench_engines.py      속도·mIoU·메모리 → docs/bench_table.md
│   ├─ power_sweep.py        모델별 전력·온도 → docs/power_table.md
│   ├─ soak_test.py          장시간 안정성
│   ├─ quant_sensitivity.py  INT8 블록별 민감도 → docs/quant_sensitivity.md
│   ├─ parse_tegrastats.py / tegrastats_log.sh   전력 로그 기록·요약
├─ models/      모델 매니페스트(.json)만 올린다. .onnx/.engine 은 각자 보드에서 생성
│   ├─ calib_*/   캘리브레이션 실험별 매니페스트
│   └─ logits/    카메라 앱용 로짓 출력 모델 (fast_scnn_logits.onnx, *_fp32.engine)
├─ camera/
│   ├─ D435_Fusion_v3.5/   카메라 앱 (RGB-D + SCNN 융합 코스트맵). 실행: v3.5.py
│   └─ legacy/             이전 버전 카메라 테스트 스크립트 (참고용)
├─ docs/        환경 기록, 성능표, 전력표, 민감도 분석
├─ checkpoints/ 학습 체크포인트 (깃 미포함, 드라이브)
├─ data/        학습·검증 이미지와 라벨 (깃 미포함, 드라이브)
├─ results/     측정 원본 결과 (깃 미포함)
├─ requirements-jetson.txt   Jetson venv 패키지 목록 (pip freeze)
└─ .gitignore   저장소 전체에서 이 파일 하나만 쓴다
```

## 실행

### 변환 → 엔진 → 측정 (Jetson, 저장소 루트에서)

```bash
# 1. 체크포인트 → ONNX
python -m deploy.export_onnx --ckpt checkpoints/best_mIoU_iter_1600.pth \
    --model-py deploy/fastscnn_mmseg.py --out models/fast_scnn.onnx --check-image data/images/0001_color.png
# 2. ONNX → 엔진 (옆에 <이름>.engine.json 매니페스트가 함께 생긴다)
python -m deploy.build_engine --onnx models/fast_scnn.onnx --precision fp32 fp16
# 3. 성능표
python -m eval.bench_engines --models gpu_fp32=models/fast_scnn_fp32.engine ... --images data/images --gt data/labels_RGB --list data/splits/val.txt
```

각 스크립트의 자세한 옵션은 파일 맨 위 docstring 에 있다. 환경(CUDA/TensorRT/PyTorch 버전)은 `docs/jetson_environment.md`.

### 카메라 앱 (camera/D435_Fusion_v3.5)

```bash
cd camera/D435_Fusion_v3.5
python v3.5.py            # 또는 Windows: run_camera.cmd
```

- `models/` 에 모델 파일을 **정확히 하나** 두고 `standard_fastscnn.json` 과 함께 둔다.
  - `.pth` (state_dict 만 든 파일) → PyTorch CPU 추론
  - `.engine` + 같은 이름의 `.engine.json` → TensorRT GPU 추론 (Jetson).
    엔진은 **로짓 출력**이어야 한다 (`export_onnx --keep-logits` 로 만든 ONNX 에서 빌드).
    `models/` 의 벤치용 엔진은 argmax 가 그래프에 들어 있어 앱이 거부한다.
- Windows 는 `camera/D435_Fusion_v3.5/requirements.txt`, Jetson 은 venv 에 `pyrealsense2` 만 추가로 설치한다.
  Jetson 에서 Windows 용 requirements.txt 를 설치하면 TensorRT/onnxruntime 스택이 깨진다.
- Jetson 에서는 GUI 창이 필요하므로 `ssh -X` 터미널에서 실행하고 Q/Esc 로 종료한다 (창의 X 버튼은 GTK OpenCV 에서 비정상 종료).

## 깃 규칙

- 브랜치를 따서 작업하고 PR 로 main 에 합친다.
- 올리지 않는 것: `data/`, `checkpoints/`, `results/`, `*.onnx`, `*.engine`, `*.calib`, 카메라 `captures/`·`logs/`.
  매니페스트 `.json` 과 `docs/` 결과표는 올린다.
- `.gitignore` 는 루트 하나만 쓴다. 하위 폴더에 따로 만들지 않는다.
