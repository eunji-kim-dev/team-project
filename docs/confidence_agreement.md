# 카메라 앱 판정 기준 일치율 (2026-11-01 13:51)

성능표(`docs/bench_table.md`)의 일치율은 픽셀 argmax 클래스만 비교한 값이다. 카메라 앱(`camera/D435_Fusion_v3.5`)은 클래스만 쓰지 않고
softmax 신뢰도가 config 의 기준값 이상인 점만 채택하므로, 클래스가 같아도 신뢰도가 기준 근처에서 흔들리면 앱의 판정이 뒤집힌다.
이 문서는 앱이 실제로 쓰는 6개 판정 기준마다 INT8 로짓 엔진이 FP32 와 얼마나 같은 판정을 내리는지, 그 판정이 정답지 기준으로 얼마나 정확한지를 적는다.
도구: `python -m eval.confidence_agreement` (원본 `results/confidence_agreement_1101/`).

## 측정 조건

- 보드: NVIDIA Jetson Orin Nano Developer Kit Super, L4T R39.2.1, NV Power Mode 25W. 시간을 재는 측정이 아니라 클럭 고정은 하지 않았다
- 소프트웨어: TensorRT 10.16.2.10, onnxruntime 1.31.1, PyTorch 2.12.0+cu132, Python 3.12.3, numpy 1.26.4, OpenCV 4.8.0
- 코드: git `e24898f` + `eval/confidence_agreement.py` (이 문서와 같은 커밋). 앱 코드는 고치지 않고 import 만 했다:
  전처리 `d435map.semantics.prepare_input`, softmax 디코드 `decode_prediction`, 엔진 로더 `d435map.trt_fastscnn.TrtBackend`, PyTorch 경로 `d435map.torch_fastscnn.TorchBackend`.
  경계 1픽셀 제외(`boundary_erode_px` 1)는 `fusion.sample_semantics` 와 같은 규칙으로 적용
- 기준값: `camera/D435_Fusion_v3.5/config.json` fusion 블록. 바닥 0.70, 장애물·그 외 0.75, 깊이 장애물·SCNN 바닥 충돌 보류 0.85, 의미적 자유공간 0.90, 자유공간 연속성 0.75
- 입력: `data/splits/val.txt` 26장, 640×480 (총 798.7만 픽셀). 정답지 `data/labels_RGB`
- 모델 (표준 사양 `models/standard_fastscnn.json`, 모두 로짓 출력 [1,3,480,640])

  | 이름 | 파일 | 실행 | sha256 앞 16 |
  |---|---|---|---|
  | cpu_fp32 (기준) | `models/logits/fast_scnn_logits.onnx` | onnxruntime CPU FP32, 4 threads | e314afd29ea14bea |
  | gpu_fp32 | `models/logits/fast_scnn_logits_fp32.engine` (교체 전 앱에 들어 있던 엔진과 같은 파일) | TensorRT FP32 GPU | 4a36b1071afe4390 |
  | int8 | `models/calib_trt_adds/fast_scnn_qat_logits_int8.engine` (최종 QAT 엔진과 같은 가중치·scale, 로짓 출력) | TensorRT INT8(Q/DQ) GPU | cae54eb7c11a1aa8 |
  | torch_cpu | `checkpoints/best_mIoU_iter_1600.pth` 에서 state_dict 만 뽑은 파일 (앱이 PC 에서 쓰던 경로) | 앱 TorchBackend, PyTorch CPU FP32 | 9c67ad82cb9e1ef6 |

  앱의 TorchBackend 는 state_dict 만 든 .pth 만 받으므로 mmseg 체크포인트는 `deploy.export_onnx.CheckpointLoader` 로 state_dict 를 꺼내 저장한 뒤 썼다
- 지표 (쌍마다, 판정마다. 판정 마스크 = 클래스 일치 AND 신뢰도 ≥ 기준값 AND 경계 제외)
  - 전체 픽셀 일치%: 두 모델의 판정 마스크가 같은 픽셀 비율. 대부분 픽셀은 둘 다 "아니오"라 항상 높다
  - 판정 IoU: 두 판정 마스크의 교집합/합집합
  - 뒤집힘% = (잃음 + 얻음) / 기준 모델이 통과시킨 픽셀. 잃음 = 기준 통과·후보 탈락, 얻음 = 기준 탈락·후보 통과
  - 기준 ±0.02 안%: 기준 모델이 통과시킨 픽셀 중 신뢰도가 기준값 ±0.02 안에 있는 비율 (뒤집힐 수 있는 픽셀이 얼마나 있나)
  - Δ신뢰도 평균(기준 통과): 기준 모델이 통과시킨 픽셀에서 후보 − 기준 신뢰도의 평균. 음수면 후보가 덜 믿는다
  - 정답지 기준: 재현율 = 정답 c 픽셀 중 통과%, 정밀도 = 통과 픽셀 중 정답 c%, 위험 누출 = 바닥 판정은 정답 장애물 중 바닥으로 통과%(부딪힘), 장애물·그 외 판정은 정답 바닥 중 통과%(못 지나감)

## 결과 1. FP32 세 경로는 완전히 같다

gpu_fp32 vs cpu_fp32, torch_cpu vs cpu_fp32: argmax 일치 100.00%, |Δ신뢰도| 평균 0.0000 / 최대 0.000, 6개 판정 모두 뒤집힘 0.00%, 판정 IoU 1.0000.
앱을 .pth(PyTorch CPU) 에서 FP32 로짓 엔진으로 바꾼 것은 판정을 바꾸지 않았다. 따라서 아래 int8 표는 세 FP32 경로 어느 것과 비교해도 같다.

## 결과 2. int8 vs fp32

argmax 일치 99.41% (성능표의 int8 일치율과 같음), |Δ신뢰도| 평균 0.0062 / p99 0.120 / 최대 0.493, 부호 평균(int8 − fp32) −0.0021

| 판정 | 클래스 | 기준 | 전체 픽셀 일치% | 판정 IoU | 뒤집힘% | 잃음% | 얻음% | 기준 ±0.02 안% | Δ신뢰도 평균(기준 통과) | 기준 통과 픽셀(만) |
|---|---|---|---|---|---|---|---|---|---|---|
| 바닥 채택 | 바닥 | 0.70 | 99.56 | 0.9746 | 2.57 | 1.58 | 0.99 | 1.10 | −0.0074 | 137.6 |
| 장애물 채택 | 장애물 | 0.75 | 99.41 | 0.9915 | 0.86 | 0.48 | 0.38 | 0.45 | −0.0012 | 555.2 |
| 그 외 채택 | 그 외 | 0.75 | 99.88 | 0.9874 | 1.27 | 0.74 | 0.53 | 0.46 | −0.0046 | 74.3 |
| 깊이 장애물·SCNN 바닥 충돌 보류 | 바닥 | 0.85 | 99.54 | 0.9719 | 2.83 | 2.06 | 0.77 | 2.01 | −0.0062 | 130.1 |
| 의미적 자유공간 내부 | 바닥 | 0.90 | 99.52 | 0.9701 | 3.01 | 2.31 | 0.69 | 2.81 | −0.0054 | 126.4 |
| 자유공간 연속성 | 바닥 | 0.75 | 99.55 | 0.9739 | 2.63 | 1.73 | 0.90 | 1.35 | −0.0072 | 135.5 |

## 결과 3. 정답지 기준 (판정 마스크 자체의 정확도)

fp32 열은 cpu_fp32 = gpu_fp32 = torch_cpu (세 값이 같다).

| 판정 | 기준 | fp32 재현율% / 정밀도% / 위험 누출% | int8 재현율% / 정밀도% / 위험 누출% | int8 − fp32 (재현율 pp / 누출 pp) |
|---|---|---|---|---|
| 바닥 채택 (바닥) | 0.70 | 90.43 / 96.15 / 0.93 | 89.96 / 96.22 / 0.90 | −0.47 / −0.03 |
| 장애물 채택 (장애물) | 0.75 | 95.98 / 98.97 / 2.43 | 95.88 / 98.98 / 2.44 | −0.10 / +0.01 |
| 그 외 채택 (그 외) | 0.75 | 92.36 / 99.25 / 0.00 | 92.20 / 99.28 / 0.00 | −0.16 / 0.00 |
| 깊이 장애물·SCNN 바닥 충돌 보류 (바닥) | 0.85 | 86.71 / 97.49 / 0.57 | 85.64 / 97.55 / 0.55 | −1.07 / −0.02 |
| 의미적 자유공간 내부 (바닥) | 0.90 | 84.58 / 97.90 / 0.46 | 83.25 / 97.94 / 0.45 | −1.33 / −0.01 |
| 자유공간 연속성 (바닥) | 0.75 | 89.51 / 96.62 / 0.80 | 88.84 / 96.70 / 0.77 | −0.67 / −0.03 |

## 읽는 법

- INT8 로 바꾸면 앱이 신뢰하는 픽셀의 0.9~3.0% 가 판정이 바뀐다 (장애물 0.86%, 바닥 계열 2.6~3.0%). argmax 일치 99.41% 보다 뒤집힘이 큰 이유는
  신뢰도가 기준 근처에 있는 픽셀이 기준값이 높을수록 많기 때문이다: 기준 ±0.02 안 비율이 0.70 → 1.10%, 0.85 → 2.01%, 0.90 → 2.81% 로 늘고 뒤집힘도 같은 순서로 2.57 → 2.83 → 3.01% 로 는다.
  즉 특정 영역이 통째로 바뀌는 것이 아니라 기준값 근처에서 ±0.1 안쪽으로 흔들리는 픽셀(|Δ| p99 0.12)이 뒤집히는 것이다
- 방향은 보수적이다. 모든 판정에서 잃음 > 얻음이고 기준 통과 픽셀의 Δ신뢰도 평균이 −0.001~−0.007 이다. INT8 은 FP32 보다 바닥을 조금 덜 믿는다
- 정답지 기준으로는 위험 누출(장애물을 바닥으로 믿는 비율)이 늘지 않았다 (0.93 → 0.90, 0.57 → 0.55, 0.46 → 0.45, 0.80 → 0.77). 대신 바닥 재현율이 0.5~1.3 pp 줄었다 (자유공간이 그만큼 덜 잡힘).
  장애물 판정은 재현율 95.98 → 95.88, 누출 2.43 → 2.44 로 같다. 26장 기준 0.0x pp 차이는 노이즈다
- 결론: 기준값을 그대로 두고 INT8 로짓 엔진으로 교체해도 앱 판정의 정답지 기준 정확도는 유지된다. 바뀌는 것은 "바닥을 1~3% 덜 믿는다"는 보수적 방향이고, 기준값을 조정할 근거는 없다
- 사진별 (int8 vs cpu_fp32, 바닥 채택 뒤집힘%): 최대 0106 7.65%, 0073 6.41%, 0107 5.30%, 0077 5.08%; 최소 0038 0.59%. 바닥 픽셀이 적은 사진(0106 은 2.6만)에서 비율이 크다. `results/confidence_agreement_1101/per_image.csv`
- 한계: 픽셀 단위 비교다. 앱은 깊이 점에서만 샘플링하고 평면 검증·격자 집계를 거치므로 지도 셀 단위의 차이는 이보다 작을 가능성이 크다 (추측, 아직 안 잼). 실제 카메라 프레임으로 두 엔진의 코스트맵을 비교하려면 저장된 캡처가 필요하다

## 앱 엔진 교체 (2026-11-01)

이 결과를 근거로 `camera/D435_Fusion_v3.5/models/` 의 FP32 로짓 엔진을 빼고 `fast_scnn_qat_logits_int8.engine` (+ `.engine.json`, sha cae54eb7c11a1aa8) 을 넣었다.
기준값(config.json fusion)은 바꾸지 않았다. 앱 로더가 READY 로 올라오고 predict() 가 유효한 라벨·신뢰도를 내는 것까지 확인했다 (GUI 는 ssh -X 터미널에서 따로 확인).
FP32 로 되돌리려면 INT8 두 파일을 빼고 `models/logits/fast_scnn_logits_fp32.engine` 과 그 `.json` 을 복사하면 된다 (폴더에 모델 파일은 정확히 하나).

## GPU 디코드 엔진 (2026-11-01 14:55)

앱의 CPU softmax 디코드(프레임당 약 15 ms)를 엔진 그래프로 옮긴 `models/calib_trt_adds/fast_scnn_qat_decode_int8.engine` (`deploy.add_decode` 로 로짓 ONNX 에
Softmax→ReduceMax(confidence), ArgMax→Cast(labels) 를 붙여 빌드, sha a2bcc67acc2de136) 을 같은 도구로 확인했다 (`results/confidence_agreement_1101_decode/`).
int8_decode vs int8(로짓 엔진 + CPU 디코드): argmax 일치 100.00%, |Δ신뢰도| 최대 0.000, 6개 판정 모두 뒤집힘 0 픽셀. Softmax·Reduce 층은 FP32 로 실행된다.
같은 방법으로 만든 PTQ 디코드 엔진 `models/logits/fast_scnn_decode_int8.engine` (sha 는 매니페스트, 엔트로피 102장): fp32 대비 99.41% / |Δ신뢰도| 평균 0.0062, QAT 그래프 디코드 엔진 대비 99.94% / 평균 0.0007 (`results/confidence_agreement_1101_ptq_b/`).
2026-11-01 15:50 부터 앱에는 PTQ 디코드 엔진이 들어가 있다 (`docs/app_pipeline.md`).

## 실행

```bash
# state_dict 만 든 .pth (앱 TorchBackend 용)가 필요하면:
python -c "import torch; from pathlib import Path; from deploy.export_onnx import CheckpointLoader; sd,_=CheckpointLoader.load(Path('checkpoints/best_mIoU_iter_1600.pth')); torch.save(dict(sd),'/tmp/fastscnn_state_dict.pth')"
python -m eval.confidence_agreement \
    --models cpu_fp32=models/logits/fast_scnn_logits.onnx gpu_fp32=models/logits/fast_scnn_logits_fp32.engine \
             int8=models/calib_trt_adds/fast_scnn_qat_logits_int8.engine torch_cpu=/tmp/fastscnn_state_dict.pth \
    --pairs int8:gpu_fp32 --images data/images --gt data/labels_RGB --list data/splits/val.txt \
    --out results/confidence_agreement_1101
```
