# 카메라 앱 전체 파이프라인 측정과 병목 (2026-11-01)

엔진 성능표(`docs/bench_table.md`)는 모델 추론만 잰 것이다. 이 문서는 실제 카메라로 앱(`camera/D435_Fusion_v3.5`)을 돌려 프레임마다 단계별 시간·드롭·입력 나이·지연·전력을 재고,
어디서 시간이 가는지 함수 단위로 찍고, 출력을 바꾸지 않는 범위에서 고친 결과를 적는다. 도구는 `python v3.5.py --duration --frame-log`, `python -m eval.app_timing`, `python -m eval.app_profile`.

## 결론

- **엔진은 프레임 시간의 1%였다.** 교체 전 앱은 FP32 엔진이든 INT8 엔진이든 프레임당 522 ms 가 걸려 1.9 fps 로 돌았고 카메라 15 fps 중 87% 를 버렸다. INT8 교체로 아낀 건 4.5 ms.
  GPU 사용률은 2%. 시간은 CPU 의 깊이→3D·바닥(211 ms)과 코스트맵(268 ms)에 있었다.
- **같은 프레임을 오프라인으로 처리하면 271 ms 였다.** 라이브의 나머지 250 ms 는 UI 스레드(약 40 Hz 렌더)와 카메라 스레드(3 ms 마다 폴링)가 처리 스레드와 GIL 을 나눠 쓴 비용이었다.
- 세 라운드로 **처리 p50 522 → 284 ms, 처리 fps 1.87 → 3.47 (+86%), 입력 나이 267 → 116 ms, 끝까지 지연 555 → 321 ms.** 출력(점유 격자·비용·자유 근거·점 분류·바닥 평면)은 저장 캡처 20장 전부 전과 동일.
- 15 fps(66.7 ms) 에는 아직 멀다. 남은 시간은 격자 1.4만 셀 × 표본 5 × 높이 9 의 투영 검사와 22만 점에 대한 numpy 연산이 대부분이라, 출력을 유지하는 미세 최적화로는 여기까지다. 다음은 알고리즘 결정(아래 "15 fps 까지").

## 측정 조건

- 보드: Jetson Orin Nano Developer Kit Super, L4T R39.2.1, NV Power Mode 25W, `jetson_clocks` 고정 (실행마다 `--show` 로 GPU Min=Max=918 MHz 확인, 끝나면 복원. 로그에 기록)
- 카메라: RealSense D435 (S/N 920312070366, 펌웨어 5.17.3.10), USB 2.1, 640×480 RGB+Depth 15 fps. 카메라를 움직이지 않은 실내 장면 (프레임당 점 22.2만 개, 유효 깊이 72%)
- 앱 실행: 가상 디스플레이 `xvfb-run -a -s "-screen 0 1600x900x24" python v3.5.py --duration 180 --frame-log …` (GTK 창을 실제로 그린다). 실행마다 180초, 앞 10 프레임 제외, 마감 = 66.7 ms (15 fps 간격)
- 소프트웨어: Python 3.12.3, numpy 1.26.4, OpenCV 4.8.0 (cv2 스레드 2), TensorRT 10.16.2.10, pyrealsense2 2.59.1. 코드 git `01b58bf` + 이 문서와 같은 커밋의 변경
- 전력: `eval/tegrastats_log.sh` 1초 간격, 대기 기준선 30초 (6.21 W, RAM 3369 MB, tj 49.4 °C). 원본 `results/app_live_1101/<실행>/` (frame_log.csv, frame_log.json 조건, latest_runtime.json, tegrastats.log, power.json)
- 지표: 처리 ms = 작업 스레드의 process() 시간 (SCNN + 깊이→3D·바닥 + 코스트맵 + 마무리). 입력 나이 = 캡처 수신 → 처리 시작. 지연 = 수신 → 처리 완료. 드롭 = 처리 중 새 캡처가 와서 대기열에서 교체된 프레임

## 라이브 측정 (실제 카메라, 180초씩)

| 항목 | fp32 엔진 (교체 전) | int8 엔진 | 라운드 1 | 라운드 2 | 라운드 3 (GPU 디코드) |
|---|---|---|---|---|---|
| 시각 | 14:29 | 14:25 | 14:41 | 14:49 | 14:58 |
| 처리 fps | 1.86 | 1.87 | 2.80 | 3.30 | **3.47** |
| 수신 fps / 드롭% | 14.8 / 87.5 | 14.8 / 87.4 | 14.8 / 81.2 | 14.8 / 77.9 | 14.8 / **76.7** |
| 처리 ms p50 / p95 | 522.5 / 727.3 | 521.6 / 725.8 | 349.1 / 415.0 | 297.9 / 348.4 | **284.4 / 338.5** |
| SCNN 단계 (준비 / 모델 / 디코드) | 28.1 (4.9 / 6.8 / 15.1) | 23.6 (4.3 / 3.4 / 15.1) | 23.6 (4.3 / 3.4 / 15.2) | 22.9 (4.2 / 3.3 / 14.9) | 10.2 (4.2 / 3.8 / 2.0) |
| 깊이→3D·바닥 | 202.3 | 211.2 | 77.6 | 62.1 | 63.3 |
| 코스트맵 (그중 후보 검토) | 265.3 (76.7) | 268.4 (80.5) | 216.6 (60.7) | 187.6 (57.5) | 188.5 (55.6) |
| 마무리 | 15.7 | 15.9 | 13.9 | 13.6 | 13.6 |
| 입력 나이 ms p50 / p95 | 275.8 / 461.0 | 267.3 / 435.2 | 149.4 / 230.2 | 129.1 / 192.8 | 115.8 / 179.1 |
| 끝까지 지연 ms p50 / p95 | 559.8 / 775.1 | 555.2 / 755.5 | 388.9 / 457.3 | 334.8 / 394.6 | 320.6 / 386.0 |
| 마감(66.7 ms) 초과% | 100 | 100 | 100 | 100 | 100 |
| 보드 전력 W (대기 대비) | 8.77 (+2.56) | 8.70 (+2.49) | 8.40 (+2.19) | 8.39 (+2.18) | 8.43 (+2.22) |
| RAM 증가 MB / GPU 사용률 / 최고 tj | 536 / 2.1% / 54.4 °C | 469 / 1.9% / 53.9 | 446 / 1.6% / 54.0 | 422 / 2.4% / 53.8 | 254 / 2.0% / 54.1 |
| 바닥 검출·SCNN 성공 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |

라운드가 바꾼 것:

- **라운드 1** — 화면을 매 루프(약 40 Hz) 그리던 것을 "새 결과·새 캡처·상태 변화가 있을 때만, 최대 10 Hz" 로 (`view.live_render_max_hz`); 바닥 RANSAC 의 표본 외적을 한 번에 계산; 깊이 지지 검사를 후보 경계 상자 안에서만.
  → 처리 522 → 349 ms. 깊이→3D·바닥 단계가 211 → 78 ms 로 준 것이 거의 전부 GIL 경합 해소분이다 (오프라인에서는 67 → 57).
- **라운드 2** — 카메라 폴링 대기 3 → 10 ms (`camera.poll_wait_ms`), 렌더 상한 5 Hz; 지지 검사를 dense 슬라이스 연산으로; 자유공간 가시성·SCNN 자유공간 투영·수평 시야의 5개 표본 위치를 루프 대신 한 배열로; 의미 샘플링·색칠의 2차원 fancy index 를 평면 인덱스로.
  → 349 → 298 ms.
- **라운드 3** — softmax 신뢰도·argmax 를 엔진 그래프로 (`deploy.add_decode`, 매니페스트 `decode_in_graph`). CPU 디코드 15 ms → GPU. 라벨 100% 일치, 신뢰도 차이 0 (`docs/confidence_agreement.md`).
  → 298 → 284 ms, RAM 증가 422 → 254 MB. 앱 폴더의 엔진은 지금 이것 (`fast_scnn_qat_decode_int8.engine`)
- 라운드 3 에서 "화면 그리기" 열이 10.9 → 58.8 ms 로 늘었다. 이 값은 UI 스레드가 render() 에 걸린 벽시계 시간이라 GIL 대기를 포함한다. 5 Hz 렌더이므로 처리에는 영향이 없고, 디코드가 사라져 처리 스레드가 GIL 을 놓는 틈이 줄어 UI 쪽 대기가 늘어난 것으로 보인다 (추측, 따로 재지 않음)

## 오프라인 처리 시간 (같은 캡처 20장, `eval.app_profile`, 클럭 고정)

카메라·화면 스레드 없이 Processor.process 만 돌린 값. 라이브와의 차이가 스레드 경합 비용이다.

| | 처리 p50 ms | SCNN | 깊이→3D·바닥 | 코스트맵 (후보 검토) | 마무리 | 출력 비교 |
|---|---|---|---|---|---|---|
| 전 (01b58bf) | 271.0 | 21.7 | 66.8 | 169.9 (59.9) | 12.6 | 기준 |
| 라운드 1 | 258.6 | 21.7 | 57.1 | 167.0 (56.2) | 12.6 | 20/20 프레임 동일 |
| 라운드 2 | 234.5 | 21.7 | 53.5 | 146.6 (50.7) | 12.5 | 20/20 프레임 동일 |
| 라운드 3 (GPU 디코드 엔진) | 228.5 | 9.7 | 53.7 | 151.7 (55.0) | 12.5 | 20/20 프레임 동일 |

회차 간 p50 변동 1~2% (라운드 3 의 코스트맵 +5 ms 는 이 변동 안). 라이브 − 오프라인: 교체 전 522 − 271 = 251 ms → 라운드 2 뒤 298 − 234 = 64 ms, 라운드 3 뒤 284 − 229 = 56 ms. 남은 64 ms 는 카메라 스레드의 프레임 복사·정렬(rs.align, 프레임마다 약 4 ms)과 librealsense 자체 스레드, 5 Hz 렌더의 몫이다.

## 함수별 프로파일 (cProfile, 프레임당 ms, 라운드 2 후 기준 정렬)

| 함수 | 전 | 라운드 2 후 | 차이 |
|---|---|---|---|
| 후보 검토 전체 (candidate_review.process) | 65.4 | 52.1 | −13.3 |
| ⤷ 국소 바닥 모델 (local_floor_model, 프레임당 5회) | 27.9 | 27.1 | −0.8 |
| ⤷ 이동 판정 광학 흐름 (MotionGuard.check) | 11.4 | 11.3 | −0.1 |
| ⤷ 깊이 지지 검사 (supported_obstacles, 5~6회) | 30.8 | 8.3 | −22.5 |
| SCNN 자유공간 투영 (projected_floor_free) | 24.7 | 23.0 | −1.6 |
| 점 색칠·가림 (colorize) | 23.0 | 22.5 | −0.5 |
| 자유공간 가시성 (visible_free) | 24.6 | 20.9 | −3.6 |
| 광선 가림 (camera_occluded, 2회) | 15.1 | 15.0 | −0.1 |
| softmax 디코드 CPU (decode_prediction) | 14.5 | 14.3 | → 라운드 3 에서 GPU |
| 바닥 RANSAC (fit_floor) | 21.0 | 9.1 | −11.9 |
| 의미 샘플링 (sample_semantics) | 15.0 | 9.0 | −6.0 |
| 수평 시야 (horizontal_view) | 8.2 | 8.3 | +0.1 |
| 격자 인덱스 (LocalGrid.indices) | 7.7 | 7.6 | −0.1 |
| 점 분류 (classify_points) | 6.4 | 6.2 | −0.2 |
| 가시 깊이 (visibility_depth) | 3.6 | 3.6 | 0 |
| 깊이 필터 (filtered_depth) | 1.3 | 1.3 | 0 |

읽는 법: 큰 감소는 (1) 지지 검사를 후보 경계 상자 안의 dense 연산으로 바꾼 것, (2) RANSAC 70회 반복의 np.cross 호출 오버헤드를 없앤 것, (3) 2차원 fancy index 를 평면 인덱스로 바꾼 것이다.
오프셋 5개를 한 배열로 묶은 것(visible_free, projected_floor_free, horizontal_view)은 기대보다 작았다. 이 함수들은 파이썬 오버헤드가 아니라 배열 원소 수 자체(셀 × 5 × 높이 9)가 비용이다.

## 출력 동일성 검증

`python -m eval.app_profile --source results/app_profile_1101/captures --dump <after> --compare results/app_profile_1101/before` 가 프레임마다 점유 격자(physical), 비용(cost), 자유 근거 비트(free_evidence),
점 분류(point_classes), 바닥 평면(normal, offset, ok) 을 변경 전 결과와 비교한다. 라운드 1·2 모두 20/20 프레임에서 차이 0 셀. 바꾼 함수들은 전부 "같은 원소 연산을 다른 순서/묶음으로" 하는 것만 골랐고,
결과가 비트 단위로 달라질 수 있는 변경(lstsq 를 정규방정식으로, float32 좌표, 표본 수 축소 등)은 넣지 않았다.

## 코스트맵 격자 10 cm (셀 1/4) 를 재 본 결과

`grid.resolution_m` 0.05 → 0.10 (셀 14,400 → 3,600). 로봇 키(0.8 m)는 높이 표본에만 쓰여 셀 크기와 무관하고, 로봇 너비(0.4 m)에서 나오는 금지 반경 0.25 m·팽창 반경 0.40 m 는 미터 단위라 그대로 적용되며 경계만 ±5 cm 단위로 양자화된다. 설정 검증(6 m 범위가 해상도의 배수)도 통과.

| | 5 cm | 10 cm |
|---|---|---|
| 오프라인 처리 p50 ms (같은 20장, 클럭 고정) | 228.5 | 181.7 (−20%) |
| 코스트맵 단계 / 후보 검토 | 151.7 / 55.0 | 104.8 / 46.8 |
| 깊이→3D·바닥 (점 단계, 불변) | 53.7 | 54.0 |
| 지도 면적 m² (3장 평균): 자유 / 장애물 / 팽창 띠 / 미확인 | 3.14 / 3.52 / 1.05 / 29.35 | 3.25 / 5.15 / 1.31 / 27.60 |

장애물 면적이 45% 는다. 얇은 물체가 10 cm 셀을 통째로 채우기 때문이다. 안전 쪽으로 보수적이지만 좁은 통로는 양쪽에서 최대 10 cm 씩 잃는다. 이 로봇(너비 0.4, 금지 반경 0.25)은 5 cm 셀에서 약 60 cm 통로를 지나고 10 cm 셀에서는 약 70 cm 가 필요하다.
그림: `results/app_profile_1101/res_compare/res005_*_costmap.png`, `res010_*_costmap.png` (같은 복도 모양, 가장자리가 거칠고 장애물이 두꺼움). 원본 `results/app_profile_1101/res010/`.
결정 기준은 시연 동선의 가장 좁은 틈이다. 70 cm 이상이면 10 cm 로 줄여도 되고, 그보다 좁으면 5 cm 를 유지한다.

## 15 fps 까지 (팀 결정이 필요한 것)

지금 라이브 284 ms, 오프라인 바닥 약 225 ms. 66.7 ms 가 되려면 3~4배가 더 필요하고, 출력을 유지하는 numpy 미세 최적화로는 20~30 ms 가 한계로 보인다. 남은 선택지는 전부 출력이 바뀌므로 지도 품질과 함께 재고 결정해야 한다.

1. **점 수 줄이기.** 깊이를 2×2 로 데시메이션하면 22만 → 5.5만 점. 점 단위 단계(깊이→3D, 분류, 색칠, 격자 인덱스, 지지 검사, 후보 검토 일부)가 1/4 로 줄어 처리 약 60 ms 감소 추정 (실측 필요). 설정값으로 넣고 `--compare` 로 지도 차이를 보면 된다.
   (격자 10 cm 는 위 절에서 실측: −47 ms. 점 데시메이션과 합치면 약 −100 ms 추정.)
2. **격자 단계의 표본 줄이기.** visible_free·projected_floor_free·horizontal_view 는 셀마다 표본 위치 5 × 높이 9 를 투영한다. 표본을 줄이거나(예: 중앙+모서리 2곳), 셀 1.4만 개 중 시야 안 셀만 계산하도록 바꾸면 이 세 함수 약 50 ms 의 절반 이상이 줄 수 있다.
3. **후보 검토.** local_floor_model 27 ms 는 5개 후보마다 lstsq 4회 반복이다. 반복 횟수·표본 수를 줄이거나 후보 수 상한을 두면 줄지만 판정이 바뀔 수 있다. MotionGuard 광학 흐름 11 ms 는 해상도(320×240)·특징 수(240)로 조절 가능.
4. **카메라 스레드.** rs.align 을 프레임마다 하지 않고 작업 스레드가 처리 직전에 한 번만 하면 남은 경합 64 ms 의 상당 부분이 줄 것이다. 프레임 수명 관리(SDK frameset 보관)를 바꿔야 해서 구조 변경이다.
5. **격자 단계를 GPU/C++ 로.** 1~4 로도 15 fps 가 안 되면 남는 길. 22만 점 투영과 격자 검사는 CuPy/Torch 로 옮기기 쉬운 형태다.

추측을 포함한 추정치이므로, 각 항목은 적용 → `eval.app_profile`(속도, 출력 차이) → 라이브 180초(fps, 드롭, 지연) 순으로 재고 채택을 결정하는 것이 맞다.

## 앱 엔진 최종: PTQ INT8 디코드 엔진 (2026-11-01 15:50)

사용자 결정으로 앱 엔진을 `models/logits/fast_scnn_decode_int8.engine` (로짓 ONNX + `deploy.add_decode`, PTQ 엔트로피 캘리브레이션 102장) 으로 바꿨다.
QAT 그래프 디코드 엔진과 비교: 속도 2.057 vs 2.074 ms (profile_engine, 클럭 고정), fp32 대비 argmax 일치 99.41% vs 99.41%, 두 엔진끼리 99.94% (캘리브레이션 scale 의 미세한 차이). 같은 물건이고, 만드는 과정이 더 단순하다 (`results/confidence_agreement_1101_ptq_b/`).

빌드 교훈: 처음 빌드(15:32)는 다른 프로세스(사용자가 띄운 앱)가 CPU 4코어를 쓰는 동안 돌아 TensorRT 자동 튜닝이 conv 41개 중 36개를 FP16 으로 골랐고 2.69 ms 가 나왔다. 보드가 빈 뒤 벤치 타이밍 캐시(`models/timing.cache`)로 다시 빌드하니 INT8 48층, 2.06 ms. **엔진은 보드가 비었을 때 빌드하고, 빌드 뒤 매니페스트의 층 정밀도 수를 확인한다.**

## 사용자 GUI 세션 (ssh -X, 클럭 미고정, 2026-11-01 15:29~15:45)

앱이 기본으로 남기는 `logs/frame_log_latest.csv` 로 본 실제 화면 세션. 처음 2분(다른 작업이 겹치기 전): 처리 p50 541 ms, 1.83 fps, 깊이→3D·바닥 275 ms, 렌더 94 ms, 바닥 검출 99.6%.
같은 코드·엔진을 가상 디스플레이·클럭 고정으로 잰 284 ms 의 두 배다. 차이의 후보는 클럭 미고정(조건 JSON `gpu_clock_locked: false`)과 X 포워딩 렌더(94 ms). 클럭 효과를 분리하려고 가상 디스플레이에서 클럭만 풀고 180초를 더 쟀다 (PTQ 디코드 엔진, 15:51, `results/app_live_1101/int8_round4_ptq_unlocked/`).

| | 클럭 고정 (라운드 3) | 클럭 미고정 | 사용자 ssh -X 세션 (미고정, 처음 2분) |
|---|---|---|---|
| 처리 fps | 3.47 | 3.10 | 1.83 |
| 처리 ms p50 / p95 | 284 / 339 | 317 / 368 | 541 / 729 |
| SCNN 단계 (모델 실행) | 10.2 (3.8) | 15.1 (8.1) | 15.2 |
| 깊이→3D·바닥 / 코스트맵 | 63 / 189 | 69 / 206 | 275 / 234 |
| 화면 그리기 p50 | 58.8 | 9.2 | 94.1 |
| 입력 나이 / 지연 p50 | 116 / 321 | 132 / 357 | 327 / 578 |
| 보드 전력 W | 8.43 | 6.37 | (기록 없음) |
| GPU 클럭 | 918 고정 | 306 (사용률 2% 라 올라가지 않음) | 미고정 |

- 클럭 미고정의 대가는 12% 다 (3.47 → 3.10 fps). GPU 는 사용률이 2% 라 DVFS 가 306 MHz 에서 안 올라가 모델 실행이 3.8 → 8.1 ms 가 되고, CPU 는 1344 MHz 근처에 머문다. 대신 전력이 2 W 적다.
- 나머지 차이(317 → 541 ms)는 ssh -X 로 화면을 PC 로 보내는 비용으로 본다 (추측, 직접 분리하지 않음): 창 1900×980 을 5 Hz 로 보내면 약 28 MB/s 라 USB 링크와 sshd 암호화가 CPU 를 쓰고, 렌더 94 ms 가 그 증거다.
- 시연 권장: (1) Jetson 에 모니터를 붙인 데스크톱 세션에서 실행하거나 창을 작게. ssh -X 는 측정·디버그용으로만. (2) 시작 전 `sudo jetson_clocks` (+12%, GPU 클럭 고정). (3) `view.live_render_max_hz` 는 5~10.

## 바뀐 파일

- 측정: `d435map/runtime.py` (프레임별 CSV + 조건 JSON, `--duration`, 변화 있을 때만 렌더), `d435map/workers.py` (처리 결과 훅), `d435map/semantics.py` (SCNN 단계 시간, 그래프 디코드 결과 검사), `d435map/application.py` (`--duration`, `--frame-log`), `d435map/acquisition.py` (`poll_wait_s`), `config.json` (`view.live_render_max_hz` 5, `camera.poll_wait_ms` 10)
- 출력 동일 최적화: `obstacle_support.py`, `candidate_review.py`, `floor.py`, `fusion.py`, `costmap.py`, `semantic_free.py`, `field_of_view.py`, `geometry.py`
- GPU 디코드: `deploy/add_decode.py` (새 파일), `deploy/segmenter.py` (출력 여러 개), `d435map/trt_fastscnn.py` (`decode_in_graph`), `models/calib_trt_adds/fast_scnn_qat_decode*.json`, `models/logits/fast_scnn_decode*.json`, 앱 `models/fast_scnn_decode_int8.engine(.json)` (PTQ)
- 도구: `eval/app_timing.py`, `eval/app_profile.py` (새 파일), `eval/confidence_agreement.py` (디코드 엔진 지원)

## 실행

```bash
# 라이브 측정 (화면 없이, 180초) → 요약
cd camera/D435_Fusion_v3.5 && xvfb-run -a -s "-screen 0 1600x900x24" python v3.5.py --duration 180 --frame-log ../../results/app_live_X/frame_log.csv; cd ../..
python -m eval.app_timing before=results/app_live_1101/int8/frame_log.csv now=results/app_live_X/frame_log.csv --budget-ms 66.7 --skip 10 --out results/app_live_X
# 오프라인 프로파일 + 출력 동일성 (저장 캡처 20장)
python -m eval.app_profile --source results/app_profile_1101/captures --runs 3 --dump results/app_profile_1101/after_X --compare results/app_profile_1101/before --out results/app_profile_1101/after_X
# 새 캡처를 저장하며 프로파일 (카메라 연결)
python -m eval.app_profile --source camera --frames 20 --every 3 --save results/app_profile_X/captures --runs 3 --dump results/app_profile_X/before --out results/app_profile_X/before
```

되돌리기: 렌더·폴링은 `config.json` 의 `view.live_render_max_hz`(크게) 와 `camera.poll_wait_ms`(3) 로 예전 동작이 된다. 엔진은 앱 `models/` 에 정확히 하나만 두면 되고, 로짓 엔진·FP32 엔진은 `models/calib_trt_adds/`, `models/logits/` 에 그대로 있다.
