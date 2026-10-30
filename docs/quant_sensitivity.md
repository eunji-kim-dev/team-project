# INT8 양자화 블록별 민감도 (2026-10-30)

측정: `python -m eval.quant_sensitivity` (val 26장, PyTorch 가짜 양자화. scale·가중치는 Q/DQ ONNX 에서 읽어 엔진과 같은 상태로 재현). 전체 표는 `results/quant_sensitivity/<모델>/sensitivity.md`.
장→바 = 정답 장애물 픽셀 중 바닥으로 찍은 % (코스트맵에서 부딪히는 오류). CI95 = 사진 단위 paired bootstrap 2000회.

## 결론

1. **QDQ(EMA·64장) INT8 의 손상은 LTD.dsconv1 한 블록, 그중에서도 depthwise 3x3 의 활성화 scale 에서 나온다.** 그 conv 하나만 float 로 두면 바닥 IoU 가 fp32 수준으로 돌아오고(+0.0034, CI [+0.0011, +0.0059]), 활성화 쪽만 꺼도 같은 효과(+0.0029), 가중치 쪽만 끄면 효과 없음(+0.0001). 그 층의 EMA scale 은 TensorRT 엔트로피 scale 의 2.3배 (꼬리값 때문에 max 기반 scale 이 너무 큼).
2. **PTQ 가 QDQ 보다 나은 이유는 전부 활성화 scale 차이.** 활성화 scale 45개를 TensorRT 엔트로피 캘리브레이션 값으로 바꿔 끼우면 PyTorch 시뮬레이션이 PTQ 엔진 수치를 그대로 재현한다(mIoU 0.9313, 장→바 1.66%). LTD 5개만 바꿔도 mIoU 0.9317, fp32 일치 99.64% 로 PTQ(99.41%)보다 좋아진다. 캘리브레이션 장수·증강·max/EMA 는 이 문제를 못 고친다.
3. **QAT 300 iter 모델의 장→바 1.99% 는 양자화가 아니라 학습이 만든 것.** 학습된 가중치로 양자화를 전부 꺼도 장→바 2.008% (원 체크포인트 1.671%). 학습이 바→장을 5.30→4.39% 로 줄이는 대신 장→바를 올렸고, mIoU 만 보는 최고 모델 선택이 그걸 골랐다. 양자화 자체의 영향은 INT8 − float = mIoU +0.0001, 장→바 −0.005%p 로 없다.

## 기준 (PyTorch 시뮬레이션)

| 모델 | 상태 | mIoU | IoU 바닥 | IoU 장애물 | IoU 그 외 | 장→바% | 바→장% | fp32 일치% |
|---|---|---|---|---|---|---|---|---|
| 0iter (EMA·64장) | 전부 float (BN 접은 체크포인트) | 0.9319 | 0.8889 | 0.9626 | 0.9444 | 1.671 | 5.302 | 100.00 |
| 0iter (EMA·64장) | 전부 INT8 (= qdq_0iter 엔진) | 0.9296 | 0.8859 | 0.9614 | 0.9417 | 1.717 | 5.463 | 99.35 |
| 0iter, LTD 5개 scale 만 엔트로피 | 전부 INT8 | 0.9317 | 0.8881 | - | - | 1.695 | 5.296 | 99.64 |
| 0iter, 45개 scale 전부 엔트로피 | 전부 INT8 | 0.9313 | 0.8876 | - | - | 1.659 | 5.481 | 99.42 |
| (참고) TensorRT PTQ 엔진 (엔트로피·102장) | INT8 | 0.9313 | 0.8878 | 0.9623 | 0.9438 | 1.646 | 5.504 | 99.41 |
| QAT 300 iter | 학습된 가중치, 양자화 끔 | 0.9312 | 0.8864 | 0.9616 | 0.9456 | 2.008 | 4.392 | 100.00 |
| QAT 300 iter | 전부 INT8 (= qat_int8 엔진) | 0.9313 | 0.8863 | 0.9616 | 0.9459 | 2.003 | 4.424 | 99.38 |

## 0iter 모델: 단위별 양자화를 끄면 (leave-one-out, 전부-INT8 대비. 음수 = 회복)

INT8 − fp32 거리: mIoU −0.0023, 바닥 IoU −0.0030, 장→바 +0.046%p

| 단위 | conv 수 | Δ장→바%p | CI95 | Δ바닥 IoU | CI95 | ΔmIoU | fp32 일치% |
|---|---|---|---|---|---|---|---|
| 스테이지 LTD | 5 | −0.025 | [−0.101, +0.056] | +0.0026 | [+0.0002, +0.0053] | +0.0021 | 99.87 |
| 스테이지 FFM | 3 | −0.020 | [−0.032, −0.011] | +0.0003 | [−0.0001, +0.0007] | +0.0001 | 99.36 |
| 스테이지 HEAD | 5 | −0.017 | [−0.028, −0.008] | +0.0002 | [−0.0001, +0.0005] | +0.0001 | 99.36 |
| 스테이지 GFE.bottleneck1·2·3, ppm_out | 9·9·9·5 | −0.007 ~ +0.004 | 전부 0 포함 | +0.0001 ~ +0.0004 | 전부 0 포함 | ±0.0001 | 99.35 |
| 블록 LTD.dsconv1 | 2 | −0.084 | [−0.163, −0.016] | +0.0034 | [+0.0006, +0.0060] | +0.0026 | 99.63 |
| 블록 LTD.dsconv2 | 2 | +0.047 | [+0.016, +0.087] | −0.0006 | [−0.0019, +0.0007] | −0.0006 | 99.38 |
| 블록 LTD.conv (첫 conv) | 1 | −0.012 | [−0.075, +0.043] | −0.0004 | [−0.0016, +0.0007] | −0.0004 | 99.36 |
| 블록 HEAD.convs0 | 2 | −0.022 | [−0.038, −0.010] | +0.0002 | [−0.0001, +0.0007] | +0.0000 | 99.36 |
| 블록 FFM | 3 | −0.020 | [−0.032, −0.011] | +0.0003 | [−0.0001, +0.0007] | +0.0001 | 99.36 |

그 단위만 INT8 로 두면(only-one): LTD 만 INT8 → fp32 일치 99.37% (전부 INT8 99.35% 와 같음), 나머지 스테이지는 각각 99.91~99.99%. 즉 양자화 오차는 거의 전부 LTD 에서 생긴다.

## 0iter 모델: conv 단위 (장→바 회복 순 상위, + 활성화/가중치 분리)

| conv | EMA/entropy | off Δ장→바 | CI95 | 활성화만 off | 가중치만 off | off Δ바닥 IoU | CI95 | 활성화만 off | 가중치만 off | off 일치% |
|---|---|---|---|---|---|---|---|---|---|---|
| LTD.dsconv1.depthwise_conv | 2.30 | −0.051 | [−0.132, +0.018] | −0.052 | +0.022 | **+0.0034** | [+0.0011, +0.0059] | **+0.0029** | +0.0001 | 99.57 |
| LTD.dsconv1.pointwise_conv | 1.96 | −0.029 | [−0.074, +0.020] | −0.037 | −0.022 | +0.0014 | [−0.0005, +0.0034] | +0.0011 | +0.0011 | 99.43 |
| HEAD.convs.0.pointwise_conv | 2.52 | −0.021 | [−0.032, −0.011] | −0.018 | −0.004 | +0.0001 | [−0.0002, +0.0005] | +0.0001 | +0.0001 | 99.36 |
| FFM.conv_lower_res | 5.83 | −0.018 | [−0.028, −0.009] | −0.010 | −0.009 | +0.0003 | [−0.0000, +0.0007] | +0.0002 | +0.0001 | 99.36 |
| LTD.conv (첫 conv) | 1.01 | −0.012 | [−0.075, +0.043] | +0.013 | −0.015 | −0.0004 | [−0.0016, +0.0007] | −0.0004 | −0.0001 | 99.36 |
| 나머지 40개 | 0.70~2.86 | −0.009 ~ +0.035 | 대부분 0 포함 | | | −0.0007 ~ +0.0007 | 대부분 0 포함 | | | 99.35~99.38 |

EMA/entropy = 이 ONNX 의 활성화 scale ÷ TensorRT PTQ 엔트로피 scale. HEAD.convs.0.pointwise_conv 와 FFM.conv_lower_res 는 장→바만 조금 움직이고(CI 가 0 을 안 걸침) 바닥 IoU·일치율은 안 움직임 → 경계 픽셀 몇 개 수준.

누적 (회복 순으로 float 로 두기): dsconv1 의 depthwise 1개 → 바닥 IoU 0.8893, 일치 99.57% / dsconv1 2개 → 0.8892, mIoU +0.0026 (fp32 까지 +0.0023), 장→바 1.633%, 일치 99.63% / 3개 이상 → 더 안 좋아짐. 일치율 기준으로 LTD 5개 전부 float → 99.83%.

## QAT 300 iter 모델: 단위별

양자화를 끄는 단위를 바꿔도 바닥 IoU 는 ±0.0003 안, 장→바는 −0.018 ~ +0.015%p 로 전부 노이즈 수준. 학습이 LTD.dsconv1 의 coarse scale 에 가중치를 맞춰 놓아(QAT 가 할 일은 했음) 양자화 손상은 없어졌지만, 그 대신 float 가중치 자체의 장→바가 2.008% 로 올라갔다. 전체 표: `results/quant_sensitivity/fast_scnn_qat/sensitivity.md`.

## TensorRT 엔진에서 확인

`eval.bench_engines` 로 val 26장, 3회 반복 (2026-10-30 12:41:19, NV Power Mode: 25W 1). 전체: `results/quant_sensitivity/trt_verify/bench_table.md`.
cpu_fp32 (onnxruntime): mIoU 0.9319, IoU 바닥 0.8889, 장→바 1.671%, 바→장 5.301%.

| 엔진 | mIoU | IoU 바닥 | IoU 장애물 | IoU 그 외 | 장→바% | 바→장% | cpu_fp32 대비 일치% | 일치% 바닥 | 활성화 MB | p50 ms(추론만) | 반복 변동% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 (참고) | 0.9320 | 0.8889 | 0.9626 | 0.9444 | 1.669 | 5.304 | 99.99 | 99.98 | 8.4 | 4.79 | 46.2 |
| ptq_int8 (TensorRT 엔트로피·102장) | 0.9313 | 0.8878 | 0.9623 | 0.9438 | 1.646 | 5.504 | 99.41 | 98.57 | 6.7 | 6.669 | 43.27 |
| qdq_ema64 (기존 QDQ 0iter, EMA·64장) | 0.9295 | 0.8860 | 0.9613 | 0.9413 | 1.717 | 5.452 | 99.35 | 98.59 | 6.8 | 5.088 | 46.87 |
| floatLTD (LTD 5개 Q/DQ 제거 → TensorRT 가 FP16 으로 돌림) | 0.9320 | 0.8886 | 0.9625 | 0.9448 | 1.687 | 5.271 | 99.86 | 99.74 | 8.1 | 4.835 | 34.11 |
| trtscaleLTD (전부 INT8, LTD 5개 활성화 scale 만 엔트로피) | 0.9315 | 0.8880 | 0.9623 | 0.9443 | 1.696 | 5.300 | 99.63 | 99.20 | 6.8 | 4.674 | 40.45 |
| trtscale_all (전부 INT8, 활성화 scale 45개 전부 엔트로피) | 0.9313 | 0.8877 | 0.9622 | 0.9439 | 1.653 | 5.485 | 99.42 | 98.60 | 6.8 | 5.083 | 45.11 |

- PyTorch 시뮬레이션 예측이 엔진에서 그대로 나옴: scale 전부 엔트로피 → PTQ 엔진과 같은 수치(0.9313 / 1.65% / 99.42%), LTD 만 엔트로피 → 99.63%, LTD 를 float 로 → mIoU 0.9320·바닥 0.8893 (fp32 와 같음), 일치 99.86%.
- 속도 열은 이번 측정에서 비교 불가: jetson_clocks 를 못 켠 상태라 반복 변동이 34~47% (팀 기준 1~2%). 클럭 고정 후 다시 잴 것. 활성화 메모리는 믿을 수 있음: LTD 를 FP16 으로 두면 +1.3 MB (고해상도 층이라서).


## 후속: TensorRT 캐시 scale 가져오기 + 장→바 조건으로 모델 선택 (2026-10-30 14시)

바꾼 것

- `deploy/trt_scales.py` (새 파일): TensorRT PTQ 캘리브레이션 캐시(`models/fast_scnn_int8_entropy.calib`, 엔트로피·102장)의 scale 을 QuantConv2d 45개에 넣는다. 대응은 캐시를 만든 FP32 ONNX 의 Conv 노드 → 입력 텐서 이름(= 캐시 항목) → 모듈 이름으로 잡고, 45개 전부 정확히 하나씩 대응되는지와 가중치 모양이 같은지 검사한다 (하나라도 어긋나면 중단). 캐시의 나머지 70개 항목은 conv 입력이 아닌 텐서(사진 입력, conv 출력, Add·Resize·MatMul 출력)라 쓰지 않는 게 맞다. `python -m deploy.trt_scales --calib <캐시> --ckpt <pth>` 로 대응표를 볼 수 있다.
- `deploy/qat_train.py --trt-calib <캐시>`: EMA/max 캘리브레이션 뒤에 활성화 scale 을 캐시 값으로 바꾼다 (`--trt-scale-convs LTD` 처럼 일부만도 가능). 쓴 scale 45개(캐시 값과 바꾸기 전 EMA 값)와 캐시 출처(경로·헤더·sha256·사진 수·만든 엔진 json)를 `<out>.json` 의 `qat.act_calib` 에 남긴다.
- 최고 모델 선택 규칙: 후보 = 학습 전(0 iter) 모델 + eval 때마다의 가중치. 같은 PyTorch 평가에서 잰 장→바% 가 학습 전 모델보다 늘지 않은 후보 중 mIoU 최고 (`--obs-floor-tol`, 기본 0 %p. `inf` 면 예전처럼 mIoU 만). 후보별 수치와 채택/제외 이유는 `qat.selection` 에. `evaluate()` 가 장→바%·바→장%·혼동행렬도 돌려준다.
- `deploy/quant.py`: `quantize_convs` 가 만든 QuantConv2d 의 scale 버퍼가 CPU 에 남아 GPU 에서 캘리브레이션이 죽던 문제 수정 (지금까지 qat_train 은 PC CPU 에서만 돌았음. Orin GPU 에서 300 iter 가 64초).

결과 (PyTorch 시뮬레이션, val 26장. iters 300, lr 1e-4, batch 4, seed 42, EMA·64장 캘리브레이션 뒤 캐시 scale 45개 적용)

| 후보 | mIoU | IoU 바닥 | IoU 장애물 | IoU 그 외 | 장→바% | 바→장% | 판정 |
|---|---|---|---|---|---|---|---|
| fp32 (BN 접음, 참고) | 0.9319 | 0.8889 | 0.9626 | 0.9444 | 1.671 | 5.302 | - |
| 새 0iter (scale 45개 = 캐시, 새 기준) | 0.9313 | 0.8876 | 0.9622 | 0.9440 | 1.659 | 5.481 | 후보 0 |
| iter 50 | 0.9314 | 0.8878 | 0.9622 | 0.9441 | 1.721 | 5.236 | 장→바 > 기준 → 제외 |
| **iter 100** | **0.9324** | **0.8894** | 0.9628 | 0.9451 | **1.650** | 5.322 | **선택** |
| iter 150 | 0.9311 | 0.8863 | 0.9620 | 0.9451 | 1.674 | 5.566 | 제외 |
| iter 200 | 0.9308 | 0.8846 | 0.9612 | 0.9465 | 2.040 | 4.476 | 제외 |
| iter 250 | 0.9312 | 0.8857 | 0.9618 | 0.9462 | 1.820 | 5.124 | 제외 |
| iter 300 | 0.9315 | 0.8863 | 0.9619 | 0.9462 | 1.811 | 5.082 | 제외 |
| (참고) 기존 QAT 300 iter: EMA scale, PC CPU 학습, mIoU 만으로 선택 | 0.9313 | 0.8863 | 0.9616 | 0.9459 | 2.003 | 4.424 | - |

- 새 0iter 는 앞 절의 시뮬레이션(scale 45개 전부 엔트로피)과 같은 수치(0.9313 / 0.8876 / 1.659%) → 가져오기가 맞게 됐다. 이게 이후 QAT 비교의 기준.
- 학습이 진행될수록 장→바가 오른다(iter 200 이후 1.8~2.0%). 바→장을 줄이는 쪽으로 가중치가 움직이면서 장→바가 늘어나는, 기존 QAT 와 같은 경향. 7개 후보 중 5개가 장→바 조건에 걸렸다.
- 이번 run 은 mIoU 최고(iter 100)가 장→바도 최저라 예전 규칙(mIoU 만)으로도 같은 모델이 뽑혔을 것. 규칙이 실제로 막은 건 iter 300(mIoU 0.9315, 장→바 1.811%) 같은 후보이고, 학습이 전부 나쁘면 0iter 가 그대로 나가는 안전장치가 생겼다.
- iter 100 이 fp32 보다 mIoU +0.0005, 바닥 IoU +0.0005 높지만 val 26장에서 이 크기는 노이즈와 구분이 안 된다 (앞 절 CI 폭 참고). "fp32 와 같은 수준" 으로 읽는 게 맞다.

### TensorRT 엔진에서 확인 (2026-10-30 14:22, 25W, jetson_clocks 고정, val 26장, 5회 반복, 쿨다운 30초)

전체 표: `results/qat_trt_select/bench_table.md`. 반복 변동 3.5~3.7% (클럭 고정 전 34~47%). cpu_fp32 (onnxruntime): mIoU 0.9319, 바닥 0.8889, 장→바 1.671%, 바→장 5.301%.

| 엔진 | mIoU | IoU 바닥 | IoU 장애물 | IoU 그 외 | 장→바% | 바→장% | cpu_fp32 대비 일치% | 일치% 바닥 | p50 ms(추론만) | p50 ms(전체) | 활성화 MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 (참고) | 0.9320 | 0.8889 | 0.9626 | 0.9444 | 1.669 | 5.304 | 99.99 | 99.98 | 3.44 | 6.97 | 8.4 |
| ptq_int8 (TensorRT 엔트로피·102장) | 0.9313 | 0.8878 | 0.9623 | 0.9438 | 1.646 | 5.504 | 99.41 | 98.57 | 2.96 | 6.47 | 6.7 |
| qdq_ema64 (기존 0iter, EMA·64장) | 0.9295 | 0.8860 | 0.9613 | 0.9413 | 1.717 | 5.452 | 99.35 | 98.59 | 3.14 | 6.64 | 6.8 |
| qat_ema64 (기존 QAT 300 iter, mIoU 만으로 선택) | 0.9313 | 0.8864 | 0.9617 | 0.9460 | 1.994 | 4.445 | 99.07 | 98.98 | 3.16 | 6.68 | 6.8 |
| **qdq_trt** (새 0iter 기준: scale 45개 = 캐시) | 0.9313 | 0.8877 | 0.9622 | 0.9439 | 1.653 | 5.485 | 99.42 | 98.60 | 3.15 | 6.67 | 6.8 |
| **qat_trt** (선택된 iter 100) | **0.9324** | **0.8894** | 0.9628 | 0.9450 | **1.647** | 5.332 | 99.17 | 98.13 | 3.15 | 6.64 | 6.8 |

- PyTorch 시뮬레이션 수치가 엔진에서 그대로 나왔다: qdq_trt 0.9313 / 0.8877 / 1.653% (시뮬 0.9313 / 0.8876 / 1.659%), qat_trt 0.9324 / 0.8894 / 1.647% (시뮬 0.9324 / 0.8894 / 1.650%).
- qdq_trt 는 PTQ 엔진과 같은 수치 → QDQ 의 출발점이 PTQ 와 같아졌다. qat_trt 는 거기서 mIoU +0.0011, 바닥 IoU +0.0017 올라 fp16 과 같은 수준이고, 장→바는 PTQ 와 같은 1.65% (기존 QAT 1.99%).
- qat_trt 의 cpu_fp32 대비 일치율(99.17%)이 qdq_trt(99.42%)보다 낮은 건 학습으로 가중치가 바뀌어 fp32 와 다른 모델이 됐기 때문. 학습된 모델엔 일치율이 품질 지표가 아니다 (정답 기준 IoU·장→바로 볼 것).
- 속도: INT8 엔진 전부 추론 2.96~3.16 ms, 전체 6.5~6.7 ms (≈150 FPS). Q/DQ 엔진이 PTQ 엔진보다 ~0.2 ms 느린 건 10/29 측정과 같은 경향 (추정: Q/DQ 배치 때문에 층 융합이 달라 86층 vs 80층). 활성화 메모리는 전부 6.8 MB.

## 재현

```bash
# 민감도 전체 (stage·block·conv + 누적 8개): 모델당 ~5분 (Jetson GPU)
python -m eval.quant_sensitivity --ckpt checkpoints/best_mIoU_iter_1600.pth --data-root data \
    --qdq-onnx models/fast_scnn_qat_0iter.onnx --out results/quant_sensitivity
# 활성화 scale 을 TensorRT 엔트로피 값으로 바꿔서 측정 + 그 상태의 Q/DQ ONNX 내보내기
python -m eval.quant_sensitivity --ckpt ... --qdq-onnx models/fast_scnn_qat_0iter.onnx --levels --cumulative 0 \
    --trt-scales-for LTD --export-name fast_scnn_qat_0iter_trtscaleLTD --run-name exports
# LTD 5개를 float 로 둔 Q/DQ ONNX
python -m eval.quant_sensitivity --ckpt ... --qdq-onnx models/fast_scnn_qat_0iter.onnx --levels --cumulative 0 \
    --export-convs LTD --export-name fast_scnn_qat_0iter_floatLTD --run-name exports
python -m deploy.build_engine --onnx results/quant_sensitivity/exports/fast_scnn_qat_0iter_floatLTD.onnx --precision qat

# TensorRT 캐시 scale 로 새 0iter 기준 + QAT 300 iter (장→바 조건 선택 포함) → 엔진 → 벤치
python -m deploy.trt_scales --calib models/fast_scnn_int8_entropy.calib --ckpt checkpoints/best_mIoU_iter_1600.pth   # 대응표만
python -m deploy.qat_train --ckpt checkpoints/best_mIoU_iter_1600.pth --data-root data --iters 0 \
    --out models/calib_trt/fast_scnn_qat_0iter.onnx --trt-calib models/fast_scnn_int8_entropy.calib
python -m deploy.qat_train --ckpt checkpoints/best_mIoU_iter_1600.pth --data-root data --iters 300 --lr 1e-4 --batch 4 --seed 42 \
    --out models/calib_trt/fast_scnn_qat.onnx --trt-calib models/fast_scnn_int8_entropy.calib
python -m deploy.build_engine --onnx models/calib_trt/fast_scnn_qat.onnx --precision qat
python -m eval.bench_engines --models cpu_fp32=models/fast_scnn.onnx ptq_int8=models/fast_scnn_int8.engine \
    qdq_trt=models/calib_trt/fast_scnn_qat_0iter_int8.engine qat_trt=models/calib_trt/fast_scnn_qat_int8.engine \
    --images data/images --gt data/labels_RGB --list data/splits/val.txt --out results/qat_trt_select --runs 5 --cooldown 30
```
