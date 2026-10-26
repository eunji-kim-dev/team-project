# Jetson 개발 환경 기록

기록일: 2026-10-26

## 1. 시스템 및 GPU 개발 환경

| 항목 | 버전 |
|---|---|
| NVIDIA L4T | 39.2.1 |
| CUDA Toolkit | 13.2 (V13.2.86) |
| cuDNN | 9.20.0.46 |
| TensorRT | 10.16.2.10 |
| TensorRT Python 바인딩 패키지 | 10.16.2.10 |
| PyTorch | 2.12.0+cu132 |
| ONNX | 1.23.2 |

## 2. 프로젝트 용도

- Fast-SCNN 모델의 ONNX 변환
- TensorRT FP32 / FP16 엔진 생성
- Jetson GPU 추론
- 추론 속도, 정확도 및 메모리 사용량 비교
- RGB-D 기반 장애물 분할 및 로컬 Costmap 생성

## 3. 추가 확인 항목

- Python 버전
- Ubuntu 및 JetPack 버전
- PyTorch CUDA 사용 가능 여부
- 실제 가상환경의 TensorRT Python import 성공 여부

## 4. 환경 관리

- Python 패키지 목록: `requirements-jetson.txt`
- 시스템 라이브러리 버전: 본 문서에 별도 기록
- TensorRT 엔진 생성 시 사용한 정밀도와 빌드 옵션 기록
