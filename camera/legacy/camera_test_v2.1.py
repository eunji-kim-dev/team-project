# v2.1 코드 리뷰 안내
# 기존 camera_test.py를 v2로 확장하면서 추가·변경한 기능의 목적과 필요성을 설명합니다.
# v2.1은 설명을 보강한 버전이며 v2의 정렬·저장·실행 로직은 변경하지 않습니다.
# [추가] 새 처리의 필요성 / [변경] 기존 방식에서 바꾼 이유 / [주의] 해석상 한계입니다.

"""RealSense RGB-D 수집: 원본 + RGB에 정렬한 Depth + 보정값/시각 기록.

필수 패키지: pip install numpy opencv-python pyrealsense2
실행: python camera_test_v2.1.py
S: 현재 프레임 저장 / Q, Esc: 종료

기본 저장 위치: 이 파일 옆 captures_v2/session_날짜_시각_고유번호/
각 capture 폴더에는 color.png, depth_raw.npy, depth_meters.npy,
depth_aligned_raw.npy, depth_aligned_meters.npy, valid_mask.png,
depth_aligned_vis.png, preview_rgb_depth.png, metadata.json이 저장됩니다.
PNG는 무손실입니다. 실제 깊이는 시각화 이미지가 아닌 NPY를 사용하세요.

기본 해상도는 640x480입니다. USB 3은 30fps, USB 2는 15fps를 먼저
사용하고, 기본 설정의 시작 실패 시 지원되는 더 낮은 FPS를 시도합니다.
--fps로 지정하면 해당 FPS만 사용합니다. 예: --fps 15
--output으로 저장 위치를 지정할 수 있습니다.
카메라는 일반 Windows 터미널/탐색기에서 실행해야 접근할 수 있습니다.
"""
# [추가] 타입 힌트로 배열·메타데이터의 역할을 드러내고, 지연 평가로 선언 순서에 따른 의존성을 줄입니다.
from __future__ import annotations

# [추가] argparse는 코드 수정 없는 옵션 설정, deque는 최근 수신 FPS 계산에 사용합니다.
# dataclass는 같은 RGB-D의 배열·시각을 묶고, datetime/time은 기록 시각과 지연 검사를 담당합니다.
# hashlib/uuid는 무결성 확인과 이름 충돌 방지, json/Path는 보정값 저장과 경로 관리에 필요합니다.
# importlib.metadata는 촬영 당시 SDK 버전을 남겨 실행 환경을 재현하기 위해 추가했습니다.
import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np
import pyrealsense2 as rs

# [변경] 터미널의 현재 위치가 달라도 저장 위치가 바뀌지 않도록 스크립트 폴더를 기준으로 합니다.
# 창 이름을 상수로 관리해 창 생성·이벤트 처리·종료 대상이 일치하도록 합니다.
HERE = Path(__file__).resolve().parent
WINDOW = 'RealSense Capture v2 | RGB + Aligned Depth | S: save / Q: quit'


# [추가] 다른 PC에서도 시각을 해석할 수 있게 로컬 시간대를 포함한 ISO 형식으로 기록합니다.
# 이 값은 PC의 기록 시각이며, 센서의 촬영 시각은 frame_info에서 별도로 기록합니다.
def now_iso():
    return datetime.now().astimezone().isoformat()


# [추가] 해상도가 같아도 두 센서의 픽셀은 대응하지 않으므로 스트림별 내부 파라미터를 보존합니다.
# fx/fy는 픽셀 단위 초점거리, ppx/ppy는 영상 중심, model/coeffs는 왜곡 모델과 계수입니다.
# 해상도·FPS·포맷을 함께 남겨 다른 촬영 설정의 보정값을 잘못 적용하지 않도록 합니다.
def intrinsics_info(profile):
    video = profile.as_video_stream_profile()
    intr = video.get_intrinsics()
    return {
        'width': intr.width, 'height': intr.height, 'fx': intr.fx, 'fy': intr.fy,
        'ppx': intr.ppx, 'ppy': intr.ppy, 'distortion_model': str(intr.model).split('.')[-1],
        'coeffs': list(intr.coeffs), 'fps': video.fps(), 'format': str(video.format()),
    }


# [추가] RGB와 Depth 센서의 물리적 위치·방향 차이를 보정하려면 좌표계 사이 회전·이동 정보가 필요합니다.
# 회전 배열 저장 순서(column-major), 이동 단위(m), 변환 방향을 명시해 재투영 실수를 줄입니다.
def extrinsics_info(source, target):
    extr = source.get_extrinsics_to(target)
    return {'rotation': list(extr.rotation), 'rotation_storage': 'column-major',
            'translation_m': list(extr.translation), 'mapping': 'P_target = R * P_source + t'}


# [추가] 같은 모델도 개체별 보정값이 달라 일련번호·펌웨어를 기록하고, USB 속도로 수신 성능을 해석합니다.
# 장치가 지원하는 정보만 읽어 모델별 예외를 줄입니다.
def device_info(device):
    result = {}
    for name in ('name', 'serial_number', 'firmware_version', 'usb_type_descriptor', 'product_id'):
        field = getattr(rs.camera_info, name)
        if device.supports(field):
            result[name] = device.get_info(field)
    return result


# [추가] 이전 자료에는 움직이는 대상의 RGB/Depth 시점 차이를 추적할 기록이 없었습니다.
# 프레임 번호·시각·시계 종류와 지원되는 노출/수신 정보를 저장해 시간 대응 문제를 분석합니다.
# [주의] 두 센서의 프레임 번호가 같다고 같은 촬영 시점이라고 단정할 수는 없습니다.
def frame_info(frame):
    info = {'frame_number': int(frame.get_frame_number()),
            'timestamp_ms': float(frame.get_timestamp()),
            'timestamp_domain': str(frame.get_frame_timestamp_domain()), 'metadata': {}}
    for name in ('actual_exposure', 'gain_level', 'actual_fps', 'frame_counter',
                 'frame_timestamp', 'sensor_timestamp', 'backend_timestamp', 'time_of_arrival'):
        # [추가] SDK·펌웨어별로 제공 항목이 달라 속성의 존재와 해당 프레임의 지원 여부를 모두 확인합니다.
        field = getattr(rs.frame_metadata_value, name, None)
        if field is not None and frame.supports_frame_metadata(field):
            try:
                info['metadata'][name] = int(frame.get_frame_metadata(field))
            # [추가] 부가 메타데이터 읽기 실패로 전체 수집을 중단하지 않고 해당 항목만 생략합니다.
            except RuntimeError:
                pass
    return info


# [추가] 공간 정렬로 촬영 시점 차이를 해결할 수는 없어 저장 전 RGB/Depth 시각 차이를 검사합니다.
# 다른 종류의 시계 숫자는 직접 비교할 수 없으므로 timestamp_domain이 같을 때만 차이를 계산합니다.
# 시각이 유효하지 않거나 허용 차이를 넘으면 저장 불가로 표시합니다.
# [주의] 같은 시계 종류도 완전한 동기화를 증명하지 않으며 하드웨어 동기화나 노출 일치를 보증하지 않습니다.
def pairing_info(depth_info, color_info, max_skew_ms):
    same_domain = depth_info['timestamp_domain'] == color_info['timestamp_domain']
    delta = abs(depth_info['timestamp_ms'] - color_info['timestamp_ms']) if same_domain else None
    comparable = delta is not None and bool(np.isfinite(delta))
    return {'source': 'same RealSense SDK frameset', 'timestamp_domains_match': same_domain,
            'timestamp_delta_ms': delta if comparable else None, 'max_skew_ms': max_skew_ms,
            'within_threshold': comparable and delta <= max_skew_ms,
            'note': 'Timestamp screening is not proof of identical exposure times or hardware synchronization.'}


# [추가] 촬영 당시 보정값이 없어 나중에 현재 카메라에 의존해야 하는 상황을 방지합니다.
# 실제 스트림의 내부 파라미터, 양방향 외부 파라미터, 깊이 단위와 SDK 버전을 한 묶음으로 남깁니다.
def calibration_info(device, depth_profile, color_profile, scale):
    return {
        'schema': 'realsense-capture-v2-calibration', 'recorded_at': now_iso(),
        'sdk_version': importlib.metadata.version('pyrealsense2'), 'device': device_info(device),
        'depth_scale_m': float(scale), 'depth_intrinsics': intrinsics_info(depth_profile),
        'color_intrinsics': intrinsics_info(color_profile),
        'depth_to_color': extrinsics_info(depth_profile, color_profile),
        'color_to_depth': extrinsics_info(color_profile, depth_profile),
    }


# [추가] x 모드로 기존 파일 덮어쓰기를 막고 UTF-8로 한글 메타데이터를 저장합니다.
# NaN/Infinity는 표준 JSON 값이 아니므로 조용히 기록하지 않고 오류로 드러나게 합니다.
def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


# [변경] 깊이를 JPEG 색상이 아닌 NPY 수치 배열로 저장해 원시 uint16/미터 float32 값을 유지합니다.
# xb 모드로 덮어쓰기를 막고 숫자 배열에 불필요한 pickle 객체 저장은 허용하지 않습니다.
def write_array(path, array):
    with Path(path).open('xb') as stream:
        np.save(stream, array, allow_pickle=False)


# [변경] RGB를 무손실 PNG로 저장하여 JPEG 재압축에 따른 색상·경계 변화를 피합니다.
# 한글 Windows 경로의 OpenCV 직접 저장 오류를 피하려고 메모리 인코딩 후 Python으로 씁니다.
# 인코딩 성공 여부와 파일 신규 생성 여부도 검사하여 저장 실패를 감지합니다.
def write_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'PNG 인코딩 실패: {path}')
    with Path(path).open('xb') as stream:
        stream.write(encoded.tobytes())


# [변경] 실제 depth_scale로 미터를 계산한 후 0.15~6m의 고정 범위를 화면 색상에 대응시킵니다.
# 장면마다 색의 거리 의미가 바뀌지 않게 하고 미측정 0은 검정으로 구별합니다.
# [주의] 클리핑은 확인용 색상에만 적용합니다. NPY 실제 깊이 수치는 이 범위로 자르지 않습니다.
def depth_visualization(raw, scale):
    meters = raw.astype(np.float32) * np.float32(scale)
    normalized = np.clip((meters - 0.15) / (6.0 - 0.15), 0, 1)
    image = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    image[raw == 0] = 0
    return image


# [추가] 같은 SDK 프레임 묶음의 RGB·원본 깊이·정렬 깊이·시각 정보를 하나의 객체로 관리합니다.
# 최신 RGB와 이전 Depth가 따로 남아 잘못 짝지어 저장되는 실수를 줄이기 위한 구조입니다.
@dataclass
class Capture:
    color: np.ndarray
    # [추가] 센서 원본 좌표와 RGB 기준 정렬 좌표를 별도 필드로 구분하여 원본을 보존합니다.
    original_depth: np.ndarray
    aligned_depth: np.ndarray
    depth_frame: dict
    color_frame: dict
    aligned_depth_intrinsics: dict
    pairing: dict
    # [추가] monotonic은 PC 시계 변경과 무관한 지연 검사, 나머지 시각은 기록과 다른 로그 대조에 씁니다.
    received_monotonic: float
    received_at: str
    received_unix_ns: int


# [변경] 같은 frameset에서 RGB·Depth를 함께 꺼내 그 묶음에 SDK 공간 정렬을 적용합니다.
# 배열을 복사해 다음 프레임이 SDK 버퍼를 재사용해도 저장할 데이터가 바뀌지 않게 합니다.
def capture_from_frames(frames, aligner, calibration, max_skew_ms):
    depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None
    received_at, received_ns, received_mono = now_iso(), time.time_ns(), time.monotonic()
    # [변경] 정렬 전에 원본 버퍼를 복사합니다. 원본 Depth 필드를 정렬 결과로 대체하지 않습니다.
    raw = np.asanyarray(depth_frame.get_data()).copy()
    color = np.asanyarray(color_frame.get_data()).copy()
    depth_details, color_details = frame_info(depth_frame), frame_info(color_frame)
    # [핵심 변경] SDK가 내부·외부 파라미터로 원본 Depth를 RGB 픽셀 좌표에 재투영합니다.
    # resize나 고정 픽셀 이동으로 센서의 시야·위치 차이를 흉내 내는 방식이 아닙니다.
    aligned_frames = aligner.process(frames)
    aligned_frame = aligned_frames.get_depth_frame()
    if not aligned_frame:
        return None
    aligned = np.asanyarray(aligned_frame.get_data()).copy()
    # [추가] 잘못된 스트림이 정상 데이터처럼 저장되지 않도록 자료형과 크기를 검사합니다.
    # NumPy shape은 (높이, 너비)이므로 640×480 영상은 (480, 640)으로 검사합니다.
    if raw.dtype != np.uint16 or aligned.dtype != np.uint16:
        raise RuntimeError('Depth는 Z16(uint16) 형식이어야 합니다.')
    if color.dtype != np.uint8 or color.shape != (480, 640, 3) or raw.shape != (480, 640):
        raise RuntimeError('요청한 원본 640x480 스트림과 실제 배열 크기가 다릅니다.')
    if aligned.shape != color.shape[:2]:
        raise RuntimeError('정렬 Depth와 RGB 크기가 다릅니다.')
    # [추가] 정렬 Depth의 해상도뿐 아니라 초점거리·영상 중심도 RGB 기준인지 검증합니다.
    # 정렬 후 RGB 배열도 비교하여 원본 RGB가 변형되지 않았는지 확인합니다.
    aligned_intr = intrinsics_info(aligned_frame.profile)
    for key in ('width', 'height', 'fx', 'fy', 'ppx', 'ppy'):
        if not np.isclose(aligned_intr[key], calibration['color_intrinsics'][key]):
            raise RuntimeError('정렬 Depth의 내부 파라미터가 RGB 좌표계와 일치하지 않습니다.')
    if not np.array_equal(np.asanyarray(aligned_frames.get_color_frame().get_data()), color):
        raise RuntimeError('정렬 과정에서 원본 RGB가 변경됐습니다.')
    return Capture(color, raw, aligned, depth_details, color_details, aligned_intr,
                   pairing_info(depth_details, color_details, max_skew_ms),
                   received_mono, received_at, received_ns)


# [변경] 샘플마다 독립 폴더를 만들어 원본·정렬본·보정값을 하나의 자료 단위로 관리합니다.
# 저장 중 중단된 샘플을 완성 자료로 오인하지 않도록 임시 폴더에서 먼저 씁니다.
def save_capture(session_dir, index, capture, calibration):
    """Commit one complete sample to a new directory; never overwrite old files."""
    session_dir = Path(session_dir)
    # [추가] 시각 대응이 나쁘거나 유효 깊이가 없는 샘플은 학습 자료로 저장하지 않도록 먼저 검사합니다.
    if not capture.pairing['within_threshold']:
        raise ValueError('RGB/Depth 시각 차이가 허용 범위를 벗어나 저장하지 않습니다.')
    if not (capture.original_depth > 0).any() or not (capture.aligned_depth > 0).any():
        raise ValueError('유효 깊이가 없어 저장하지 않습니다.')
    stem = f'capture_{index:06d}'
    final = session_dir / stem
    # [추가] 일부 파일만 저장된 상태는 .pending으로 남기고 완료 후에만 최종 이름으로 바꿉니다.
    # 기존 폴더나 미완성 폴더가 같은 번호로 있으면 덮어쓰지 않고 오류로 알립니다.
    pending = session_dir / f'.{stem}.pending'
    if final.exists() or pending.exists():
        raise FileExistsError(f'기존 촬영 폴더를 덮어쓰지 않습니다: {stem}')
    pending.mkdir()
    scale = np.float32(calibration['depth_scale_m'])
    try:
        # [변경] 원본 Depth는 센서 좌표계, 정렬 Depth는 RGB 좌표계로 나누어 저장합니다.
        # 각각 정수 원시값과 미터 단위 실수값을 남겨 수치 보존과 활용 편의를 함께 제공합니다.
        # 원시값을 무조건 mm로 가정하지 않고 실제 장치의 depth_scale을 곱합니다.
        write_png(pending / 'color.png', capture.color)
        write_array(pending / 'depth_raw.npy', capture.original_depth)
        write_array(pending / 'depth_meters.npy', capture.original_depth.astype(np.float32) * scale)
        write_array(pending / 'depth_aligned_raw.npy', capture.aligned_depth)
        write_array(pending / 'depth_aligned_meters.npy', capture.aligned_depth.astype(np.float32) * scale)
        # [추가] 깊이 0은 실제 거리 0m가 아닌 미측정이므로 학습에서 제외할 수 있게 마스크를 저장합니다.
        # 정렬의 빈 영역을 임의로 보간하지 않아 존재하지 않는 표면/깊이를 만들지 않습니다.
        write_png(pending / 'valid_mask.png', np.uint8(capture.aligned_depth > 0) * 255)
        vis = depth_visualization(capture.aligned_depth, scale)
        write_png(pending / 'depth_aligned_vis.png', vis)
        write_png(pending / 'preview_rgb_depth.png', np.hstack((capture.color, vis)))
        # [추가] 복사·전송 후 손상이나 변경을 확인할 수 있도록 데이터 파일별 SHA-256을 기록합니다.
        # metadata.json 자신은 자기 해시를 포함할 수 없어 이 해시 목록에서는 제외됩니다.
        checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in pending.iterdir() if path.is_file()}
        # [추가] 샘플 폴더 하나만 옮겨도 보정값·시각을 잃지 않도록 metadata.json에 모두 포함합니다.
        # 배열별 자료형·단위·좌표계와 정렬 방식을 명시하여 나중에 학습·분석에서 혼동하지 않게 합니다.
        metadata = {
            'schema': 'realsense-capture-v2-sample', 'capture_id': stem, 'saved_at': now_iso(),
            'received_at_host': capture.received_at, 'received_unix_ns': capture.received_unix_ns,
            'calibration': calibration, 'aligned_depth_intrinsics': capture.aligned_depth_intrinsics,
            'depth_frame': capture.depth_frame, 'color_frame': capture.color_frame,
            'temporal_pairing': capture.pairing,
            # [주의] SDK 정렬은 원본 Z16 샘플을 RGB 위치에 대응시킵니다.
            # 값은 원래 깊이 센서의 Z 샘플을 유지하며 물체까지의 유클리드 거리로 변환하는 처리가 아닙니다.
            'alignment': {'method': 'RealSense SDK rs.align(rs.stream.color)',
                          'target_grid': 'original color pixels', 'hole_filling': False,
                          'raw_depth_preserved': True,
                          'depth_semantics': 'SDK-aligned Z16 retains source depth samples, not Euclidean range.'},
            'arrays': {'color.png': 'original BGR8 sensor buffer saved losslessly as PNG',
                       'depth_raw.npy': 'uint16 original depth grid; meters = value * depth_scale_m',
                       'depth_meters.npy': 'float32 meters, original depth grid',
                       'depth_aligned_raw.npy': 'uint16 SDK-aligned depth, RGB grid',
                       'depth_aligned_meters.npy': 'float32 meters, RGB grid',
                       'valid_mask.png': 'uint8 RGB grid: 255 valid depth, 0 missing'},
            # [추가] 미측정이 많은 샘플을 선별할 수 있게 유효 깊이 비율을 기록합니다. 인식 정확도는 아닙니다.
            'original_valid_fraction': float(np.mean(capture.original_depth > 0)),
            'aligned_valid_fraction': float(np.mean(capture.aligned_depth > 0)),
            'files_sha256': checksums,
        }
        # [추가] 데이터·메타데이터를 모두 기록한 뒤 최종 폴더명으로 바꿔 정상 완료를 표시합니다.
        write_json(pending / 'metadata.json', metadata)
        pending.rename(final)
    # [추가] 실패한 임시 폴더는 원인 분석·가능한 파일 복구를 위해 남기며 완성 자료로 취급하지 않습니다.
    except Exception:
        print(f'저장 실패. 완성되지 않은 파일은 이 폴더에 남아 있습니다: {pending}', file=sys.stderr)
        raise
    return final


# [변경] 기존 640×480/30fps 고정 요청을 장치 선택·지원 모드 검사·연결 속도별 FPS 선택으로 확장했습니다.
# USB 2의 대역폭 부담과 지원하지 않는 모드 요청에 따른 시작 실패를 줄이기 위한 처리입니다.
def start_camera(args):
    context = rs.context()
    devices = context.query_devices()
    if not len(devices):
        raise RuntimeError('연결된 RealSense 카메라가 없습니다.')
    # [추가] 다른 카메라와 그 보정값을 실수로 선택하지 않도록 여러 장치 연결 시 일련번호를 요구합니다.
    if not args.serial and len(devices) != 1:
        raise RuntimeError('카메라가 여러 대입니다. --serial로 촬영할 장치를 지정하세요.')
    device = next((d for d in devices if not args.serial or
                   d.get_info(rs.camera_info.serial_number) == args.serial), None)
    if device is None:
        raise RuntimeError('지정한 일련번호의 카메라가 없습니다.')
    info = device_info(device)
    serial = info['serial_number']
    # [변경] 기본값은 USB 2에서 15fps, 그 외에서 30fps를 먼저 시도하고 시작 실패 시 낮은 FPS로 재시도합니다.
    # 640×480 해상도를 유지하며 전송 부담을 줄이고, --fps를 지정한 경우에는 그 값만 시도합니다.
    rates = [args.fps] if args.fps else ([15, 6] if str(info.get('usb_type_descriptor', '')).startswith('2') else [30, 15, 6])
    # [추가] 카메라가 보고한 640×480 Z16/BGR8 공통 FPS만 후보로 삼아 미지원 설정 요청을 피합니다.
    supported = {'depth': set(), 'color': set()}
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            if video.width() == 640 and video.height() == 480:
                if profile.stream_type() == rs.stream.depth and profile.format() == rs.format.z16:
                    supported['depth'].add(profile.fps())
                elif profile.stream_type() == rs.stream.color and profile.format() == rs.format.bgr8:
                    supported['color'].add(profile.fps())
    rates = [fps for fps in rates if fps in supported['depth'] and fps in supported['color']]
    if not rates:
        raise RuntimeError(f'요청한 640x480 RGB/Depth FPS가 지원되지 않습니다: {supported}')
    failures = []
    # [추가] FPS 재시도마다 파이프라인을 새로 만들고 실패한 기존 스트림을 종료해 점유 충돌을 줄입니다.
    # 실패 내역은 반환하여 실제 설정이 선택된 과정을 세션 기록에서 확인할 수 있게 합니다.
    for fps in rates:
        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        started = False
        try:
            print(f'카메라 시작: USB {info.get("usb_type_descriptor", "?")} | 640x480 / {fps}fps')
            profile = pipeline.start(config)
            started = True
            # [추가] start 성공만으로 판단하지 않고 RGB·Depth가 함께 수신되는지 제한 시간 내 검사합니다.
            # [주의] 첫 수신 검사가 지속 FPS나 장기간 안정성까지 보증하지는 않습니다.
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                ok, frames = pipeline.try_wait_for_frames(500)
                if ok and frames.get_depth_frame() and frames.get_color_frame():
                    return pipeline, profile, fps, failures
            raise RuntimeError('8초 동안 RGB/Depth 프레임을 함께 수신하지 못했습니다.')
        except RuntimeError as exc:
            failures.append({'fps': fps, 'error': str(exc)})
            if started:
                pipeline.stop()
            print(f'시작 실패: {fps}fps / {exc}')
    raise RuntimeError(f'카메라 스트림을 시작하지 못했습니다: {failures}')


# [변경] main으로 실행부를 묶어 import 기반 테스트·리뷰 도구가 카메라를 자동으로 열지 않게 합니다.
def main():
    # [추가] 저장 위치·FPS·장치·시각 차이 기준을 실행 옵션으로 제공해 코드 직접 수정의 필요를 줄입니다.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=HERE / 'captures_v2')
    parser.add_argument('--fps', type=int, default=None, help='명시하면 해당 FPS만 사용')
    parser.add_argument('--serial', default='', help='여러 카메라 중 사용할 일련번호')
    parser.add_argument('--max-skew-ms', type=float, default=None, help='저장 허용 시각 차이, 기본값은 프레임 주기의 75%%')
    args = parser.parse_args()
    # [추가] 음수 FPS와 NaN 같은 잘못된 인자를 카메라·파일 작업 전에 거부합니다.
    if args.fps is not None and args.fps <= 0:
        parser.error('--fps는 양수여야 합니다.')
    if args.max_skew_ms is not None and (not np.isfinite(args.max_skew_ms) or args.max_skew_ms <= 0):
        parser.error('--max-skew-ms는 유한한 양수여야 합니다.')
    # [추가] 화면 표시·색상 변환에 OpenCV가 과도한 CPU 스레드를 만들지 않도록 제한합니다.
    cv2.setNumThreads(2)
    pipeline = None
    try:
        pipeline, profile, fps, failures = start_camera(args)
        device = profile.get_device()
        # [추가] 요청값이 아닌 실제 시작된 프로필에서 보정값을 읽어 FPS 재시도 결과도 정확히 남깁니다.
        calibration = calibration_info(device, profile.get_stream(rs.stream.depth),
                                       profile.get_stream(rs.stream.color), device.first_depth_sensor().get_depth_scale())
        # [추가] 기본 시간 차이 허용값은 프레임 주기(1000/fps ms)의 75%로 정한 수집 품질 필터입니다.
        # 30fps는 25ms, 15fps는 50ms이며 더 엄격한 값은 --max-skew-ms로 지정할 수 있습니다.
        # [주의] 이는 동기화의 물리적 보증값이 아니라 명시적인 저장 허용 기준입니다.
        max_skew_ms = args.max_skew_ms if args.max_skew_ms is not None else 750 / fps
        # [추가] 날짜·시각과 무작위 식별자를 결합해 재실행·동시 실행에서 기존 세션을 덮어쓰지 않습니다.
        name = 'session_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        session = args.output.resolve() / name
        session.mkdir(parents=True, exist_ok=False)
        # [추가] 첫 저장 전부터 세션 보정값·실행 환경·시작 실패 내역을 남겨 수집 자료의 출처를 보존합니다.
        # 스크립트 해시도 기록해 어느 코드로 수집했는지 나중에 비교할 수 있게 합니다.
        write_json(session / 'calibration.json', calibration)
        write_json(session / 'session.json', {'schema': 'realsense-capture-v2-session', 'started_at': now_iso(),
                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'calibration': calibration, 'selected_fps': fps, 'max_skew_ms': max_skew_ms,
                   'startup_failures': failures, 'python_version': sys.version,
                   'array_dimensions': 'NPY: height, width; color PNG: height, width, 3'})
        print(f'저장 폴더: {session}\nS: 현재 RGB-D 저장 / Q, Esc: 종료')
        # [핵심 추가] RGB 픽셀 좌표를 목표로 하는 SDK 정렬기를 한 번 생성하고 프레임마다 재사용합니다.
        # RGB 자체를 변형하는 대신 Depth 샘플이 놓이는 좌표를 RGB 기준으로 옮깁니다.
        aligner = rs.align(rs.stream.color)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 530)
        latest = None
        count = received = 0
        last_frame_time = time.monotonic()
        # [추가] 요청 FPS와 실제 수신 FPS를 구별해 표시할 수 있도록 최근 수신 시각을 제한된 크기로 보관합니다.
        arrivals = deque(maxlen=300)
        while True:
            # [변경] 프레임을 무기한 기다리지 않아 수신 지연 중에도 종료 키·창 이벤트를 처리할 수 있게 합니다.
            ok, frames = pipeline.try_wait_for_frames(200)
            if ok:
                capture = capture_from_frames(frames, aligner, calibration, max_skew_ms)
                if capture is not None:
                    latest = capture
                    received += 1
                    last_frame_time = time.monotonic()
                    arrivals.append(last_frame_time)
            now = time.monotonic()
            # [추가] 수신이 끊겼는데 마지막 영상을 실시간처럼 계속 보여주지 않도록 중단 기준을 둡니다.
            if now - last_frame_time > 8:
                raise RuntimeError('카메라 프레임 수신이 8초 이상 중단됐습니다. USB 연결과 다른 카메라 앱을 확인하세요.')
            # [추가] 아직 RGB-D가 준비되지 않았을 때는 대기 화면을 표시하고 종료 키를 처리합니다.
            if latest is None:
                splash = np.full((530, 1280, 3), 25, np.uint8)
                cv2.putText(splash, 'Waiting for paired RGB/Depth frames... Q: quit', (25, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (235, 235, 235), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, splash)
                if cv2.waitKey(1) & 255 in (27, ord('q')):
                    break
                continue
            # [추가] PC 시계 변경과 무관한 시각으로 프레임 지연과 최근 5초 실제 수신 빈도를 계산합니다.
            age = now - latest.received_monotonic
            while len(arrivals) > 1 and arrivals[0] < now - 5:
                arrivals.popleft()
            fps_now = (len(arrivals) - 1) / max(now - arrivals[0], .001) if len(arrivals) > 1 else 0
            # [추가] 시각 대응·최신 프레임 여부·초기 10개 프레임 수신을 모두 만족할 때만 저장을 허용합니다.
            # [주의] 최소 준비 대기이며 자동 노출이나 깊이 정확도가 충분히 안정됐다는 보증은 아닙니다.
            ready = latest.pairing['within_threshold'] and age < max(.5, 3/fps) and received >= 10
            # [변경] 오른쪽 화면에는 정렬 전 Depth가 아닌 RGB에 정렬된 Depth를 보여줍니다.
            # USB 속도·실제 FPS·시각 차이·저장 가능 상태도 표시해 수집 중 문제를 알아차릴 수 있게 합니다.
            preview = np.hstack((latest.color, depth_visualization(latest.aligned_depth, calibration['depth_scale_m'])))
            top = np.full((50, 1280, 3), 25, np.uint8)
            delta = latest.pairing['timestamp_delta_ms']
            skew_text = f'{delta:.1f}ms' if delta is not None else 'not comparable'
            status = 'READY' if ready else 'WAIT / SAVE BLOCKED'
            cv2.putText(top, f'RGB | ALIGNED DEPTH | USB {calibration["device"].get("usb_type_descriptor", "?")} | {fps_now:.1f}fps',
                        (12, 21), cv2.FONT_HERSHEY_SIMPLEX, .57, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(top, f'{status} | time gap {skew_text} | saved {count} | S: save / Q: quit',
                        (12, 43), cv2.FONT_HERSHEY_SIMPLEX, .53, (110, 230, 140) if ready else (80, 150, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, np.vstack((top, preview)))
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q')):
                break
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            # [변경] S를 누르면 화면에 대응하는 최신 Capture 전체를 한 번에 저장합니다.
            # 표시용 RGB와 저장용 Depth를 서로 다른 수신 호출에서 가져오는 방식을 피합니다.
            if key == ord('s'):
                if not ready:
                    print('저장 보류: 준비 중이거나 RGB/Depth 시각 차이·프레임 지연이 큽니다.')
                    continue
                try:
                    destination = save_capture(session, count + 1, latest, calibration)
                # [추가] 품질 기준 미달은 이번 샘플 저장만 보류하고 다음 프레임을 계속 받을 수 있게 합니다.
                except ValueError as exc:
                    print(f'저장 보류: {exc}')
                    continue
                count += 1
                print(f'[저장 {count}] {destination} | 시각 차이 {skew_text}')
        print(f'수집 종료: {count}쌍 저장 / {session}')
    # [변경] 정상 종료뿐 아니라 오류 발생 시에도 카메라와 창을 정리해 다음 실행의 점유 문제를 줄입니다.
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()


# [추가] 직접 실행할 때만 수집을 시작하고 오류는 콘솔 메시지와 종료 코드로 전달합니다.
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\n수집 오류: {exc}', file=sys.stderr)
        # [추가] 이 PC의 제한된 자동 실행 환경에서 재현된 접근 오류에는 실행 환경을 안내합니다.
        # 카메라 미연결이나 드라이버 문제로 오해해 불필요한 재설치를 하지 않게 돕는 메시지입니다.
        if 'Mutex_TotalFailure' in str(exc) or 'CreateNamedMutex' in str(exc):
            print('일반 Windows 터미널에서 실행하세요. 제한된 자동 실행 환경에서는 카메라 접근이 차단될 수 있습니다.', file=sys.stderr)
        raise SystemExit(1)