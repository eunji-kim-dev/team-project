"""D435 센서 연결·보정·동일 프레임 RGB-D 취득. 독립 모듈이며 이전 실행기를 import하지 않는다."""
from dataclasses import dataclass
from datetime import datetime
import importlib.metadata
import time
import numpy as np
import pyrealsense2 as rs

# [v3.5] 필요한 now_iso 기능만 현재 모듈에서 직접 사용.
def now_iso():
    return datetime.now().astimezone().isoformat()


# [v3.5] 필요한 intrinsics_info 기능만 현재 모듈에서 직접 사용.
def intrinsics_info(profile):
    video = profile.as_video_stream_profile()
    intr = video.get_intrinsics()
    return {'width': intr.width, 'height': intr.height, 'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy, 'distortion_model': str(intr.model).split('.')[-1], 'coeffs': list(intr.coeffs), 'fps': video.fps(), 'format': str(video.format())}


# [v3.5] 필요한 extrinsics_info 기능만 현재 모듈에서 직접 사용.
def extrinsics_info(source, target):
    extr = source.get_extrinsics_to(target)
    return {'rotation': list(extr.rotation), 'rotation_storage': 'column-major', 'translation_m': list(extr.translation), 'mapping': 'P_target = R * P_source + t'}


# [v3.5] 필요한 device_info 기능만 현재 모듈에서 직접 사용.
def device_info(device):
    result = {}
    for name in ('name', 'serial_number', 'firmware_version', 'usb_type_descriptor', 'product_id'):
        field = getattr(rs.camera_info, name)
        if device.supports(field):
            result[name] = device.get_info(field)
    return result


# [v3.5] 필요한 frame_info 기능만 현재 모듈에서 직접 사용.
def frame_info(frame):
    info = {'frame_number': int(frame.get_frame_number()), 'timestamp_ms': float(frame.get_timestamp()), 'timestamp_domain': str(frame.get_frame_timestamp_domain()), 'metadata': {}}
    for name in ('actual_exposure', 'gain_level', 'actual_fps', 'frame_counter', 'frame_timestamp', 'sensor_timestamp', 'backend_timestamp', 'time_of_arrival'):
        field = getattr(rs.frame_metadata_value, name, None)
        if field is not None and frame.supports_frame_metadata(field):
            try:
                info['metadata'][name] = int(frame.get_frame_metadata(field))
            except RuntimeError:
                pass
    return info


# [v3.5] 필요한 pairing_info 기능만 현재 모듈에서 직접 사용.
def pairing_info(depth_info, color_info, max_skew_ms):
    same_domain = depth_info['timestamp_domain'] == color_info['timestamp_domain']
    delta = abs(depth_info['timestamp_ms'] - color_info['timestamp_ms']) if same_domain else None
    comparable = delta is not None and bool(np.isfinite(delta))
    return {'source': 'same RealSense SDK frameset', 'timestamp_domains_match': same_domain, 'timestamp_delta_ms': delta if comparable else None, 'max_skew_ms': max_skew_ms, 'within_threshold': comparable and delta <= max_skew_ms, 'note': 'Timestamp screening is not proof of identical exposure times or hardware synchronization.'}


# [v3.5] 필요한 calibration_info 기능만 현재 모듈에서 직접 사용.
def calibration_info(device, depth_profile, color_profile, scale):
    # Jetson에서 소스로 빌드한 SDK도 pip 메타데이터 없이 사용할 수 있다.
    try:version=importlib.metadata.version('pyrealsense2')
    except importlib.metadata.PackageNotFoundError:version=getattr(rs,'__version__','system build')
    return {'schema': 'realsense-capture-v2-calibration', 'recorded_at': now_iso(), 'sdk_version': version, 'device': device_info(device), 'depth_scale_m': float(scale), 'depth_intrinsics': intrinsics_info(depth_profile), 'color_intrinsics': intrinsics_info(color_profile), 'depth_to_color': extrinsics_info(depth_profile, color_profile), 'color_to_depth': extrinsics_info(color_profile, depth_profile)}


# [v3.5] 필요한 Capture 기능만 현재 모듈에서 직접 사용.
@dataclass
class Capture:
    color: np.ndarray
    original_depth: np.ndarray
    aligned_depth: np.ndarray
    depth_frame: dict
    color_frame: dict
    aligned_depth_intrinsics: dict
    pairing: dict
    received_monotonic: float
    received_at: str
    received_unix_ns: int


# [v3.5] 필요한 capture_from_frames 기능만 현재 모듈에서 직접 사용.
def capture_from_frames(frames, aligner, calibration, max_skew_ms):
    depth_frame, color_frame = (frames.get_depth_frame(), frames.get_color_frame())
    if not depth_frame or not color_frame:
        return None
    received_at, received_ns, received_mono = (now_iso(), time.time_ns(), time.monotonic())
    raw = np.asanyarray(depth_frame.get_data()).copy()
    color = np.asanyarray(color_frame.get_data()).copy()
    depth_details, color_details = (frame_info(depth_frame), frame_info(color_frame))
    aligned_frames = aligner.process(frames)
    aligned_frame = aligned_frames.get_depth_frame()
    if not aligned_frame:
        return None
    aligned = np.asanyarray(aligned_frame.get_data()).copy()
    if raw.dtype != np.uint16 or aligned.dtype != np.uint16:
        raise RuntimeError('Depth는 Z16(uint16) 형식이어야 합니다.')
    if color.dtype != np.uint8 or color.shape != (480, 640, 3) or raw.shape != (480, 640):
        raise RuntimeError('요청한 원본 640x480 스트림과 실제 배열 크기가 다릅니다.')
    if aligned.shape != color.shape[:2]:
        raise RuntimeError('정렬 Depth와 RGB 크기가 다릅니다.')
    aligned_intr = intrinsics_info(aligned_frame.profile)
    for key in ('width', 'height', 'fx', 'fy', 'ppx', 'ppy'):
        if not np.isclose(aligned_intr[key], calibration['color_intrinsics'][key]):
            raise RuntimeError('정렬 Depth의 내부 파라미터가 RGB 좌표계와 일치하지 않습니다.')
    if not np.array_equal(np.asanyarray(aligned_frames.get_color_frame().get_data()), color):
        raise RuntimeError('정렬 과정에서 원본 RGB가 변경됐습니다.')
    return Capture(color, raw, aligned, depth_details, color_details, aligned_intr, pairing_info(depth_details, color_details, max_skew_ms), received_mono, received_at, received_ns)


# [v3.5] 필요한 start_camera 기능만 현재 모듈에서 직접 사용.
def start_camera(args, target_fps=None):
    context = rs.context()
    devices = context.query_devices()
    if not len(devices):
        raise RuntimeError('연결된 RealSense 카메라가 없습니다.')
    if not args.serial and len(devices) != 1:
        raise RuntimeError('카메라가 여러 대입니다. --serial로 촬영할 장치를 지정하세요.')
    device = next((d for d in devices if not args.serial or d.get_info(rs.camera_info.serial_number) == args.serial), None)
    if device is None:
        raise RuntimeError('지정한 일련번호의 카메라가 없습니다.')
    info = device_info(device)
    serial = info['serial_number']
    fallback = [15, 6] if str(info.get('usb_type_descriptor', '')).startswith('2') else [30, 15, 6]
    # [v3.5] Device/profile discovery once; target and USB fallback share one loop.
    rates = [args.fps] if args.fps else list(dict.fromkeys(([target_fps] if target_fps else []) + fallback))
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
    for fps in rates:
        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        started = False
        try:
            print(f"카메라 시작: USB {info.get('usb_type_descriptor', '?')} | 640x480 / {fps}fps")
            profile = pipeline.start(config)
            started = True
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                ok, frames = pipeline.try_wait_for_frames(500)
                if ok and frames.get_depth_frame() and frames.get_color_frame():
                    return (pipeline, profile, fps, failures)
            raise RuntimeError('8초 동안 RGB/Depth 프레임을 함께 수신하지 못했습니다.')
        except RuntimeError as exc:
            failures.append({'fps': fps, 'error': str(exc)})
            if started:
                pipeline.stop()
            print(f'시작 실패: {fps}fps / {exc}')
    raise RuntimeError(f'카메라 스트림을 시작하지 못했습니다: {failures}')
