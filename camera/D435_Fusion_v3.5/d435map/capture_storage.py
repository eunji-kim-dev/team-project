"""촬영 폴더와 원본 RGB-D 저장. 이전 데이터의 파일명·수치·JSON 스키마 호환 유지."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import re
import sys
import cv2
import numpy as np

# [v3.5] 필요한 write_json 기능만 현재 모듈에서 직접 사용.
def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


# [v3.5] 필요한 write_array 기능만 현재 모듈에서 직접 사용.
def write_array(path, array):
    with Path(path).open('xb') as stream:
        np.save(stream, array, allow_pickle=False)


# [v3.5] 필요한 write_png 기능만 현재 모듈에서 직접 사용.
def write_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'PNG 인코딩 실패: {path}')
    with Path(path).open('xb') as stream:
        stream.write(encoded.tobytes())


# [v3.5] 필요한 depth_visualization 기능만 현재 모듈에서 직접 사용.
def depth_visualization(raw, scale):
    meters = raw.astype(np.float32) * np.float32(scale)
    normalized = np.clip((meters - 0.15) / (6.0 - 0.15), 0, 1)
    image = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_RAINBOW)
    image[raw == 0] = 0
    return image


# [v3.5] 필요한 output_directory 기능만 현재 모듈에서 직접 사용.
def output_directory(path=None):
    root = (Path(path) if path is not None else Path.cwd()).resolve()
    if not root.is_dir():
        raise ValueError(f'저장 기준 경로는 이미 존재하는 폴더여야 합니다: {root}')
    return root


# [v3.5] 필요한 next_capture_number 기능만 현재 모듈에서 직접 사용.
def next_capture_number(root):
    highest = 0
    pattern = re.compile('^\\.?(\\d{4,})_\\d{8}_\\d{6}(?:\\.pending)?$')
    for entry in Path(root).iterdir():
        match = pattern.fullmatch(entry.name)
        if entry.is_dir() and match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


# [v3.5] 필요한 reserve_capture_directory 기능만 현재 모듈에서 직접 사용.
def reserve_capture_directory(root):
    root = output_directory(root)
    number = next_capture_number(root)
    while True:
        saved_at = datetime.now().astimezone()
        stem = f'{number:04d}_{saved_at:%Y%m%d_%H%M%S}'
        final = root / stem
        pending = root / f'.{stem}.pending'
        if final.exists() or pending.exists():
            number += 1
            continue
        try:
            pending.mkdir(exist_ok=False)
        except FileExistsError:
            number += 1
            continue
        return (number, final, pending, saved_at)


# [v3.5] 필요한 save_capture 기능만 현재 모듈에서 직접 사용.
def save_capture(root, capture, calibration, session_info):
    if not capture.pairing['within_threshold']:
        raise ValueError('RGB/Depth 시각 차이가 허용 범위를 벗어나 저장하지 않습니다.')
    if not (capture.original_depth > 0).any() or not (capture.aligned_depth > 0).any():
        raise ValueError('유효 깊이가 없어 저장하지 않습니다.')
    number, final, pending, saved_at = reserve_capture_directory(root)
    prefix = f'{number:04d}_'
    scale = np.float32(calibration['depth_scale_m'])
    try:
        write_png(pending / f'{prefix}color.png', capture.color)
        write_array(pending / f'{prefix}depth_raw.npy', capture.original_depth)
        write_array(pending / f'{prefix}depth_meters.npy', capture.original_depth.astype(np.float32) * scale)
        write_array(pending / f'{prefix}depth_aligned_raw.npy', capture.aligned_depth)
        write_array(pending / f'{prefix}depth_aligned_meters.npy', capture.aligned_depth.astype(np.float32) * scale)
        write_png(pending / f'{prefix}valid_mask.png', np.uint8(capture.aligned_depth > 0) * 255)
        vis = depth_visualization(capture.aligned_depth, scale)
        write_png(pending / f'{prefix}depth_aligned_vis.png', vis)
        write_png(pending / f'{prefix}preview_rgb_depth.png', np.hstack((capture.color, vis)))
        write_json(pending / f'{prefix}calibration.json', calibration)
        write_json(pending / f'{prefix}session.json', session_info)
        checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in pending.iterdir() if path.is_file()}
        metadata = {'schema': 'realsense-capture-v2.2-sample', 'capture_id': final.name, 'capture_sequence': number, 'filename_prefix': prefix, 'session_id': session_info['session_id'], 'saved_at': saved_at.isoformat(), 'received_at_host': capture.received_at, 'received_unix_ns': capture.received_unix_ns, 'calibration': calibration, 'aligned_depth_intrinsics': capture.aligned_depth_intrinsics, 'depth_frame': capture.depth_frame, 'color_frame': capture.color_frame, 'temporal_pairing': capture.pairing, 'alignment': {'method': 'RealSense SDK rs.align(rs.stream.color)', 'target_grid': 'original color pixels', 'hole_filling': False, 'raw_depth_preserved': True, 'depth_semantics': 'SDK-aligned Z16 retains source depth samples, not Euclidean range.'}, 'arrays': {f'{prefix}color.png': 'original BGR8 sensor buffer saved losslessly as PNG', f'{prefix}depth_raw.npy': 'uint16 original depth grid; meters = value * depth_scale_m', f'{prefix}depth_meters.npy': 'float32 meters, original depth grid', f'{prefix}depth_aligned_raw.npy': 'uint16 SDK-aligned depth, RGB grid', f'{prefix}depth_aligned_meters.npy': 'float32 meters, RGB grid', f'{prefix}valid_mask.png': 'uint8 RGB grid: 255 valid depth, 0 missing'}, 'original_valid_fraction': float(np.mean(capture.original_depth > 0)), 'aligned_valid_fraction': float(np.mean(capture.aligned_depth > 0)), 'files_sha256': checksums}
        write_json(pending / f'{prefix}metadata.json', metadata)
        pending.rename(final)
    except Exception:
        print(f'저장 실패. 완성되지 않은 파일은 이 폴더에 남아 있습니다: {pending}', file=sys.stderr)
        raise
    return final
