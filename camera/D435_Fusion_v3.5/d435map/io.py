"""같은 Capture의 원본·계산 결과를 직접 저장. 이전 실행 파일 의존 없음."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
import numpy as np
import cv2
from .engine import Packet
from .capture_storage import save_capture,write_json,write_array,write_png
from .geometry import voxel_indices
from .display import classification_color, cost_color, Dashboard, semantic_overlay, valid_semantic
from .costmap import obstacle_risk,display_cost

ROOT=Path(__file__).resolve().parents[1]

def read_png(path):
    image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f"PNG decode failed: {path}")
    return image

def load_capture(directory, clock=None):
    root=Path(directory)
    files=sorted(root.glob("*_metadata.json"))
    if not files:
        files=sorted(root.glob("metadata.json"))
    if len(files)!=1: raise ValueError(f"Expected one metadata.json: {root}")
    meta=json.loads(files[0].read_text(encoding="utf-8-sig"))
    # [v3.5 추가] 좌우 반전 깊이에 원본 보정값을 적용하는 잘못된 재현을 거부한다.
    if any(k in meta for k in ("augmentation","horizontal_flip","flipped")):
        raise ValueError("Augmented capture needs explicit calibrated reflection transform")
    cal=meta["calibration"]
    prefix=files[0].name.removesuffix("metadata.json")
    # [v3.5 추가] 팀에서 폴더/파일만 재번호화한 자료도 실제 파일명 접두어로 읽는다. 원본 JSON은 수정하지 않는다.
    def array(name): return np.load(root/(prefix+name),allow_pickle=False)
    t=time.monotonic() if clock is None else float(clock)
    cap=SimpleNamespace(color=read_png(root/(prefix+"color.png")),
        original_depth=array("depth_raw.npy"),aligned_depth=array("depth_aligned_raw.npy"),
        depth_frame=meta["depth_frame"],color_frame=meta["color_frame"],
        aligned_depth_intrinsics=meta["aligned_depth_intrinsics"],
        pairing=meta["temporal_pairing"],received_monotonic=t,
        received_at=meta["received_at_host"],received_unix_ns=meta["received_unix_ns"])
    return Packet(cap,cal,root.name,True)

def write_ply(path,points,rgb,classes):
    # [v3.5 추가] PLY 색은 RGB이며 좌표는 미터/바닥 기준. class는 기하+의미 융합 판정이며 SCNN 정답이 아니다.
    dtype=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("red","u1"),("green","u1"),("blue","u1"),("class","u1")])
    rows=np.empty(len(points),dtype=dtype)
    for i,key in enumerate(("x","y","z")): rows[key]=points[:,i]
    for i,key in enumerate(("red","green","blue")): rows[key]=rgb[:,i]
    rows["class"]=classes
    header=("ply\nformat binary_little_endian 1.0\ncomment units meters; X forward Y left Z up\n"
            f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar class\nend_header\n")
    with Path(path).open("xb") as f:
        f.write(header.encode("ascii")); f.write(rows.tobytes())

def save_extras(destination,prefix,s,cfg):
    # [Jetson 경량화] 지연한 RGB 점군/분리 결과를 같은 저장 프레임에서 완성한다.
    s.ensure_color()
    # [v3.5] 입력칸 변경 전후가 섞이지 않도록 해당 프레임의 설정으로 저장.
    cfg=getattr(s,"applied_config",None) or cfg
    dest=Path(destination)
    # 기존 11파일의 성공 상태와 추가 파일의 성공 상태를 구별한다. 부분 저장도 감지 가능.
    pending=dest/(prefix+"v3.5_pending.json")
    write_json(pending,{"frame_id":s.stats["frame_id"],"state":"writing additional outputs"})
    def arr(name,value): write_array(dest/(prefix+name),value)
    def png(name,value): write_png(dest/(prefix+name),value)
    ids=voxel_indices(s.points,cfg["processing"]["cloud_save_voxel_m"])
    write_ply(dest/"pointcloud.ply",s.points[ids],s.rgb[ids],s.point_classes[ids])
    arr("pointcloud_color_valid.npy",s.color_valid[ids])
    arr("costmap.npy",s.cost)  # 호환용 보수적 비용: 미확인=255를 유지.
    risk=s.risk_cost if s.risk_cost is not None else obstacle_risk(s.physical,cfg["grid"]["resolution_m"],cfg["robot"])
    arr("costmap_risk.npy",risk)
    arr("costmap_unknown.npy",s.physical==-1)
    if s.rear_unknown is not None:arr("costmap_rear_unknown.npy",s.rear_unknown)
    arr("costmap_display.npy",display_cost(s.physical,risk))
    arr("occupancy.npy",s.physical)
    # [v3.5 추가] 비용 값 255와 시야 밖을 구분하며 기존 uint8 비용 형식은 유지한다.
    in_view=s.in_view if s.in_view is not None else np.ones_like(s.cost,bool)
    arr("costmap_in_view.npy",in_view)
    state=s.cost.astype(np.uint16);state[~in_view]=256
    arr("costmap_state.npy",state)
    arr("volume_clear.npy",s.volume_clear)
    arr("floor_evidence.npy",s.free_evidence)
    if s.semantic_free is not None:arr("semantic_free_inferred.npy",s.semantic_free)
    if s.semantic_free_continuity is not None:arr("semantic_free_continuity.npy",s.semantic_free_continuity)
    arr("depth_processing_meters.npy",s.filtered_depth)
    png("processing_valid_native.png",s.measured_valid.astype(np.uint8)*255)
    native=np.full(s.filtered_depth.shape,255,np.uint8)
    native.ravel()[s.source_indices]=s.point_classes
    png("geometry_labels_native.png",native)
    # [v3.5 추가] 보류된 장애물 후보를 원본 Depth 좌표로 저장해 원인 추적을 돕는다.
    if s.rejected_obstacles_native is not None:
        png("obstacle_rejected_native.png",s.rejected_obstacles_native)
    # [v3.5] RGB 바닥 예측과 Depth 높이가 충돌한 native 픽셀을 별도 보존.
    if s.semantic_conflict_native is not None:
        png("semantic_depth_conflict_native.png",s.semantic_conflict_native)
    png("geometry_color_native.png",classification_color(native))
    png("geometry_labels_color_grid.png",s.color_classes)
    png("costmap_conservative.png",cost_color(s.cost,in_view)[::-1,::-1])
    png("costmap.png",cost_color(display_cost(s.physical,risk),in_view,s.physical==-1)[::-1,::-1])
    if s.local_review_native is not None:png("local_floor_held_native.png",s.local_review_native)
    if s.temporal_review_native is not None:png("temporal_held_native.png",s.temporal_review_native)
    # [v3.5 추가] 수치/평면 PNG 형식은 유지하고 낮은 원근 시점과 진단을 별도 저장.
    from .map_view import ground_view
    png("costmap_ground_view.png",ground_view(s,s.cost,cost_color(s.cost,in_view),cfg,900,500,show_obstacles=False))
    from .obstacle_view import obstacle_view
    # [v3.5] Low Angle/High Angle 점군을 같은 스냅샷/설정으로 저장. 일반 obstacle_cloud.png도 유지.
    from .cloud_colors import prepare_obstacle_cloud
    prepared_cloud=prepare_obstacle_cloud(s,cfg)
    low_angle_picture=obstacle_view(s,cfg,900,500,prepared=prepared_cloud)
    png("obstacle_cloud.png",low_angle_picture)
    png("obstacle_cloud_low_angle.png",low_angle_picture)
    png("obstacle_cloud_high_angle.png",obstacle_view(s,cfg,900,500,view_name="high_angle",prepared=prepared_cloud))
    write_json(dest/(prefix+"costmap_diagnostics.json"),s.stats.get("map_diagnostics",{}))
    # [v3.5 추가] 동일 프레임의 실제 모델 예측만 추가 저장. 없으면 거짓 마스크 파일을 만들지 않는다.
    if valid_semantic(s):
        png("scnn_labels.png",s.semantic.labels)
        if s.semantic.confidence is not None:arr("scnn_confidence.npy",s.semantic.confidence)
        if s.semantic_native is not None:
            semantic_native=np.full(s.filtered_depth.shape,255,np.uint8)
            semantic_native.ravel()[s.source_indices]=s.semantic_native
            png("scnn_labels_native.png",semantic_native)
        png("scnn_label_color.png",classification_color(s.semantic.labels))
        png("scnn_overlay.png",semantic_overlay(s,cfg["semantics"]["overlay_alpha"]))
    png("dashboard.png",Dashboard(cfg).render(s))
    record={"schema":"D435-depth-local-map-v3.5","extension_version":"3.4","frame":s.stats,
        "depth_frame":s.packet.capture.depth_frame,"color_frame":s.packet.capture.color_frame,
        "received_unix_ns":s.packet.capture.received_unix_ns,
        "map":s.grid_metadata,"floor":s.floor.record(),
        "candidate_review":s.stats.get("map_diagnostics",{}).get("candidate_review",{}),
        "costmap_layers":{"occupancy":"int8 -1 unknown,0 observed free,100 obstacle",
            "costmap":"uint8 conservative navigation cost, unknown remains255",
            "costmap_risk":"uint8 0..254 distance risk independent of observation; zero is NOT free evidence",
            "costmap_unknown":"bool observation unknown, retained even under colored risk",
            "costmap_rear_unknown":"bool camera ray blocked: risk removed, physical free becomes unknown; measured obstacles preserved",
            "costmap_display":"risk prioritized, unknown without risk255; display only",
            "planning_rule":"never infer traversability from risk alone; occupancy unknown blocks / requires a separate policy",
            "hatch":"unknown+risk; outside FOV stays black"},
        "T_floor_from_native_depth_optical":s.transform.tolist(),"settings":cfg,
        "native_depth_definition":"optical Z in meters; not Euclidean range",
        "aligned_depth_definition":"source depth Z samples on color grid; not exact color optical Z",
        "color_order":{"capture":"BGR8","PLY":"RGB8","PNG":"standard RGB encoding via OpenCV BGR buffer"},
        "geometry_classes":{"floor":0,"geometry_or_semantic_depth_obstacle":1,"outside_band_or_unclassified":2,"invalid_or_unmapped":255},
        "scnn":s.stats.get("scnn"),"not_semantic_ground_truth":True,"PLY_untextured_points":"RGB 130,130,130; explicit mask in pointcloud_color_valid.npy","interpolation":"none; zero/missing never filled",
        "map_accumulation":cfg["mapping"]["mode"],
        "fusion":s.stats.get("map_diagnostics",{}).get("fusion",{}),
        "costmap_state_values":"uint16; 0..255 original cost meanings, 256 outside horizontal camera FOV",
        "costmap_in_view":"bool [X,Y]; false rendered black; unknown with risk is colored+hatch; unknown without risk gray",
        "volume_clear":"bool; complete sampled height checks passed, not continuous collision certification",
        "floor_evidence_bits":{"1":"measured floor","2":"confident SCNN floor + measured plane",
                               "4":"sampled height volume clear","8":"reserved legacy low-obstacle (unused in v3.5)","16":"SCNN obstacle/other vs geometric floor","32":"Depth obstacle vs confident SCNN floor: held unknown","64":"SCNN projected floor inferred free; no direct depth required","128":"bounded SCNN floor continuity inference; subset of bit64"},
        "free_policy":cfg["grid"]["free_space_policy"],
        "semantic_free_inferred":"bool separate SCNN+plane inferred free support; not a measured surface or full height clearance",
        "semantic_free_continuity":"bool subset of inferred free: bounded same-frame floor connection, not measured depth",
        "costmap_ground_view":"compatibility preview of costs on Z=0 only; no obstacle cloud",
        "obstacle_cloud_view":"Low Angle and High Angle virtual viewpoints of the same measured obstacle points; no costmap colors",
        "view_images":"saved with config presets; temporary keyboard rotations/zoom are display-only",
        "pointcloud_view_color":"RAINBOW of Euclidean distance from actual camera (T translation); default 0..6m; PLY keeps original RGB",
        "grid_origin":"floor below current native depth optical center; configured height/roll/pitch/yaw; no world pose estimation"}
    write_json(dest/(prefix+"processing_v3.5.json"),record)
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file() and p!=pending}
    write_json(dest/(prefix+"v3.5_complete.json"),{"frame_id":s.stats["frame_id"],"files_sha256":checksums})
    pending.unlink()  # [v3.5 추가] 이 함수가 방금 만든 상태 파일 하나만 제거한다.
    return dest

def save_live(root,s,session,cfg):
    # 전달된 latest가 아니라 화면에 표시한 계산 완료 스냅샷의 Capture를 저장한다.
    cap=s.packet.capture
    dest=save_capture(root,cap,s.packet.calibration,session)
    prefix=dest.name.split("_",1)[0]+"_"
    return save_extras(dest,prefix,s,cfg)

def save_replay(root,s,cfg):
    # 재현 결과는 원본 폴더를 수정하지 않고 명시한 출력 경로에만 생성한다.
    dest=Path(root)/s.packet.source_id
    dest.mkdir(parents=True,exist_ok=False)
    prefix=s.packet.source_id.split("_",1)[0]+"_"
    return save_extras(dest,prefix,s,cfg)
