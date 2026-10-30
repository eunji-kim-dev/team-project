"""설정 검증: 잘못된 반경/높이/격자 입력으로 정상처럼 지도를 만들지 않는다."""
import copy
import json
import math
from pathlib import Path
from .robot import derive_robot

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None):
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    # [v3.5 추가] 이전 config.json도 새 표시/지도 모드 기본값으로 열 수 있게 한다.
    config.setdefault("mapping", {"mode":"camera_relative"})
    config.setdefault("recording", {"fps":15,"max_width":1920,"max_height":1080})
    if path and Path(path).resolve() != (ROOT / "config.json").resolve():
        supplied = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        for group, values in supplied.items():
            if group not in config or not isinstance(values, dict):
                raise ValueError(f"Unknown configuration section: {group}")
            extra = values.keys() - config[group].keys()
            if extra:
                raise ValueError(f"Unknown settings in {group}: {extra}")
            config[group].update(values)
    # [v3.5] 너비에서 수평 비용 거리를 유도. 로봇 키는 grid.obstacle_max_height_m 사용.
    config["robot"]=derive_robot(config["robot"]["width_m"],config["robot"])
    validate(config)
    return config

def validate(c):
    for section in c.values():
        for key, value in section.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
    g, r, d, f = (c[k] for k in ("grid", "robot", "depth", "floor"))
    positive = [(g,"resolution_m"), (g,"free_vertical_step_m"),
                (g,"obstacle_stale_s"), (g,"free_stale_s"),
                (c["camera"],"height_m"), (c["camera"],"target_fps"),
                (c["processing"],"max_hz"), (c["processing"],"max_frame_age_s"),
                (f,"inlier_distance_m"), (f,"max_samples"), (f,"ransac_iterations")]
    for section, key in positive:
        if section[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not (0 < d["min_m"] < d["max_m"]):
        raise ValueError("depth range must be 0 < min < max")
    if not (0 < d["min_valid_fraction"] <= 1 and 0 < f["min_ratio"] <= 1):
        raise ValueError("fractions must lie in (0,1]")
    if not (0 <= g["obstacle_min_height_m"] < g["obstacle_max_height_m"]):
        raise ValueError("invalid obstacle height interval")
    if r["radius_m"] < 0 or r["safety_margin_m"] < 0:
        raise ValueError("robot radius/margin cannot be negative")
    if r["inflation_radius_m"] < r["radius_m"] + r["safety_margin_m"]:
        raise ValueError("total inflation radius must include forbidden radius")
    if r.get("cost_direction","all") not in ("all","camera_visible"):
        raise ValueError("robot.cost_direction must be all or camera_visible")
    if "width_m" in r:
        expected=derive_robot(r["width_m"],r)
        if r["cost_direction"]=="camera_visible":
            for key in ("radius_m","safety_margin_m","forbidden_radius_m","inflation_extra_m","inflation_radius_m"):
                if not math.isclose(r[key],expected[key],abs_tol=1e-9):
                    raise ValueError("Edit robot.width_m; radius and inflation are derived")
    if r["cost_scaling_per_m"] <= 0:
        raise ValueError("cost decay must be positive")
    if not (0 <= g["obstacle_hold_s"] < g["obstacle_stale_s"]):
        raise ValueError("hold must be shorter than stale timeout")
    for key in ("clear_confirmations", "min_obstacle_points"):
        if int(g[key]) != g[key] or g[key] < 1:
            raise ValueError(f"{key} must be an integer >= 1")
    for axis in ("x", "y"):
        extent = g[f"{axis}_max_m"] - g[f"{axis}_min_m"]
        if extent <= 0 or not math.isclose(extent / g["resolution_m"], round(extent / g["resolution_m"]), abs_tol=1e-6):
            raise ValueError("grid extents must be positive multiples of resolution")
    if ((g["x_max_m"]-g["x_min_m"])*(g["y_max_m"]-g["y_min_m"]) / g["resolution_m"]**2) > 100000:
        raise ValueError("grid exceeds 100000 cells; reduce range/resolution cost")
    if g["obstacle_min_height_m"]<=f["inlier_distance_m"]:
        raise ValueError("Obstacle minimum height must exceed floor inlier distance")
    for section,key in [(f,"max_samples"),(f,"min_points"),(f,"ransac_iterations"),(c["processing"],"display_max_points"),(c["view"],"width"),(c["view"],"height"),(c["camera"],"target_fps")]:
        if section[key]<3 or int(section[key])!=section[key]: raise ValueError(f"{key} must be an integer >=3")
    if f["max_samples"]<f["min_points"] or not (0<f["max_tilt_deg"]<85):
        raise ValueError("Invalid floor sample count or tilt limit")
    if d["bilateral_sigma_m"]<=0 or d["neighbor_delta_m"]<=0 or d["neighbor_min"] not in (1,2,3,4):
        raise ValueError("Invalid depth filter settings")
    if min(c["processing"]["display_voxel_m"],c["processing"]["cloud_save_voxel_m"])<0:
        raise ValueError("Voxel size must be >=0")
    s=c["semantics"]
    if not isinstance(s["enabled"],bool) or not isinstance(s["model_dir"],str) or not s["model_dir"].strip():
        raise ValueError("semantics.enabled must be bool; model_dir must be a nonempty directory path")
    if not 0<=s["overlay_alpha"]<=1:
        raise ValueError("semantics.overlay_alpha must lie in [0,1]")
    if c["view"]["width"] < 900 or c["view"]["height"] < 700:
        raise ValueError("dashboard minimum size: 900 x 700")
    if c["mapping"]["mode"] not in ("camera_relative","fixed_accumulated"):
        raise ValueError("mapping.mode must be camera_relative or fixed_accumulated")
    if c["grid"].get("visibility_policy","strict") not in ("strict","valid_center"):
        raise ValueError("visibility_policy must be strict or valid_center")
    if c["grid"].get("visibility_min_neighbors",5) not in range(1,10):
        raise ValueError("visibility_min_neighbors must be in [1,9]")
    if c["semantics"].get("torch_threads",2) not in range(1,17):
        raise ValueError("torch_threads must be in [1,16]")
    if c["grid"].get("free_space_policy","volume_confirmed") not in ("observed_floor","volume_confirmed"):
        raise ValueError("free_space_policy must be observed_floor or volume_confirmed")
    fuser=c.get("fusion",{})
    if fuser:
        if not isinstance(fuser["enabled"],bool) or not isinstance(fuser["use_floor_seeds"],bool):
            raise ValueError("fusion switches must be bool")
        for key in ("floor_confidence","obstacle_confidence"):
            if not 0.5<=fuser[key]<=1:raise ValueError("fusion confidence must be in [.5,1]")
        if fuser["boundary_erode_px"] not in range(4):raise ValueError("boundary_erode_px must be 0..3")
        if not f["inlier_distance_m"]<=fuser["semantic_floor_tolerance_m"]<g["obstacle_min_height_m"]:
            raise ValueError("semantic floor tolerance must not overlap geometry obstacle band")
        if not f["inlier_distance_m"]<fuser["semantic_obstacle_min_height_m"]<=g["obstacle_min_height_m"]:
            raise ValueError("semantic obstacle height must exceed floor noise tolerance")
        if not isinstance(fuser["defer_floor_obstacle_conflict"],bool):
            raise ValueError("defer_floor_obstacle_conflict must be bool")
        if not .5<=fuser["floor_obstacle_confidence"]<=1:
            raise ValueError("floor_obstacle_confidence must be in [.5,1]")
        if not math.isclose(fuser["semantic_obstacle_min_height_m"],g["obstacle_min_height_m"]):
            raise ValueError("SCNN and Depth must share the obstacle minimum height")
        if type(fuser["semantic_obstacle_min_points"]) is not int or fuser["semantic_obstacle_min_points"]<1:
            raise ValueError("semantic_obstacle_min_points must be positive integer")
    if type(c["fusion"].get("semantic_free_enabled",False)) is not bool:
        raise ValueError("semantic_free_enabled must be bool")
    if not .5<=c["fusion"].get("semantic_free_confidence",.9)<=1:
        raise ValueError("semantic_free_confidence must be .5..1")
    # [v3.5] SCNN 연결 보완 거리/신뢰도 검증. false로 연결 단계만 끌 수 있다.
    if type(c["fusion"].get("semantic_free_continuity_enabled",False)) is not bool:
        raise ValueError("semantic_free_continuity_enabled must be bool")
    if not .5<=c["fusion"].get("semantic_free_continuity_confidence",.75)<=1:
        raise ValueError("semantic_free_continuity_confidence must be .5..1")
    if not 0<=c["fusion"].get("semantic_free_continuity_distance_m",.15)<=.5:
        raise ValueError("semantic_free_continuity_distance_m must be 0..0.5m")
    col=c["pointcloud_color"]
    if col["mode"] not in ("rainbow","red"):raise ValueError("pointcloud_color.mode must be rainbow or red")
    if not 0<=col["min_distance_m"]<col["max_distance_m"]:raise ValueError("Invalid pointcloud color range")
    if type(col["display_max_points"]) is not int or not 1<=col["display_max_points"]<=200000:
        raise ValueError("display_max_points must be 1..200000")
    support=c["obstacle_support"]
    if not isinstance(support["enabled"],bool) or type(support["min_neighbors"]) is not int or not 1<=support["min_neighbors"]<=8:
        raise ValueError("obstacle_support needs bool enabled and 1..8 integer min_neighbors")
    if support["depth_jump_base_m"]<=0 or support["depth_jump_per_m2"]<0:
        raise ValueError("obstacle depth continuity tolerance must be positive / nonnegative")
    recovery=c["recovery"]
    if not isinstance(recovery["enabled"],bool) or not 0<=recovery["retry_interval_s"]<=60:
        raise ValueError("invalid camera recovery settings")
    if type(recovery["ui_wait_ms"]) is not int or not 1<=recovery["ui_wait_ms"]<=100:
        raise ValueError("recovery ui_wait_ms must be integer 1..100")
    v=c["view"]
    if v["cloud_mode"] not in ("obstacle_low_angle","rgb"):
        raise ValueError("cloud_mode must be obstacle_low_angle or rgb")
    if v["costmap_view"] not in ("ground","top"):
        raise ValueError("costmap_view must be ground or top")
    # [v3.5] Low Angle/High Angle는 표시 설정이며 센서 설치 보정이나 지도 좌표와 무관하다.
    for name in ("low_angle","high_angle"):
        v=c[name]
        if not (.05<=v["costmap_eye_height_m"]<=5 and .0<=v["costmap_eye_back_m"]<=5):
            raise ValueError(f"Invalid {name} virtual eye position")
        if not (-80<=v["costmap_pitch_deg"]<=80 and 35<=v["costmap_hfov_deg"]<=120):
            raise ValueError(f"Invalid {name} virtual pitch / FOV")
        if not .2<=v["cloud_center_y_fraction"]<=.8:
            raise ValueError(f"{name}.cloud_center_y_fraction must be in [.2,.8]")
    # [v3.5] 높이 기준과 분리한 후보 신뢰도/단기 확인 설정.
    review=c["candidate_review"];temporal=c["temporal_review"]
    for sec,key in [(review,"enabled"),(review,"local_floor_enabled"),(temporal,"enabled")]:
        if type(sec[key]) is not bool:raise ValueError(f"{key} must be bool")
    for sec,key,minimum in [(review,"max_component_points",1),(review,"local_min_points",6),
       (review,"local_max_samples",6),(review,"max_local_checks",1),(temporal,"confirm_frames",2),
       (temporal,"motion_min_tracks",6)]:
        if type(sec[key]) is not int or sec[key]<minimum:raise ValueError(f"Invalid {key}")
    if review["local_max_samples"]<review["local_min_points"]:raise ValueError("Too few local samples")
    if review["local_min_sectors"] not in (2,3,4):raise ValueError("Invalid local_min_sectors")
    for sec,keys in [(review,("max_height_m","max_vertical_span_m","max_xy_span_m","local_radius_m",
        "local_min_extent_m","local_max_rmse_m","local_max_tilt_deg","local_max_shift_m")),
        (temporal,("match_xy_m","match_z_m","max_gap_s","motion_flow_px","motion_rotation_deg",
         "motion_scale_delta","motion_depth_delta_m"))]:
        for key in keys:
            if sec[key]<=0:raise ValueError(f"{key} must be positive")
    if not 0<=review["local_inner_radius_m"]<review["local_radius_m"]:raise ValueError("Invalid local radius")
    if review["max_height_m"]<g["obstacle_min_height_m"]:raise ValueError("Candidate height below minimum")
    if review["local_max_tilt_deg"]>=45:raise ValueError("Local floor tilt must be <45 degrees")
    for sec,key in [(review,"semantic_strong_fraction"),(review,"local_min_inlier_ratio"),
                    (temporal,"motion_depth_fraction"),(temporal,"motion_min_inlier_ratio")]:
        if not 0<sec[key]<=1:raise ValueError(f"Invalid {key}")
    # [추가] 녹화 설정은 지도/SCNN 계산과 독립적이며 과도한 해상도를 제한한다.
    rec=c.get("recording",{"fps":15,"max_width":1920,"max_height":1080})
    if isinstance(rec["fps"],bool) or not 1<=rec["fps"]<=60:
        raise ValueError("recording.fps must be 1..60")
    for key in ("max_width","max_height"):
        if type(rec[key]) is not int or not 16<=rec[key]<=4096:
            raise ValueError(f"recording.{key} must be integer 16..4096")
    return copy.deepcopy(c)
