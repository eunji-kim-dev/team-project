"""Jetson 표시 부담/CPU 스레드 수 제한. 장애물 높이/지도 해상도/관측 점은 변경하지 않는다."""
import json
import os
import platform
from pathlib import Path

DEFAULTS={"profile":"auto","math_threads":1,"opencv_threads":1,"ui_fps":15,
          "render_max_width":1280,"render_max_height":800,"cloud_display_max_points":8000}


def jetson_profile(settings):
    mode=settings.get("profile","auto")
    if mode!="auto":return mode=="jetson"
    if platform.system()!="Linux" or platform.machine().lower() not in ("aarch64","arm64"):return False
    if Path("/etc/nv_tegra_release").exists():return True
    try:model=Path("/proc/device-tree/model").read_text(errors="ignore").lower()
    except OSError:return False
    return "jetson" in model or "nvidia" in model


def configure_native_threads(root,argv):
    # NumPy/OpenCV를 import하기 전에 실행. 여러 라이브러리의 CPU 스레드 경쟁을 줄인다.
    cfg=json.loads((root/"config.json").read_text(encoding="utf-8-sig"))
    settings=dict(DEFAULTS,**cfg.get("performance",{}))
    override=None
    for i,arg in enumerate(argv):
        if arg=="--config" and i+1<len(argv):override=argv[i+1]
        elif arg.startswith("--config="):override=arg.split("=",1)[1]
    if override:
        settings.update(json.loads(Path(override).read_text(encoding="utf-8-sig")).get("performance",{}))
    if jetson_profile(settings):
        count=settings["math_threads"]
        if type(count) is not int or not 1<=count<=16:raise ValueError("performance.math_threads must be 1..16")
        for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            os.environ[name]=str(count)


def apply_profile(cfg):
    settings=cfg["performance"]
    if settings["profile"] not in ("auto","desktop","jetson"):
        raise ValueError("performance.profile must be auto/desktop/jetson")
    bounds={"math_threads":(1,16),"opencv_threads":(1,16),"ui_fps":(5,60),
            "render_max_width":(1200,4096),"render_max_height":(800,2160),
            "cloud_display_max_points":(1000,100000)}
    for name,(lo,hi) in bounds.items():
        value=settings[name]
        if type(value) is not int or not lo<=value<=hi:raise ValueError(f"performance.{name} must be {lo}..{hi}")
    if jetson_profile(settings):
        cfg["view"]["render_max_size"]=[settings["render_max_width"],settings["render_max_height"]]
        cfg["view"]["ui_fps"]=settings["ui_fps"]
        cfg["pointcloud_color"]["display_max_points"]=min(cfg["pointcloud_color"]["display_max_points"],settings["cloud_display_max_points"])
        cfg["processing"]["display_max_points"]=min(cfg["processing"]["display_max_points"],settings["cloud_display_max_points"])
        cfg["semantics"]["torch_threads"]=min(cfg["semantics"].get("torch_threads",2),settings["math_threads"])
    return cfg
