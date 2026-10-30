"""D435 v3.5 독립 실행기: 현재 모듈만 직접 연결한다."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import json
import cv2
import numpy as np
from d435map.configuration import load_config,validate
from d435map.engine import Processor
from d435map.io import load_capture,save_replay
from d435map.display import Dashboard
from d435map.window import create_window,window_size

def replay(args,cfg):
    # [engine 추가] 카메라 없는 재생도 조기 Q/예외를 포함해 GPU 메모리를 정리한다.
    processor=Processor(cfg)
    try:return _replay(args,cfg,processor)
    finally:processor.semantic.close()


def _replay(args,cfg,processor):
    root=args.replay
    if list(root.glob("*_metadata.json")) or (root/"metadata.json").exists():
        folders=[root]
    else:
        folders=sorted({p.parent for p in root.rglob("*_metadata.json")})
    if args.limit: folders=folders[:args.limit]
    if not folders: raise ValueError("No saved v2.2 capture folders found")
    dashboard=Dashboard(cfg); rows=[]
    clock_origin=time.monotonic(); first_unix=None; previous_unix=None
    if args.export: args.export.mkdir(parents=True,exist_ok=True)
    replay_window="D435 v3.5 replay"
    if not args.headless:create_window(replay_window,cfg)
    print("REPLAY: fixed-pose accumulation" if args.sequence_fixed else "REPLAY: chronological candidate history, current-frame map" if args.sequence else "REPLAY: independent captures; short history reset")
    for i,folder in enumerate(folders):
        # [v3.5 추가] 여러 방향으로 촬영한 자료를 하나의 고정 지도에 잘못 누적하지 않는다.
        if not (args.sequence_fixed or args.sequence):
            processor.grid.reset()
            processor.last_signature=None
            processor.reviewer.reset()
        try:
            packet=load_capture(folder)
            if args.sequence_fixed or args.sequence:
                stamp=packet.capture.received_unix_ns
                if previous_unix is not None and stamp<=previous_unix:
                    raise ValueError("Sequence timestamps must strictly increase; folder order is not chronological")
                if first_unix is None: first_unix=stamp
                previous_unix=stamp
                packet.capture.received_monotonic=clock_origin+(stamp-first_unix)/1e9
            snapshot=processor.process(packet)
            row=dict(snapshot.stats,folder=str(folder),floor=snapshot.floor.record(),
                     free_cells=int((snapshot.physical==0).sum()),
                     obstacle_cells=int((snapshot.physical==100).sum()),
                     unknown_cells=int((snapshot.physical==-1).sum()))
            rows.append(row)
            if args.export: save_replay(args.export,snapshot,cfg)
            print(f"[{i+1}/{len(folders)}] {folder.name} floor={snapshot.floor.ok} "
                  f"points={len(snapshot.points)} compute={snapshot.stats['processing_ms']:.1f}ms "
                  f"free={row['free_cells']} obstacle={row['obstacle_cells']}")
            if not args.headless:
                while True:
                    cv2.imshow(replay_window,dashboard.render(snapshot,status=f"{i+1}/{len(folders)} SPACE/N next, S export, Q quit",size=window_size(replay_window,cfg)))
                    key=cv2.waitKey(30)&255
                    dashboard.key(key)
                    if key in (ord("q"),27): return rows
                    if key==ord("s"):
                        if not args.output: raise ValueError("--output existing directory required for replay S save")
                        dest=save_replay(args.output,snapshot,cfg); print(dest)
                    if key in (ord("n"),ord(" "),13): break
        except Exception as exc:
            processor.reviewer.reset()
            rows.append({"folder":str(folder),"error":str(exc)})
            print("REPLAY FAILED:",folder,exc)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps({"mode":"offline CPU compute, excludes camera acquisition/display/disk",
            "fixed_pose_sequence":args.sequence_fixed,"candidate_sequence":args.sequence or args.sequence_fixed,"settings":cfg,"captures":rows},ensure_ascii=False,indent=2),encoding="utf-8")
    return rows

def prepare_output(path=None,project_root=None):
    # [v3.5 추가] VS Code의 작업 디렉터리와 무관하게 저장 상위 폴더를 준비한다.
    # 개별 순번/날짜 촬영 폴더와 데이터는 기존 코드가 S키 입력 시에만 만든다.
    root=Path(__file__).resolve().parents[1] if project_root is None else Path(project_root)
    target=(root/"captures") if path is None else Path(path).expanduser()
    target=target.resolve()
    target.mkdir(parents=True,exist_ok=True)
    if not target.is_dir():raise ValueError(f"Save root is not a directory: {target}")
    return target

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,help="JSON override; defaults to bundled config.json")
    parser.add_argument("--models-dir",type=Path,help="Folder with exactly one .pth OR .engine and standard_fastscnn.json; default: models")
    parser.add_argument("--check-model",action="store_true",help="Check model I/O and run one synthetic inference without camera or GUI")
    parser.add_argument("--output",type=Path,help="Save root (created if missing); default: captures in the project folder")
    parser.add_argument("--fps",type=int,help="Explicit acquisition FPS, no fallback when set")
    parser.add_argument("--serial",default="")
    parser.add_argument("--map-mode",choices=["camera_relative","fixed_accumulated"],help="Fresh local frame (default) or fixed camera accumulation")
    parser.add_argument("--max-skew-ms",type=float)
    parser.add_argument("--replay",type=Path,help="One saved capture or parent directory")
    parser.add_argument("--headless",action="store_true",help="Replay without GUI")
    parser.add_argument("--export",type=Path,help="Explicitly save replay outputs under this directory")
    parser.add_argument("--report",type=Path,help="Write replay measurements JSON")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--sequence",action="store_true",help="Chronological frames: short candidate history, current-frame map")
    parser.add_argument("--sequence-fixed",action="store_true",help="Only for chronological frames at ONE fixed pose")
    args=parser.parse_args()
    if args.limit is not None and args.limit<1: parser.error("--limit must be positive")
    cfg=load_config(args.config)
    if args.map_mode: cfg["mapping"]["mode"]=args.map_mode
    if args.sequence_fixed: cfg["mapping"]["mode"]="fixed_accumulated"
    if args.models_dir:
        cfg["semantics"]["model_dir"]=str(args.models_dir.resolve())
        cfg["semantics"]["enabled"]=True
    validate(cfg)
    from .performance import jetson_profile
    optimized=jetson_profile(cfg["performance"])
    cv2.setNumThreads(cfg["performance"]["opencv_threads"] if optimized else 2)
    print("Performance:","jetson" if optimized else "desktop","| full-depth geometry / display-only limits")
    try:
        if args.check_model:
            from .semantics import FastSCNN
            model=FastSCNN(cfg["semantics"])
            try:
                print("SCNN:",model.status,"|",model.message)
                if model.net is None:raise RuntimeError(model.message)
                inp=model.spec["input"]
                # 실행 호환성 검사이며 의미 정확도/mIoU 검증이 아니다.
                result=model.predict(np.full((inp["height"],inp["width"],3),127,np.uint8),"model-check")
                print(json.dumps(result.record(),ensure_ascii=False,indent=2))
                if result.status!="OK":raise RuntimeError(result.message)
                print("Model inference OK. This synthetic input does NOT measure accuracy or live FPS.")
            finally:model.close()
        elif args.replay:
            if not args.headless:args.output=prepare_output(args.output)
            rows=replay(args,cfg)
            if any("error" in row for row in rows): raise RuntimeError("Replay had failed captures; inspect report/console")
        else:
            if args.headless or args.export or args.report or args.sequence_fixed or args.sequence:
                parser.error("headless/export/report/sequence/sequence-fixed require --replay")
            args.output=prepare_output(args.output)
            print(f"Save root: {args.output}")
            from d435map.runtime import run_live
            run_live(cfg,args)
    finally:
        if args.replay:cv2.destroyAllWindows()
