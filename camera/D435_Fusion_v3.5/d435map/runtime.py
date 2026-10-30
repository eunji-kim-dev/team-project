"""v3.5 직접 실행 루프. 이전 main/CV 프록시/중복 미리보기/argv 재파싱을 제거."""
from pathlib import Path
from datetime import datetime
import copy,faulthandler,hashlib,json,sys,time,uuid
import cv2
from . import sensor
from .engine import Packet
from .display import Dashboard
from .window import create_window,window_size
from .workers import LatestWorker,SaveWorker,rate
from .acquisition import LatestAcquisition
from .recovery import responsive_start,run_with_recovery
from .recording import VideoRecorder

WINDOW="D435 v3.5 | RGB / SCNN / COSTMAP / Low Angle / High Angle"

def window_open():
    try:return cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)>=1
    except cv2.error:return False

def request_save(saver,root,snapshot,session,cfg,acquisition_health,processing_health):
    # [v3.5] S 시점의 표시 완료 프레임을 고정. 최신 RGB를 별도로 섞지 않는다.
    if snapshot is None:raise ValueError("No processed frame available")
    if acquisition_health.get("received_frames",0)<10:
        raise ValueError("Waiting for the first 10 paired captures")
    if time.monotonic()-snapshot.packet.capture.received_monotonic>cfg["processing"]["max_frame_age_s"]:
        raise ValueError("Displayed result too old to save")
    saved=copy.copy(snapshot)
    saved.stats=dict(snapshot.stats,live_health={"acquisition":acquisition_health,"processing":processing_health})
    return saver.submit(root,saved,session)

def run_live(cfg,args):
    # [v3.5] 계산/저장 스레드는 한 번 생성하며 재연결 중 모델을 다시 읽지 않는다.
    run_started=time.monotonic();worker=LatestWorker(cfg);saver=SaveWorker(cfg)
    dashboard=Dashboard(cfg,interactive_robot=True)
    recorder=VideoRecorder(args.output,cfg.get("recording"))
    print("SCNN:",worker.processor.semantic.status,"|",worker.processor.semantic.message)
    displayed=None;acquisition=None;report={};events=[];exit_reason="starting"
    log_dir=Path(__file__).resolve().parents[1]/"logs";log_dir.mkdir(parents=True,exist_ok=True)
    event_path=log_dir/"runtime_events.jsonl";event_path.write_text("",encoding="utf8")
    fault_file=None
    if not faulthandler.is_enabled():
        fault_file=(log_dir/"native_faults.log").open("w",encoding="utf8");faulthandler.enable(fault_file)

    def on_event(event):
        row=dict(event,recorded_at=datetime.now().astimezone().isoformat(),
                 acquisition=acquisition.health() if acquisition else {},processing=worker.health())
        current=worker.latest()
        if current:
            stats=current.stats;review=stats.get("map_diagnostics",{}).get("candidate_review",{})
            row["last_processed"]={"frame_id":stats["frame_id"],"received_at":stats["received_at"],
                "processing_ms":stats["processing_ms"],"candidate_review_ms":review.get("elapsed_ms"),
                "history_state":review.get("motion",{}).get("reason")}
        row["last_render_ms"]=getattr(dashboard,"last_render_ms",None)
        events.append(row);del events[:-64]
        with event_path.open("a",encoding="utf8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
        print(row.get("kind"),row.get("error",""))

    def show(image):
        # [추가] MP4에는 상태 문구가 없는 원본 대시보드를 전달한다.
        # REC/경과시간/저장 완료 안내는 화면용 복사본에만 그린다.
        recorder.submit(image)
        cv2.imshow(WINDOW,recorder.decorate(image))
        return image

    def recording_key(key,image):
        if key in (ord("r"),ord("R")):
            print(recorder.toggle(image));return True
        return False

    def pump(message):
        # [추가] 카메라 재연결 중에도 녹화/정지 및 창 이벤트를 계속 처리한다.
        image=show(dashboard.render(None,status=message,size=window_size(WINDOW,cfg)))
        key=cv2.waitKey(cfg["recovery"]["ui_wait_ms"])&255
        recording_key(key,image)
        return key not in (27,ord("q")) and window_open()

    def disconnect():
        nonlocal displayed
        worker.reset();displayed=None

    def cleanup_ready():
        return acquisition is None or not acquisition.cleanup_pending()

    def connected_session():
        nonlocal acquisition,displayed,report
        pipeline,profile,fps,failures=responsive_start(
            lambda:sensor.start_camera(args,cfg["camera"]["target_fps"]),pump)
        local_acquisition=None
        try:
            device=profile.get_device()
            depth=profile.get_stream(sensor.rs.stream.depth);color=profile.get_stream(sensor.rs.stream.color)
            calibration=sensor.calibration_info(device,depth,color,device.first_depth_sensor().get_depth_scale())
            supported=[]
            for detector in device.query_sensors():
                for stream in detector.get_stream_profiles():
                    if stream.is_video_stream_profile():
                        v=stream.as_video_stream_profile()
                        supported.append(dict(stream=str(stream.stream_type()),format=str(stream.format()),
                                              width=v.width(),height=v.height(),fps=v.fps()))
            report={"device":calibration["device"],"supported_streams":supported,
                    "active_streams":[calibration["depth_intrinsics"],calibration["color_intrinsics"]]}
            print("Actual device:",report["device"],"| active FPS:",fps)
            skew=args.max_skew_ms if args.max_skew_ms is not None else 750/fps
            entry=Path(__file__).resolve().parents[1]/"v3.5.py"
            session={"schema":"realsense-capture-v2.2-session","producer":"D435 Fusion v3.5 independent",
                "session_id":"run_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8],
                "started_at":sensor.now_iso(),"script_sha256":hashlib.sha256(entry.read_bytes()).hexdigest(),
                "calibration":calibration,"selected_fps":fps,"max_skew_ms":skew,
                "startup_failures":failures,"python_version":sys.version,"output_directory":str(args.output),
                "array_dimensions":"NPY: height, width; color PNG: height, width, 3",
                "v3.5_extension":{"entry":"v3.5.py","device_report":report,
                    "queue_policy":"latest one frame, replace older pending input","legacy_runtime_loaded":False}}
            local_acquisition=LatestAcquisition(pipeline,sensor.capture_from_frames,calibration,
                sensor.rs.align(sensor.rs.stream.color),skew,
                lambda cap:worker.submit(Packet(cap,calibration,"live",False)))
            acquisition=local_acquisition;opened=time.monotonic()
            while True:
                # [v3.5] 새 5패널에 필요한 최신 RGB/완료 결과만 읽는다.
                # 예전 2분할/Depth 컬러맵/준비 화면의 이중 렌더링은 만들지 않는다.
                fresh=acquisition.latest();health=acquisition.health();proc_health=worker.health()
                age=health["capture_completion_age_ms"]
                if (age is not None and age>8000) or (fresh is None and time.monotonic()-opened>8):
                    raise RuntimeError("카메라 프레임 수신이 8초 이상 중단됐습니다.")
                displayed=worker.latest()
                rates={"acquire":health["fps"],"cloud":rate(worker.cloud_times),
                       "map":rate(worker.map_times),"scnn":rate(worker.scnn_times)}
                status=worker.error or health.get("error","")
                if saver.busy:status="SAVING"
                elif saver.status.startswith("SAVE FAILED"):status=saver.status
                image=show(dashboard.render(displayed,rates,status,live_capture=fresh,
                                                    size=window_size(WINDOW,cfg)))
                key=cv2.waitKey(15)&255
                if not window_open():return
                if recording_key(key,image):continue
                if dashboard.key(key):
                    width=dashboard.robot_input.take_request()
                    if width is not None:worker.request_robot_width(width)
                    height=dashboard.height_input.take_request()
                    if height is not None:worker.request_robot_height(height)
                    continue
                if key in (ord("q"),27):return
                # [추가] R은 녹화 전용. 기존 수동 지도 초기화는 T로 이동.
                if key in (ord("t"),ord("T")):worker.reset();displayed=None
                elif key==ord("s"):
                    try:print(request_save(saver,args.output,displayed,session,cfg,health,proc_health))
                    except ValueError as exc:print("저장 보류:",exc)
        finally:
            if local_acquisition is not None:local_acquisition.stop()
            else:pipeline.stop()

    try:
        create_window(WINDOW,cfg);cv2.setMouseCallback(WINDOW,dashboard.mouse)
        exit_reason=run_with_recovery(connected_session,pump,disconnect,cleanup_ready,cfg["recovery"],on_event)
    except Exception as exc:
        exit_reason=f"fatal: {type(exc).__name__}: {exc}";raise
    finally:
        # [추가] 정리 대기시간이 영상 길이에 더해지지 않게 종료 시각부터 고정.
        recorder.stop()
        try:
            record={"elapsed_s":time.monotonic()-run_started,
                "acquisition":acquisition.health() if acquisition else {},"processing":worker.health(),
                "device_report":report,"last_result_stats":displayed.stats if displayed else None,
                "exit_reason":exit_reason,"recovery_events":events,"runtime_architecture":"v3.5 direct"}
            (log_dir/"latest_runtime.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf8")
        except Exception as exc:print("Runtime log write failed:",exc)
        try:worker.close()
        finally:
            try:saver.close()
            finally:
                try:recorder.close()
                finally:
                    cv2.destroyAllWindows()
                    if fault_file is not None:faulthandler.disable();fault_file.close()
