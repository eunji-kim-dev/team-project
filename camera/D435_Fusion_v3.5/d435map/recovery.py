"""수신 중단/장치 분리에는 창을 유지하고 재시도. 코드 오류는 숨기지 않는다."""
import threading
import time
import traceback

class UserStopped(Exception):
    pass

def recoverable_camera_error(exc):
    if not isinstance(exc,RuntimeError):return False
    text=str(exc).lower()
    permanent=("지원되지 않습니다","여러 대","보정","자료형","shape","dtype")
    if any(x in text for x in permanent):return False
    return any(x in text for x in (
        "프레임 수신이 8초","8초 동안 rgb/depth","연결된 realsense 카메라가 없습니다",
        "지정한 일련번호의 카메라가 없습니다","카메라 스트림을 시작하지 못했습니다",
        "device or resource busy","no device connected","device disconnected",
        "frame didn't arrive","usb device","failed to set power state"))

def responsive_start(start,pump):
    # [v3.5] SDK 시작/노출 준비가 늦어져도 UI 이벤트와 Q를 처리한다.
    # 종료 요청 후 뒤늦게 성공한 파이프라인도 닫아 장치를 점유하지 않는다.
    done=threading.Event();cancel=threading.Event();result={}
    close_lock=threading.Lock();closed=False
    def close_once():
        nonlocal closed
        with close_lock:
            if closed or "value" not in result:return
            closed=True
        try:result["value"][0].stop()
        except Exception:pass
    def run():
        try:result["value"]=start()
        except Exception as exc:result["error"]=exc
        finally:
            done.set()
            if cancel.is_set():close_once()
    thread=threading.Thread(target=run,name="RGBD-start",daemon=True);thread.start()
    while not done.wait(.01):
        if not pump("STARTING CAMERA | Q / Esc: quit"):
            cancel.set()
            # 시작이 막 끝난 경계에서는 완료 파이프라인을 직접 정리한다.
            if done.is_set() and "value" in result:
                threading.Thread(target=close_once,daemon=True).start()
            raise UserStopped()
    if "error" in result:raise result["error"]
    return result["value"]

def run_with_recovery(run,pump,on_disconnect,cleanup_ready,cfg,on_event):
    attempt=0
    while True:
        try:
            run()
            return "normal"
        except UserStopped:
            return "user stopped while starting/reconnecting"
        except Exception as exc:
            event={"kind":"run_error","error":f"{type(exc).__name__}: {exc}",
                   "traceback":traceback.format_exc(),"monotonic":time.monotonic()}
            on_event(event)
            if not cfg["enabled"] or not recoverable_camera_error(exc):raise
            attempt+=1;on_disconnect()
            # 종료되지 않은 SDK 객체 위에 새 스레드를 계속 쌓지 않는다.
            while not cleanup_ready():
                if not pump("WAITING FOR SDK CLEANUP | Q / Esc: quit"):return "user stopped during cleanup"
            deadline=time.monotonic()+cfg["retry_interval_s"]
            while time.monotonic()<deadline:
                if not pump(f"CAMERA LOST | reconnect attempt {attempt} | Q / Esc: quit"):
                    return "user stopped during reconnect"
            on_event({"kind":"reconnect_attempt","attempt":attempt,"monotonic":time.monotonic()})
