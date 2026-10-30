"""최신 1프레임 처리와 제한된 저장 큐. 오래된 처리 대기열을 쌓지 않는다."""
from collections import deque
import threading
import time
from .engine import Processor
from .robot import derive_robot,validate_robot_height
from .io import save_live

def rate(times,now=None):
    now=time.monotonic() if now is None else now
    recent=[t for t in list(times) if t>=now-5]
    return (len(recent)-1)/max(now-recent[0],.001) if len(recent)>1 else 0.

class LatestWorker:
    def __init__(self,cfg):
        self.cfg=cfg; self.processor=Processor(cfg)
        self.condition=threading.Condition()
        self.pending_robot=None; self.pending_height=None
        self.pending=None; self.result=None; self.error=""
        self.stop=False; self.reset_requested=False; self.dropped=0; self.epoch=0
        self.busy_since=None; self.max_processing_ms=0.; self.failures=0; self.last_error=""
        self.cloud_times=deque(maxlen=300); self.map_times=deque(maxlen=300)
        self.scnn_times=deque(maxlen=300)
        self.thread=threading.Thread(target=self._run,name="RGBD-geometry",daemon=True)
        self.thread.start()
    def request_robot_width(self,width):
        # [v3.5] UI는 요청만 전달. 계산 중 설정을 수정하거나 카메라를 재시작하지 않는다.
        robot=derive_robot(width,self.cfg["robot"])
        with self.condition:
            self.pending_robot=robot
            self.condition.notify()
        return robot
    def request_robot_height(self,height):
        # [추가] UI에서 계산 설정을 직접 바꾸지 않고 최신 요청 하나만 전달한다.
        height=validate_robot_height(height,self.cfg["grid"]["obstacle_min_height_m"])
        with self.condition:
            self.pending_height=height
            self.condition.notify()
        return height
    def submit(self,packet):
        with self.condition:
            if self.pending is not None: self.dropped+=1
            self.pending=packet
            self.condition.notify()
    def reset(self):
        with self.condition:
            self.reset_requested=True
            self.epoch+=1
            self.pending=None
            self.result=None
            self.condition.notify()
    def latest(self):
        with self.condition: return self.result
    def health(self):
        with self.condition:
            return dict(busy_ms=0. if self.busy_since is None else (time.monotonic()-self.busy_since)*1000,
                        max_processing_ms=self.max_processing_ms,failures=self.failures,
                        last_error=self.last_error,dropped_pending=self.dropped)
    def _run(self):
        last=0.
        while True:
            with self.condition:
                while not self.stop and self.pending is None and not self.reset_requested:
                    self.condition.wait(.2)
                if self.stop: return
                if self.reset_requested:
                    self.processor.grid.reset()
                    self.processor.last_signature=None
                    if hasattr(self.processor,"reviewer"):self.processor.reviewer.reset()
                    self.reset_requested=False
                if self.pending is None: continue
                wait=last+1/self.cfg["processing"]["max_hz"]-time.monotonic()
                if wait>0:
                    self.condition.wait(wait)
                    continue
                if self.pending_robot is not None:
                    self.processor.set_robot(self.pending_robot)
                    self.pending_robot=None
                if self.pending_height is not None:
                    self.processor.set_robot_height(self.pending_height)
                    self.pending_height=None
                packet=self.pending; self.pending=None
                epoch=self.epoch
                last=time.monotonic()
                self.busy_since=last
            try:
                result=self.processor.process(packet)
                now=time.monotonic()
                with self.condition:
                    if epoch!=self.epoch: continue
                    self.result=result; self.error=""
                    self.cloud_times.append(now)
                    if result.stats.get("scnn",{}).get("status")=="OK": self.scnn_times.append(now)
                    if result.stats["map_updated"]: self.map_times.append(now)
            except Exception as exc:
                if hasattr(self.processor,"reviewer"):self.processor.reviewer.reset()
                with self.condition:
                    self.error=f"{type(exc).__name__}: {exc}"
                    self.last_error=self.error;self.failures+=1
            finally:
                with self.condition:
                    self.max_processing_ms=max(self.max_processing_ms,(time.monotonic()-last)*1000)
                    self.busy_since=None
    def close(self):
        with self.condition:
            self.stop=True; self.condition.notify()
        self.thread.join(timeout=10)
        if self.thread.is_alive(): raise RuntimeError("processing worker did not stop")
        # [engine 추가] 실행 중 버퍼를 해제하지 않도록 추론 스레드 join 이후에만 정리.
        self.processor.semantic.close()

class SaveWorker:
    def __init__(self,cfg):
        self.cfg=cfg
        self.condition=threading.Condition(); self.job=None; self.busy=False
        self.stop=False; self.status=""
        self.thread=threading.Thread(target=self._run,name="RGBD-save",daemon=False)
        self.thread.start()
    def submit(self,root,s,session):
        with self.condition:
            if self.busy or self.job is not None: raise ValueError("Previous save still running; retry S shortly.")
            self.job=(root,s,session); self.status="Save queued for "+s.stats["frame_id"]
            self.condition.notify()
        return self.status
    def _run(self):
        while True:
            with self.condition:
                while self.job is None and not self.stop: self.condition.wait(.2)
                if self.job is None and self.stop: return
                job=self.job; self.job=None; self.busy=True
            try:
                dest=save_live(*job,self.cfg)
                self.status="Saved: "+str(dest)
                print(self.status)
            except Exception as exc:
                self.status=f"SAVE FAILED: {exc}"
                print(self.status)
            finally:
                with self.condition: self.busy=False
    def close(self):
        with self.condition:
            self.stop=True; self.condition.notify()
        self.thread.join(timeout=30)
        if self.thread.is_alive(): raise RuntimeError("save worker still writing; do not remove output files")
