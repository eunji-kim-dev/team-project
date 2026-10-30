"""SDK 취득·정렬 스레드. GUI/계산은 최신 복사 프레임만 읽는다."""
from collections import deque
import threading
import time

class LatestAcquisition:
    def __init__(self,pipeline,capture,calibration,aligner,max_skew_ms,on_capture):
        self.pipeline=pipeline;self.capture=capture;self.calibration=calibration
        self.aligner=aligner;self.max_skew_ms=max_skew_ms;self.on_capture=on_capture
        self.condition=threading.Condition();self.halt=threading.Event()
        self.latest_capture=None;self.received_frames=0;self.arrivals=deque(maxlen=300)
        self.events=deque(maxlen=64);self.timeouts=0;self.incomplete=0;self.errors=0
        self.last_success=None;self.align_ms=0.;self.max_align_ms=0.
        self.max_gap_ms=0.;self.error="";self.last_error="";self.closed=False
        self.phase="starting";self.phase_since=time.monotonic();self.stop_thread=None
        self.backend="poll_for_frames" if hasattr(pipeline,"poll_for_frames") else "try_wait_for_frames"
        self.last_wait_notice=time.monotonic()
        self.thread=threading.Thread(target=self._run,name="RGBD-acquire-align",daemon=True)
        self.thread.start()
    def _run(self):
        while not self.halt.is_set():
            try:
                with self.condition:self.phase="SDK "+self.backend;self.phase_since=time.monotonic()
                # [v3.5] SDK 문서상 non-blocking poll로 최신 frameset을 가져온다.
                # 200ms 대기 호출에 GUI/정리가 묶이는 일을 줄인다.
                if self.backend=="poll_for_frames":
                    frames=self.pipeline.poll_for_frames();ok=bool(frames)
                else:ok,frames=self.pipeline.try_wait_for_frames(200)
                if not ok:
                    now=time.monotonic()
                    with self.condition:
                        if now-self.last_wait_notice>=.2:
                            self.timeouts+=1;self.last_wait_notice=now
                    self.halt.wait(.003)
                    continue
                start=time.monotonic()
                with self.condition:self.phase="copy + align";self.phase_since=start
                cap=self.capture(frames,self.aligner,self.calibration,self.max_skew_ms)
                del frames  # NumPy 복사 완료 후 SDK frameset 핸들을 즉시 해제한다.
                elapsed=(time.monotonic()-start)*1000
                if cap is None:
                    with self.condition:self.incomplete+=1
                    continue
                if self.halt.is_set():break
                self.on_capture(cap)
                now=time.monotonic()
                with self.condition:
                    gap=0. if self.last_success is None else (now-self.last_success)*1000
                    self.max_gap_ms=max(self.max_gap_ms,gap)
                    self.align_ms=elapsed;self.max_align_ms=max(self.max_align_ms,elapsed)
                    if gap>500:self.events.append({"kind":"capture_gap","monotonic":now,"ms":gap})
                    self.last_success=now;self.arrivals.append(now);self.error=""
                    self.last_wait_notice=now;self.phase="ready";self.phase_since=now
                    self.latest_capture=cap
                    self.received_frames+=1;self.condition.notify_all()
            except Exception as exc:
                with self.condition:
                    self.errors+=1;self.error=f"{type(exc).__name__}: {exc}";self.last_error=self.error
                    self.events.append({"kind":"acquisition_error","monotonic":time.monotonic(),"message":self.error})
                self.halt.wait(.05)
    def latest(self):
        with self.condition:return self.latest_capture
    def health(self):
        now=time.monotonic()
        with self.condition:
            recent=[t for t in self.arrivals if t>=now-5]
            fps=(len(recent)-1)/max(now-recent[0],.001) if len(recent)>1 else 0.
            return dict(fps=fps,capture_age_ms=None if self.last_success is None else
                    (now-getattr(self.latest_capture,"received_monotonic",self.last_success))*1000,
                capture_completion_age_ms=None if self.last_success is None else (now-self.last_success)*1000,
                align_ms=self.align_ms,max_align_ms=self.max_align_ms,max_capture_gap_ms=self.max_gap_ms,
                sdk_timeouts=self.timeouts,incomplete_frames=self.incomplete,errors=self.errors,
                error=self.error,last_error=self.last_error,received_frames=self.received_frames,events=list(self.events),
                sdk_phase=self.phase,sdk_phase_ms=(now-self.phase_since)*1000,backend=self.backend,
                cleanup_pending=self.cleanup_pending())
    def stop(self):
        if self.closed:return
        self.closed=True;self.halt.set()
        with self.condition:self.condition.notify_all()
        # [v3.5] pipeline.stop 자체가 멈춰도 창 이벤트를 영구적으로 막지 않는다.
        # 정리가 끝나기 전에는 recovery가 새 카메라 시작을 허용하지 않는다.
        self.thread.join(timeout=.3)
        def close_sdk():
            try:self.pipeline.stop()
            except Exception as exc:
                with self.condition:self.last_error="stop: "+str(exc);self.errors+=1
        self.stop_thread=threading.Thread(target=close_sdk,name="RGBD-stop",daemon=True)
        self.stop_thread.start();self.stop_thread.join(timeout=.3)
        self.thread.join(timeout=.1)
    def cleanup_pending(self):
        return self.closed and (self.thread.is_alive() or
            (self.stop_thread is not None and self.stop_thread.is_alive()))
