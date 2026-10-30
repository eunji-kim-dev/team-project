"""R키 대시보드 MP4 녹화. 최신 화면 2장만 보관하며 인코딩은 UI와 분리한다."""
from datetime import datetime
from pathlib import Path
import math
import threading
import time
import cv2
import numpy as np


def video_size(image, max_width, max_height):
    # MP4 인코더의 짝수 크기 요구를 만족하고 원본보다 확대하지 않는다.
    h,w=image.shape[:2]
    scale=min(1.,max_width/w,max_height/h)
    return max(2,int(w*scale)//2*2),max(2,int(h*scale)//2*2)


def fit_video(image, size):
    # 녹화 도중 창 크기를 바꿔도 처음 정한 영상 크기 안에 등비 배치한다.
    w,h=size;ih,iw=image.shape[:2]
    if (iw,ih)==size:return image
    scale=min(w/iw,h/ih);nw,nh=max(1,round(iw*scale)),max(1,round(ih*scale))
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    out=np.zeros((h,w,3),np.uint8)
    x,y=(w-nw)//2,(h-nh)//2;out[y:y+nh,x:x+nw]=resized
    return out


class VideoRecorder:
    def __init__(self, root, settings=None):
        cfg=settings or {}
        self.root=Path(root);self.fps=float(cfg.get("fps",15))
        self.max_width=int(cfg.get("max_width",1920));self.max_height=int(cfg.get("max_height",1080))
        self.condition=threading.Condition();self.thread=None
        self.state="idle";self.message="";self.message_until=0.
        self.started=0.;self.stopped=None;self.latest=None;self.previous=None
        self.written=0;self.path=None;self.pending=None;self.last_path=None

    def toggle(self, image):
        with self.condition:
            if self.state in ("starting","recording"):
                self._stop_locked()
                return "Recording finishing..."
            if self.thread is not None and self.thread.is_alive():
                return "Previous recording is still finishing; wait before pressing R again."
            if image is None:
                return "No dashboard available to record."
            self.started=time.monotonic();self.stopped=None
            self.latest=(self.started,image);self.previous=None
            self.written=0;self.path=None;self.pending=None
            self.state="starting";self.message="";self.message_until=0.
            self.thread=threading.Thread(target=self._run,args=(image,),name="Dashboard-MP4",daemon=True)
            self.thread.start()
            return "Recording started. R: stop and save MP4."

    def _stop_locked(self):
        if self.state in ("starting","recording"):
            self.stopped=time.monotonic();self.state="finishing"
            self.condition.notify_all()

    def stop(self):
        with self.condition:self._stop_locked()

    def submit(self, image):
        # dashboard.render가 새 배열을 반환하므로 전달 후 배열을 변경하지 않는다.
        # [追加] フレームを大量コピー/蓄積せず最新2枚の参照だけ置き換える。
        with self.condition:
            if self.state not in ("starting","recording"):return
            self.previous=self.latest;self.latest=(time.monotonic(),image)
            self.condition.notify_all()

    def decorate(self, image):
        # 녹화 상태는 실행 창에만 표시한다. 원본 화면을 변경하지 않아 MP4에 섞이지 않는다.
        with self.condition:
            state=self.state;elapsed=max(0,(self.stopped or time.monotonic())-self.started)
            message=self.message if time.monotonic()<self.message_until or state=="error" else ""
        if state in ("starting","recording"):
            seconds=int(elapsed);label=f"REC {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d} | R: stop"
            color=(60,60,255)
        elif state=="finishing":label="Saving MP4...";color=(0,210,255)
        elif message:label=message;color=(60,60,255) if state=="error" else (90,230,130)
        else:return image
        # [추가] 상태 표시가 필요할 때만 화면용 복사본을 만든다.
        image=image.copy()
        h,w=image.shape[:2];font=cv2.FONT_HERSHEY_SIMPLEX
        scale=min(.56,max(.25,(w-24)/max(1,cv2.getTextSize(label,font,1.,1)[0][0])))
        tw,th=cv2.getTextSize(label,font,scale,1)[0];x=max(6,w-tw-18)
        cv2.rectangle(image,(x-6,5),(w-6,th+19),(20,20,20),-1)
        cv2.putText(image,label,(x,th+12),font,scale,color,1,cv2.LINE_AA)
        return image

    def _reserve(self):
        self.root.mkdir(parents=True,exist_ok=True)
        # 마이크로초와 배타적 예약으로 기존 사진 폴더/동영상을 덮어쓰지 않는다.
        stem="recording_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for number in range(10000):
            suffix="" if number==0 else f"_{number}"
            final=self.root/(stem+suffix+".mp4")
            pending=self.root/("."+stem+suffix+".pending.mp4")
            if final.exists():continue
            try:
                with pending.open("xb"):pass
            except FileExistsError:continue
            return final,pending
        raise RuntimeError("Cannot reserve a unique recording filename")

    def _run(self, initial):
        writer=None
        try:
            final,pending=self._reserve()
            with self.condition:
                self.path=final;self.pending=pending;source=initial
            size=video_size(source,self.max_width,self.max_height)
            # 동봉 OpenCV의 MP4V 인코더 사용. 별도 프로그램이나 패키지 설치가 필요 없다.
            writer=cv2.VideoWriter(str(pending),cv2.VideoWriter_fourcc(*"mp4v"),self.fps,size)
            if not writer.isOpened():raise RuntimeError("MP4 encoder could not open the output file")
            with self.condition:
                if self.stopped is None:self.state="recording"
            # 첫 코덱/DLL 준비 시간은 느린 지속 인코딩과 구분한다.
            lag_limit=max(2.,time.monotonic()-self.started+.5)
            cached_source=None;prepared=None;count=0
            while True:
                with self.condition:
                    end=self.stopped
                    # 실제 경과시간으로 영상 프레임 수를 정한다. UI/SCNN이 느릴 때 이전 화면을
                    # 반복해 빨리 감기처럼 저장되지 않게 한다. 길이 오차는 최대 1프레임.
                    if end is not None and count>=max(1,math.ceil((end-self.started)*self.fps)):
                        break
                    due=self.started+count/self.fps
                    delay=due-time.monotonic()
                    if delay>0:
                        self.condition.wait(delay);continue
                    if time.monotonic()-due>lag_limit:
                        raise RuntimeError("MP4 encoding cannot keep up; lower recording.fps or resolution")
                    # 예정 시각보다 나중에 들어온 화면을 과거 프레임에 넣지 않는다.
                    for candidate in (self.latest,self.previous):
                        if candidate is not None and candidate[0]<=due:
                            source=candidate[1];break
                if source is not cached_source:
                    prepared=fit_video(source,size);cached_source=source
                writer.write(prepared);count+=1
                with self.condition:self.written=count
            writer.release();writer=None
            # release 후 다시 열어 확인한다. 미완료 파일은 완성 MP4로 표시하지 않는다.
            capture=cv2.VideoCapture(str(pending))
            try:
                readable,frame=capture.read()
                frames=int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                if not readable or frame.shape[1::-1]!=size or frames!=count:
                    raise RuntimeError(f"MP4 verification failed: encoded={count}, readable={frames}")
            finally:capture.release()
            pending.rename(final)
            with self.condition:
                self.last_path=final;self.state="idle"
                self.message="Saved: "+final.name;self.message_until=time.monotonic()+6
            print(f"MP4 saved: {final} | {size[0]}x{size[1]} | {count/self.fps:.2f}s | {self.fps:g} FPS")
        except Exception as exc:
            if writer is not None:
                try:writer.release()
                except Exception:pass
            with self.condition:
                self.state="error";self.message="REC ERROR - see console"
            print(f"MP4 recording failed: {exc} | unfinished file: {self.pending}")
        finally:
            with self.condition:
                self.latest=None;self.previous=None;self.condition.notify_all()

    def close(self):
        # Q/Esc/창 닫기/예외 종료 때도 정지 시각을 고정하고 MP4를 마무리한다.
        self.stop()
        thread=self.thread
        if thread is not None:
            thread.join(timeout=10)
            if thread.is_alive():
                print(f"MP4 finalization timed out; unfinished file retained: {self.pending}")
