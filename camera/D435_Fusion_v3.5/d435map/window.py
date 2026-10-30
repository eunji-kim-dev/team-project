"""[v3.5] OpenCV 창 크기와 대시보드 출력 크기를 동기화하여 비등방 확대를 방지."""
import cv2

def create_window(name,cfg):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(name,cfg["view"]["width"],cfg["view"]["height"])

def window_size(name,cfg):
    fallback=(cfg["view"]["width"],cfg["view"]["height"])
    try:
        _,_,w,h=cv2.getWindowImageRect(name)
        # 최소화/생성 직후에는 0 크기가 가능하므로 마지막 기본 크기로 표시를 유지한다.
        if w>0 and h>0:return int(w),int(h)
    except cv2.error:
        pass
    return fallback
