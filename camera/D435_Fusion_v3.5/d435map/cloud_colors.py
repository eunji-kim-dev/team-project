"""[v3.5] 실제 카메라 거리로 표시 색을 만든다. 기하/Costmap 및 원본 RGB는 변경하지 않음."""
from functools import lru_cache
import numpy as np
import cv2

@lru_cache(maxsize=1)
def rainbow_lut():
    # 색상 팔레트를 바꾸려면 이 상수를 수정. RAINBOW: 가까운 빨강 → 먼 보라.
    return cv2.applyColorMap(np.arange(256,dtype=np.uint8),cv2.COLORMAP_RAINBOW).reshape(256,3)

def distance_colors(points,camera_position,settings):
    # [v3.5] Low Angle/High Angle 가상 시점이 아니라 보정된 실제 센서 중심 기준 3D 직선거리(m).
    distances=np.linalg.norm(np.asarray(points)-np.asarray(camera_position),axis=1)
    if settings["mode"]=="red":
        return np.tile(np.array([20,20,245],np.uint8),(len(points),1)),distances
    lo,hi=settings["min_distance_m"],settings["max_distance_m"]
    bins=np.rint(np.clip((distances-lo)/(hi-lo),0,1)*255).astype(np.uint8)
    return rainbow_lut()[bins],distances

def prepare_obstacle_cloud(snapshot,cfg):
    # [v3.5] Low Angle/High Angle 두 화면에서 선택·거리·색상 계산을 공유. 실제 계산/저장 점군은 축소하지 않음.
    g=cfg["grid"];setting=cfg["pointcloud_color"]
    ids=np.flatnonzero(snapshot.point_classes==1)
    points=snapshot.points[ids]
    inside=(points[:,0]>=g["x_min_m"])&(points[:,0]<g["x_max_m"])&(points[:,1]>=g["y_min_m"])&(points[:,1]<g["y_max_m"])
    points=points[inside]
    maximum=setting["display_max_points"]
    if len(points)>maximum:points=points[np.linspace(0,len(points)-1,maximum,dtype=int)]
    colors,distances=distance_colors(points,snapshot.transform[:3,3],setting)
    return points,colors,distances
