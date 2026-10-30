"""[v3.5 추가] 동일 프레임 SCNN을 보정값으로 native Depth 점에 대응시키는 융합.
SCNN은 깊이 없는 점을 만들지 않는다. 명확한 기하 장애물은 SCNN 바닥 예측으로 지우지 않는다.
"""
import numpy as np
import cv2
from .floor import fit_floor

def sample_semantics(semantic,frame_id,shape,uv,color_ok,cfg):
    count=len(uv)
    labels=np.full(count,255,np.uint8);confidence=np.zeros(count,np.float32)
    info={"active":False,"reason":"model unavailable or fusion disabled","mapped_points":0}
    if not cfg["enabled"] or semantic is None or semantic.status!="OK":
        return labels,confidence,info
    if semantic.frame_id!=frame_id:
        info["reason"]="semantic frame mismatch";return labels,confidence,info
    mask=semantic.labels;score=getattr(semantic,"confidence",None)
    if not isinstance(mask,np.ndarray) or mask.shape!=shape or mask.dtype!=np.uint8 or not np.isin(mask,[0,1,2]).all():
        info["reason"]="invalid semantic labels";return labels,confidence,info
    if not isinstance(score,np.ndarray) or score.shape!=shape or not np.isfinite(score).all() or np.any((score<0)|(score>1)):
        info["reason"]="valid softmax confidence unavailable";return labels,confidence,info
    # [v3.5 추가] z-buffer에서 보이는 Color 대응만 사용. 경계 1픽셀은 클래스 혼합을 피하도록 제외.
    interior=np.ones(shape,bool)
    radius=cfg["boundary_erode_px"]
    if radius:
        interior=np.zeros(shape,bool);kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
        for ident in (0,1,2):
            interior |= cv2.erode((mask==ident).astype(np.uint8),kernel,
                                  borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    ids=np.flatnonzero(color_ok)
    if len(ids):
        pixels=np.rint(uv[ids]).astype(np.int32)
        inside=(pixels[:,0]>=0)&(pixels[:,0]<shape[1])&(pixels[:,1]>=0)&(pixels[:,1]<shape[0])
        ids=ids[inside];pixels=pixels[inside];u,v=pixels.T
        ids=ids[interior[v,u]];u,v=np.rint(uv[ids]).astype(np.int32).T
        labels[ids]=mask[v,u];confidence[ids]=score[v,u]
    info.update(active=True,reason="same-frame calibrated semantic-depth fusion",mapped_points=int((labels!=255).sum()))
    return labels,confidence,info

def fit_fused_floor(points,labels,confidence,floor_cfg,cfg,active):
    # [v3.5 추가] 의미론적 바닥 후보에도 기존 높이/기울기/면적/RANSAC 검증을 모두 적용.
    strong=(labels==0)&(confidence>=cfg["floor_confidence"])
    if active and cfg["use_floor_seeds"] and strong.sum()>=floor_cfg["min_points"]:
        plane=fit_floor(points[strong],floor_cfg)
        if plane.ok:
            return plane,"SCNN floor seeds + depth plane"
    return fit_floor(points,floor_cfg),"depth plane fallback"

def classify_points(points,plane,labels,confidence,config,active):
    f=config["fusion"];g=config["grid"];n=np.asarray(plane.normal)
    height=(points@n+plane.offset_m)/n[2]
    geo_floor=np.abs(height)<=config["floor"]["inlier_distance_m"]
    geo_obstacle=(height>=g["obstacle_min_height_m"])&(height<=g["obstacle_max_height_m"])
    sf=(labels==0)&(confidence>=f["floor_confidence"])&active
    so=(labels==1)&(confidence>=f["obstacle_confidence"])&active
    other=(labels==2)&(confidence>=f["obstacle_confidence"])&active
    # [v3.5] 사용자 지정 5cm 기준은 Depth/SCNN 공통으로 적용한다.
    # SCNN이 장애물이어도 3~5cm 점을 별도 장애물로 추가하던 예외는 없앤다.
    low_obstacle=np.zeros(len(points),bool)
    extended=sf&(np.abs(height)<=f["semantic_floor_tolerance_m"])&~geo_obstacle
    conflict=geo_floor&(so|other)
    floor=(geo_floor&~conflict)|extended
    floor &= ~geo_obstacle & ~low_obstacle
    # [v3.5] 높이 판정은 그대로 둔다. 같은 시각/보정된 RGB 위치에서 SCNN이
    # 강하게 바닥이라고 판단한 점만 충돌로 보류. 바닥으로 재라벨링하지 않는다.
    held=(geo_obstacle&(labels==0)&(confidence>=f["floor_obstacle_confidence"])&
          active&f["defer_floor_obstacle_conflict"])
    return dict(height=height,floor=floor,semantic_floor=floor&sf,depth_semantic_conflict=held,
                geometry_obstacle=geo_obstacle,semantic_obstacle=low_obstacle,conflict=conflict)
