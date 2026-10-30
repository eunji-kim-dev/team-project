"""[v3.5] 불확실한 작은 후보의 단기 확인. 지도 누적/과거 점군 복사는 하지 않는다."""
import cv2
import numpy as np

class MotionGuard:
    def __init__(self,cfg):
        self.cfg=cfg;self.reset()
    def reset(self):
        self.previous=None;self.stamp=None;self.signature=None
    def check(self,bgr,depth,stamp,signature):
        c=self.cfg
        gray=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(320,240),interpolation=cv2.INTER_AREA)
        small=cv2.resize(depth,(160,120),interpolation=cv2.INTER_NEAREST)
        previous=self.previous;last=self.stamp;sig=self.signature
        self.previous=(gray,small);self.stamp=stamp;self.signature=signature
        reason=None
        if previous is None:reason="first_frame"
        elif signature!=sig:reason="calibration_changed"
        elif stamp<=last:reason="non_monotonic_time"
        elif stamp-last>c["max_gap_s"]:reason="frame_gap"
        if reason:return dict(stable=False,reason=reason)
        old,old_depth=previous
        valid=np.isfinite(small)&np.isfinite(old_depth)&(small>0)&(old_depth>0)
        if not valid.any():return dict(stable=False,reason="depth_unverified",depth_changed_fraction=None,depth_signed_shift_m=None,tracks=0)
        changed=float(np.mean(np.abs(small[valid]-old_depth[valid])>c["motion_depth_delta_m"]))
        signed_shift=float(np.median(small[valid]-old_depth[valid]))
        # [v3.5] 원거리 깊이의 무작위 흔들림을 이동으로 오인하지 않도록
        # 다수 픽셀의 변화와 같은 방향의 중앙값 이동이 함께 있을 때만 Depth 이동 증거로 사용.
        coherent_depth_change=changed>=c["motion_depth_fraction"] and abs(signed_shift)>=c["motion_depth_delta_m"]
        diag=dict(stable=False,reason="motion_unverified",depth_changed_fraction=changed,depth_signed_shift_m=signed_shift,tracks=0)
        features=cv2.goodFeaturesToTrack(old,maxCorners=240,qualityLevel=.01,minDistance=8,blockSize=5)
        if features is None or len(features)<c["motion_min_tracks"]:return diag
        try:
            new,status,_=cv2.calcOpticalFlowPyrLK(old,gray,features,None,winSize=(21,21),maxLevel=3)
            if new is None:return diag
            back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,old,new,None,winSize=(21,21),maxLevel=3)
            if back is None:return diag
            ok=(status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm(back-features,axis=2).ravel()<1.)
            a=features.reshape(-1,2)[ok];b=new.reshape(-1,2)[ok];diag["tracks"]=len(a)
            if len(a)<c["motion_min_tracks"]:return diag
            M,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.,maxIters=300,confidence=.99)
            if M is None:return diag
            keep=inliers.ravel()>0;ratio=float(keep.mean())
            flow=float(np.median(np.linalg.norm(b[keep]-a[keep],axis=1)))
            angle=float(abs(np.rad2deg(np.arctan2(M[1,0],M[0,0]))))
            scale=float(abs(np.hypot(M[0,0],M[1,0])-1))
            # 넓게 분포한 배경 특징만 정지 근거로 인정. 움직이는 작은 물체만 추적하지 않는다.
            tiles=np.floor(a[keep]/[80,60]).astype(int)
            coverage=len(np.unique(tiles,axis=0))
            stable=(keep.sum()>=c["motion_min_tracks"] and ratio>=c["motion_min_inlier_ratio"] and coverage>=4
                    and flow<c["motion_flow_px"] and angle<c["motion_rotation_deg"]
                    and scale<c["motion_scale_delta"] and not coherent_depth_change)
            diag.update(stable=bool(stable),reason="stable" if stable else "motion_or_scene_change",
                        flow_px=flow,rotation_deg=angle,scale_delta=scale,inlier_ratio=ratio,background_tiles=coverage)
            return diag
        except cv2.error:
            return diag

class CandidateHistory:
    def __init__(self,cfg):
        self.cfg=cfg;self.reset()
    def reset(self):
        self.tracks=[];self.stamp=None
    def update(self,centers,stamp,stable):
        c=self.cfg
        if not stable or self.stamp is None or stamp<=self.stamp or stamp-self.stamp>c["max_gap_s"]:
            self.tracks=[]
        self.stamp=stamp
        current=[];counts=[];used=set()
        for center in centers:
            count=1;best=None;best_distance=float("inf")
            for i,(old,hits) in enumerate(self.tracks):
                if i in used:continue
                xy=np.linalg.norm(center[:2]-old[:2]);z=abs(center[2]-old[2])
                if xy<=c["match_xy_m"] and z<=c["match_z_m"] and xy+z<best_distance:
                    best=i;best_distance=xy+z
            if best is not None:
                used.add(best);count=min(self.tracks[best][1]+1,c["confirm_frames"])
            current.append((np.asarray(center).copy(),count));counts.append(count)
        # [v3.5] 이번 프레임에 없는 후보는 즉시 기록에서 제거. 과거 장애물을 지도에 남기지 않음.
        self.tracks=current
        return np.asarray(counts,dtype=np.int32)>=c["confirm_frames"],counts
