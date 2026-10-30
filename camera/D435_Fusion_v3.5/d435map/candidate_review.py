"""[v3.5] 작은 저고도 후보만 국소 바닥 + 단기 증거로 재검사. 5cm 기준은 유지."""
import time
import numpy as np
import cv2
from .obstacle_support import supported_obstacles
from .temporal import MotionGuard,CandidateHistory

def local_floor_model(floor_points,center,plane,cfg):
    """후보 자체를 제외한 주변 측정 바닥만 robust least squares로 적합."""
    dxy=floor_points[:,:2]-center[:2]
    dist2=np.einsum("ij,ij->i",dxy,dxy)
    choose=(dist2<cfg["local_radius_m"]**2)&(dist2>cfg["local_inner_radius_m"]**2)
    p=floor_points[choose]
    info={"support_points":len(p),"trusted":False}
    if len(p)<cfg["local_min_points"]:return None,info
    if len(p)>cfg["local_max_samples"]:p=p[np.linspace(0,len(p)-1,cfg["local_max_samples"],dtype=int)]
    xy=p[:,:2]-center[:2]
    if np.any(np.ptp(xy,axis=0)<cfg["local_min_extent_m"]):return None,info
    sectors=(xy[:,0]>=0).astype(int)*2+(xy[:,1]>=0)
    if len(np.unique(sectors))<cfg["local_min_sectors"]:return None,info
    A=np.column_stack((xy,np.ones(len(p))));z=p[:,2];good=np.ones(len(p),bool)
    for _ in range(4):
        if good.sum()<cfg["local_min_points"]:return None,info
        abc=np.linalg.lstsq(A[good],z[good],rcond=None)[0]
        residual=z-A@abc;median=np.median(residual[good])
        mad=1.4826*np.median(np.abs(residual[good]-median))
        good=np.abs(residual-median)<max(.006,2.5*mad)
    if good.sum()<cfg["local_min_points"] or good.mean()<cfg["local_min_inlier_ratio"]:return None,info
    abc=np.linalg.lstsq(A[good],z[good],rcond=None)[0]
    residual=z[good]-A[good]@abc
    rmse=float(np.sqrt(np.mean(residual**2)))
    tilt=float(np.rad2deg(np.arctan(np.linalg.norm(abc[:2]))))
    n=np.asarray(plane.normal)
    global_z=-(center[:2]@n[:2]+plane.offset_m)/n[2]
    shift=float(abs(abc[2]-global_z))
    # 바닥 지지가 한쪽에만 남거나 경사가 불안정하면 국소 판정 자체를 보류.
    xy_good=xy[good]
    coverage=len(np.unique((xy_good[:,0]>=0).astype(int)*2+(xy_good[:,1]>=0)))
    trusted=(rmse<=cfg["local_max_rmse_m"] and tilt<=cfg["local_max_tilt_deg"]
             and shift<=cfg["local_max_shift_m"] and coverage>=cfg["local_min_sectors"]
             and np.all(np.ptp(xy_good,axis=0)>=cfg["local_min_extent_m"])
             and np.linalg.eigvalsh(np.cov(xy_good.T)).min()>1e-5)
    info.update(trusted=bool(trusted),inliers=int(good.sum()),rmse_m=rmse,tilt_deg=tilt,shift_m=shift)
    return (abc if trusted else None),info

class CandidateReviewer:
    def __init__(self,cfg):
        self.cfg=cfg;self.motion=MotionGuard(cfg["temporal_review"]);self.history=CandidateHistory(cfg["temporal_review"])
    def reset(self):
        self.motion.reset();self.history.reset()
    def process(self,points,indices,depth,obstacle,on_floor,labels,confidence,plane,bgr,stamp,signature):
        started=time.perf_counter();cfg=self.cfg;c=cfg["candidate_review"];t=cfg["temporal_review"]
        accepted=obstacle.copy();local_held=np.zeros(len(points),bool);temporal_held=local_held.copy()
        diag={"enabled":c["enabled"],"components_reviewed":0,"local_held_points":0,"temporal_held_points":0,
              "confirmed_small_components":0,"reviewed_components":[]}
        if not c["enabled"]:
            self.reset();return accepted,local_held,temporal_held,diag
        motion=self.motion.check(bgr,depth,stamp,signature) if t["enabled"] else {"stable":True,"reason":"temporal_disabled"}
        diag["motion"]=motion
        native=np.zeros(depth.shape,np.uint8);native.ravel()[indices[obstacle]]=1
        count,cc,stats,_=cv2.connectedComponentsWithStats(native,8)
        component=cc.ravel()[indices]
        sizes=stats[:,cv2.CC_STAT_AREA]
        small_ids=np.flatnonzero((sizes<=c["max_component_points"])&(sizes>0))
        small_ids=small_ids[small_ids!=0]
        n=np.asarray(plane.normal);height=(points@n+plane.offset_m)/n[2]
        strong=(labels==1)&(confidence>=cfg["fusion"]["obstacle_confidence"])
        floor_points=points[on_floor&~obstacle]
        groups=[];centers=[];checks=0
        # [v3.5] 배열 정렬 한 번으로 후보별 전체 점군 탐색을 피한다.
        eligible=np.flatnonzero(obstacle&np.isin(component,small_ids))
        order=eligible[np.argsort(component[eligible],kind="stable")]
        split=np.flatnonzero(np.diff(component[order]))+1 if len(order) else []
        for ids in np.split(order,split):
            if not len(ids):continue
            p=points[ids];span=np.ptp(p,axis=0)
            if height[ids].max()>c["max_height_m"] or np.ptp(height[ids])>c["max_vertical_span_m"] or span[:2].max()>c["max_xy_span_m"]:continue
            if strong[ids].mean()>=c["semantic_strong_fraction"]:continue
            diag["components_reviewed"]+=1
            center=p.mean(0);detail={"points":len(ids),"center_xyz_m":center.tolist(),"height_range_m":[float(height[ids].min()),float(height[ids].max())]}
            remaining=ids
            if c["local_floor_enabled"] and checks<c["max_local_checks"]:
                checks+=1;abc,info=local_floor_model(floor_points,center,plane,c);detail["local_floor"]=info
                if abc is not None:
                    local_h=p[:,2]-((p[:,:2]-center[:2])@abc[:2]+abc[2])
                    above=local_h>=cfg["grid"]["obstacle_min_height_m"]
                    detail["local_height_range_m"]=[float(local_h.min()),float(local_h.max())]
                    local_held[ids[~above]]=True
                    remaining=ids[above]
                    if len(remaining):
                        candidates=np.zeros(len(points),bool);candidates[remaining]=True
                        supported,_,_=supported_obstacles(depth,indices,candidates,cfg["obstacle_support"])
                        local_held[remaining[~supported[remaining]]]=True;remaining=remaining[supported[remaining]]
            if len(remaining):
                groups.append(remaining);centers.append(points[remaining].mean(0))
            if len(diag["reviewed_components"])<32:diag["reviewed_components"].append(detail)
        if t["enabled"]:
            confirmed,streaks=self.history.update(centers,stamp,motion["stable"])
            for ids,ok in zip(groups,confirmed):
                if not ok:temporal_held[ids]=True
            diag["confirmed_small_components"]=int(confirmed.sum());diag["confirmation_streaks"]=streaks
        else:self.history.reset()
        accepted &= ~(local_held|temporal_held)
        diag.update(local_held_points=int(local_held.sum()),temporal_held_points=int(temporal_held.sum()),
                    local_checks=checks,elapsed_ms=(time.perf_counter()-started)*1000)
        return accepted,local_held,temporal_held,diag
