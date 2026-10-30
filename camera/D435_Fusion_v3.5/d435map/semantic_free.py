"""[v3.5] SCNN 바닥을 검증된 평면에 투영. 깊이/점군을 합성하지 않는다."""
import numpy as np
import cv2
from .geometry import project
from .costmap import camera_occluded


def connect_floor_gaps(seeds,eligible,resolution,max_distance_m):
    """확인된 바닥 주변의 짧은 빈틈만 연결. 가림/비바닥을 건너뛰지 않는다."""
    # [v3.5] 4방향 경로 길이를 제한하여 모서리 대각선으로 자유가 새지 않게 한다.
    steps=int(np.floor(max_distance_m/resolution+1e-7))
    grown=seeds.copy()
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    for _ in range(steps):
        extension=cv2.dilate(grown.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,
                             borderValue=0).astype(bool)&eligible&~grown
        if not extension.any():break
        grown |= extension
    return grown&~seeds


def projected_floor_free(grid,semantic,frame_id,plane,T,cal,raw,in_view,blocked,config,
                         measured_free=None,continuity_out=None):
    out=np.zeros(grid.shape,bool)
    if continuity_out is not None:continuity_out.fill(False)
    info={"enabled":config["fusion"].get("semantic_free_enabled",False),"candidate_cells":0,
          "continuity_candidates":0,"continuity_added_cells":0,
          "reason":"disabled or unavailable",
          "definition":"SCNN+plane inferred ground, not measured depth or full robot-height clearance"}
    f=config["fusion"]
    if not (info["enabled"] and f["enabled"] and plane.ok and config["grid"]["free_space_policy"]=="observed_floor"):
        return out,info
    if semantic is None or semantic.status!="OK" or semantic.frame_id!=frame_id:return out,info
    labels,score=semantic.labels,semantic.confidence
    expected=(cal["color_intrinsics"]["height"],cal["color_intrinsics"]["width"])
    if not isinstance(labels,np.ndarray) or labels.shape!=expected or labels.dtype!=np.uint8:return out,info
    if not isinstance(score,np.ndarray) or score.shape!=expected or not np.isfinite(score).all() or np.any((score<0)|(score>1)) or labels.max(initial=0)>2:return out,info
    # 기존 5/5 고신뢰도 투영 정책은 유지한다.
    interior=((labels==0)&(score>=f["semantic_free_confidence"])).astype(np.uint8)
    interior=cv2.erode(interior,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    cells=np.argwhere(in_view&~blocked)
    if not len(cells):return out,info
    xy=grid.xy[in_view&~blocked];ok=np.ones(len(cells),bool)
    continuity=f.get("semantic_free_continuity_enabled",False)
    votes=np.zeros(len(cells),np.uint8);depth_votes=votes.copy()
    relaxed=np.ones(len(cells),bool)
    n=np.asarray(plane.normal);half=grid.g["resolution_m"]*.49
    camera_height_above_plane=(n@T[:3,3]+plane.offset_m)/n[2]
    ext=cal["depth_to_color"];R=np.asarray(ext["rotation"]).reshape(3,3,order="F")
    translation=np.asarray(ext["translation_m"])
    for index,offset in enumerate(((0,0),(-half,-half),(-half,half),(half,-half),(half,half))):
        qxy=xy+offset
        z=-(qxy@n[:2]+plane.offset_m)/n[2]
        floor=np.column_stack((qxy,z))
        optical=(floor-T[:3,3])@T[:3,:3]
        color=optical@R.T+translation
        uv=project(color,cal["color_intrinsics"]);u,v=np.rint(uv).astype(np.int32).T
        within=(color[:,2]>0)&(u>=0)&(v>=0)&(u<expected[1])&(v<expected[0])
        su,sv=np.clip(u,0,expected[1]-1),np.clip(v,0,expected[0]-1)
        ok &= within&(interior[sv,su]>0)
        if continuity:
            # [v3.5] 연결 보완: 중심+모서리 모두 바닥 라벨, 중심 포함 3/5는 점수 기준 통과.
            # RGB 장애물/천장 라벨을 덮어 칠하지 않고 기존 바닥 내부 빈틈만 연결한다.
            trusted=(labels[sv,su]==0)&(score[sv,su]>=f["semantic_free_continuity_confidence"])
            relaxed &= within&(labels[sv,su]==0)
            if index==0:relaxed &= trusted
            votes += trusted
        dn=project(optical,cal["depth_intrinsics"]);du,dv=np.rint(dn).astype(np.int32).T
        native=(du>=0)&(dv>=0)&(du<raw.shape[1])&(dv<raw.shape[0])
        measured=raw[np.clip(dv,0,raw.shape[0]-1),np.clip(du,0,raw.shape[1]-1)]
        tolerance=np.maximum(.08,.01*optical[:,2]**2)
        valid_range=native&(optical[:,2]>=config["depth"]["min_m"])&(optical[:,2]<=config["depth"]["max_m"])
        agreement=(measured<=0)|(~np.isfinite(measured))|(np.abs(measured-optical[:,2])<=tolerance)
        ok &= valid_range&agreement
        if continuity:
            # [v3.5] 경사 시선에서는 작은 바닥 높이 오차가 큰 Z 차이로 변환된다.
            # 같은 광선의 측정점 높이 = 카메라의 평면 위 높이 * (1 - 측정Z/바닥Z).
            # 연결 단계는 실제 장애물 기준 5cm를 넘는 표면을 가림으로 거부하고,
            # 3/5 표본의 바닥 높이 일치 또는 결측(추정)을 요구한다.
            present=np.isfinite(measured)&(measured>0)
            sample_height=camera_height_above_plane*(1-measured/np.maximum(optical[:,2],1e-6))
            nearer=present&(sample_height>=config["grid"]["obstacle_min_height_m"])
            height_agreement=(~present)|(np.abs(sample_height)<=f["semantic_floor_tolerance_m"])
            relaxed &= valid_range&~nearer
            depth_votes += height_agreement
    out[cells[ok,0],cells[ok,1]]=True
    info["candidate_cells"]=int(out.sum())
    allowed=np.zeros(grid.shape,bool)
    if continuity:
        eligible=relaxed&(votes>=3)&(depth_votes>=3)
        allowed[cells[eligible,0],cells[eligible,1]]=True
    # 보류 후보와 장애물 뒤로 보완이 번지지 않게 현재 프레임 광선 가림을 검사한다.
    targets=out|allowed
    if continuity and measured_free is not None:targets |= measured_free&in_view&~blocked
    hidden=camera_occluded(blocked,grid.g["resolution_m"],(grid.g["x_min_m"],grid.g["y_min_m"]),T[:2,3],targets)
    strict_hidden=int((out&hidden).sum())
    out &= ~hidden
    if continuity:
        allowed &= ~hidden
        seeds=out.copy()
        if measured_free is not None:seeds |= measured_free&in_view&~blocked&~hidden
        added=connect_floor_gaps(seeds,allowed,grid.g["resolution_m"],
                                f["semantic_free_continuity_distance_m"])
        out |= added
        if continuity_out is not None:continuity_out[:]=added
        info.update(continuity_candidates=int(allowed.sum()),continuity_added_cells=int(added.sum()),
                    continuity_max_distance_m=f["semantic_free_continuity_distance_m"],
                    continuity_confidence=f["semantic_free_continuity_confidence"])
    info.update(reason="same-frame SCNN + validated plane + obstacle/uncertain veto",
                occluded_candidates=strict_hidden,eligible_cells=int(out.sum()))
    return out,info
