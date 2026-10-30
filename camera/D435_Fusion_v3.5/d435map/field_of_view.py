"""[v3.5 추가] 실제 native Depth 보정값으로 수평 시야와 지도 바깥 표시를 분리."""
import numpy as np
from .geometry import rays

def horizontal_view(grid,intr,T,height_max,plane=None):
    # D435라는 이름만으로 90도를 고정하지 않고 실제 스트림의 좌/우 경계 광선을 읽는다.
    ray=rays(intr)
    row=int(np.clip(round(intr["ppy"]),0,intr["height"]-1))
    edge=ray[row,[0,-1]]
    angles=np.arctan2(edge[:,0],edge[:,2])
    left,right=float(angles.min()),float(angles.max())
    xy=grid.xy.reshape(-1,2)
    half=grid.g["resolution_m"]*.5
    offsets=((0,0),(-half,-half),(-half,half),(half,-half),(half,half))
    visible=np.zeros(len(xy),bool)
    # 지면~로봇 높이 구간의 수평 시야 합집합. 근거리 수직 사각/거리 누락은 안쪽 회색으로 유지.
    heights=np.unique([0.,height_max*.5,height_max,float(np.clip(T[2,3],0,height_max))])
    # [v3.5 지연 개선] roll/pitch=0이면 높이가 수평 방위각에 영향을 주지 않는다.
    # 이 경우 같은 방위각을 네 높이에서 반복 계산할 필요가 없다. 기울어진 설치는 원래 검사 유지.
    if T[2,0]==0 and T[2,2]==0:heights=heights[:1]
    for dx,dy in offsets:
        q=xy+[dx,dy]
        base_z=np.zeros(len(q))
        if plane is not None and plane.ok:
            n=np.asarray(plane.normal);base_z=-(q@n[:2]+plane.offset_m)/n[2]
        for z in heights:
            world=np.column_stack((q,base_z+z))
            optical=(world-T[:3,3])@T[:3,:3]
            bearing=np.arctan2(optical[:,0],optical[:,2])
            visible |= (optical[:,2]>0)&(bearing>=left)&(bearing<=right)
    return visible.reshape(grid.shape),dict(native_depth_hfov_deg=float(np.rad2deg(right-left)),
        left_edge_deg=float(np.rad2deg(left)),right_edge_deg=float(np.rad2deg(right)),
        definition="native depth horizontal FOV; cell-center/corners across robot height; vertical blind spots remain inside unknown")
