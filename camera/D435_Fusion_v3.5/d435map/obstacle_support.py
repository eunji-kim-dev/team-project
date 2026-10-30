"""동일 native Depth 프레임의 연결 지지. 판정 규칙 그대로, 후보 픽셀만 검사."""
import numpy as np

def supported_obstacles(depth, indices, candidates, cfg):
    native=np.zeros(depth.shape,bool)
    native.ravel()[indices]=candidates
    native &= np.isfinite(depth)&(depth>0)
    if not cfg["enabled"]:
        return candidates.copy(),np.zeros_like(candidates),{"enabled":False,"candidate_points":int(candidates.sum()),"rejected_points":0}
    h,w=depth.shape;flat=native.ravel();z=depth.ravel()
    active=np.flatnonzero(flat);ys,xs=np.divmod(active,w)
    count=np.zeros(h*w,np.uint8);pairs=[]
    # [v3.5 지연 개선] 전체 영상의 8방향 실수 배열을 반복 생성하지 않는다.
    # 실제 후보끼리의 4방향 무방향 간선을 각각 양 끝점에 더해 동일한 8이웃 개수를 계산.
    for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
        inside=(ys+dy<h)&(xs+dx>=0)&(xs+dx<w)
        a=active[inside];b=a+dy*w+dx
        linked=flat[b];a=a[linked];b=b[linked]
        near=np.minimum(z[a],z[b])
        limit=cfg["depth_jump_base_m"]+cfg["depth_jump_per_m2"]*near**2
        connected=np.abs(z[a]-z[b])<=limit
        a=a[connected];b=b[connected]
        count[a]+=1;count[b]+=1
        pairs.append((a,b))
    strong=flat&(count>=cfg["min_neighbors"])
    accepted=strong.copy()
    # 끝점 복구는 원래의 strong만 참조. 반복 팽창하지 않는다.
    for a,b in pairs:
        accepted[a] |= strong[b]
        accepted[b] |= strong[a]
    keep=accepted[indices]&candidates
    rejected=candidates&~keep
    return keep,rejected,dict(enabled=True,candidate_points=int(candidates.sum()),
        accepted_points=int(keep.sum()),rejected_points=int(rejected.sum()),
        rule="8-neighbor native-depth support; retain endpoints adjoining strong support; no temporal history",
        min_neighbors=cfg["min_neighbors"],depth_jump_base_m=cfg["depth_jump_base_m"],
        depth_jump_per_m2=cfg["depth_jump_per_m2"])
