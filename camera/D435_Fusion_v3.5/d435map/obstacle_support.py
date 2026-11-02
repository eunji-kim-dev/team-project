"""동일 native Depth 프레임의 연결 지지. 판정 규칙 그대로, 후보 픽셀만 검사."""
import numpy as np

_RULE="8-neighbor native-depth support; retain endpoints adjoining strong support; no temporal history"

def _support_window(depth, pixels, cfg):
    """후보 픽셀(평면 인덱스)의 경계 상자(+1픽셀) 안에서 8이웃 지지를 dense 배열로 계산. 후보별 accepted(bool) 반환.
    [Jetson 속도] 이웃 관계는 후보끼리만 보므로 결과는 전체 영상 검사와 같다. 인덱스 리스트 gather 대신 슬라이스 연산을 쓴다."""
    h,w=depth.shape
    ys,xs=np.divmod(pixels,w)
    y0=max(int(ys.min())-1,0);y1=min(int(ys.max())+2,h);x0=max(int(xs.min())-1,0);x1=min(int(xs.max())+2,w)
    z=depth[y0:y1,x0:x1];sh,sw=z.shape
    native=np.zeros((sh,sw),bool);native[ys-y0,xs-x0]=True
    native &= np.isfinite(z)&(z>0)
    base=cfg["depth_jump_base_m"];per=cfg["depth_jump_per_m2"]
    count=np.zeros((sh,sw),np.uint8);links=[]
    # 실제 후보끼리의 4방향 무방향 간선을 양 끝점에 더해 8이웃 개수를 계산 (원래 규칙 그대로)
    for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
        ya0,ya1=0,sh-dy;xa0,xa1=max(0,-dx),sw-max(0,dx)
        if ya1<=ya0 or xa1<=xa0:continue
        a=(slice(ya0,ya1),slice(xa0,xa1));b=(slice(ya0+dy,ya1+dy),slice(xa0+dx,xa1+dx))
        za=z[a];zb=z[b]
        near=np.minimum(za,zb)
        limit=base+per*near**2
        connected=native[a]&native[b]&(np.abs(za-zb)<=limit)
        count[a]+=connected;count[b]+=connected
        links.append((a,b,connected))
    strong=native&(count>=cfg["min_neighbors"])
    accepted=strong.copy()
    # 끝점 복구는 원래의 strong만 참조. 반복 팽창하지 않는다.
    for a,b,connected in links:
        accepted[a]|=connected&strong[b]
        accepted[b]|=connected&strong[a]
    return accepted[ys-y0,xs-x0]

def supported_subset(depth, indices, ids, cfg):
    """점 번호 ids 만의 지지 판정 (후보 검토용). supported_obstacles(...)[0][ids] 와 같지만 전체 길이 배열을 만들지 않는다."""
    if not cfg["enabled"] or not len(ids):
        return np.ones(len(ids),bool)
    return _support_window(depth,indices[ids],cfg)

def supported_obstacles(depth, indices, candidates, cfg):
    if not cfg["enabled"]:
        return candidates.copy(),np.zeros_like(candidates),{"enabled":False,"candidate_points":int(candidates.sum()),"rejected_points":0}
    where=np.flatnonzero(candidates)
    keep=np.zeros_like(candidates)
    if len(where):
        keep[where]=_support_window(depth,indices[where],cfg)
    rejected=candidates&~keep
    return keep,rejected,dict(enabled=True,candidate_points=int(len(where)),
        accepted_points=int(keep.sum()),rejected_points=int(rejected.sum()),
        rule=_RULE,min_neighbors=cfg["min_neighbors"],depth_jump_base_m=cfg["depth_jump_base_m"],
        depth_jump_per_m2=cfg["depth_jump_per_m2"])
