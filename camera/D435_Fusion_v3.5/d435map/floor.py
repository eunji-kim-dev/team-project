"""설치 높이/방향으로 제한한 RANSAC. 최대 평면을 무조건 바닥으로 고르지 않는다."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class FloorResult:
    ok: bool
    reason: str
    normal: list
    offset_m: float
    inliers: int = 0
    candidate_count: int = 0
    rmse_m: float = 0.0
    tilt_deg: float = 0.0
    height_at_origin_m: float = 0.0
    def record(self):
        return asdict(self)

def fit_floor(points, cfg):
    failure=lambda reason: FloorResult(False,reason,[0.,0.,1.],0.)
    # [v3.5 추가] 설치값으로 변환한 후 예상 바닥 부근/카메라 앞쪽에서만 후보를 찾는다.
    candidates=points[(np.abs(points[:,2])<cfg["seed_height_tolerance_m"]) &
                      (points[:,0]>.25)&(points[:,0]<5.8)]
    if len(candidates)<cfg["min_points"]:
        return failure("too few points near expected floor")
    rng=np.random.default_rng(12345)
    if len(candidates)>cfg["max_samples"]:
        candidates=candidates[rng.choice(len(candidates),cfg["max_samples"],replace=False)]
    cos_limit=np.cos(np.deg2rad(cfg["max_tilt_deg"]))
    best=None; score=0
    for _ in range(cfg["ransac_iterations"]):
        a,b,c=candidates[rng.choice(len(candidates),3,replace=False)]
        n=np.cross(b-a,c-a); norm=np.linalg.norm(n)
        if norm<1e-8: continue
        n=n/norm
        if n[2]<0: n=-n
        d=-float(n@a)
        if n[2]<cos_limit or abs(d/n[2])>cfg["max_height_error_m"]: continue
        inlier=np.abs(candidates@n+d)<cfg["inlier_distance_m"]
        if int(inlier.sum())>score: best=inlier; score=int(inlier.sum())
    if best is None or score<cfg["min_points"] or score/len(candidates)<cfg["min_ratio"]:
        return failure("no floor plane satisfying support / pose constraints")
    for _ in range(2):
        selected=candidates[best]
        # 거의 수평인 평면으로 제한했으므로 z=ax+by+c의 최소제곱으로 정밀화한다.
        abc=np.linalg.lstsq(np.column_stack((selected[:,:2],np.ones(len(selected)))),
                            selected[:,2],rcond=None)[0]
        n=np.array([-abc[0],-abc[1],1.]); length=np.linalg.norm(n); n/=length
        d=-abc[2]/length
        best=np.abs(candidates@n+d)<cfg["inlier_distance_m"]
    selected=candidates[best]
    if len(selected)<cfg["min_points"] or len(selected)/len(candidates)<cfg["min_ratio"]:
        return failure("floor support lost in refinement")
    if n[2]<cos_limit or abs(d/n[2])>cfg["max_height_error_m"]:
        return failure("refined floor inconsistent with configured pose")
    extent=np.ptp(selected[:,:2],axis=0)
    if extent[0]<cfg["min_extent_x_m"] or extent[1]<cfg["min_extent_y_m"]:
        return failure("floor support too narrow")
    residual=selected@n+d
    return FloorResult(True,"OK",n.tolist(),float(d),len(selected),len(candidates),
                       float(np.sqrt(np.mean(residual**2))),float(np.rad2deg(np.arccos(n[2]))),
                       float(-d/n[2]))
