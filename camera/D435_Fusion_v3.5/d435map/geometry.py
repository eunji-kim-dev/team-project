"""원본 Depth 광학계 -> 고정 바닥 좌표계. 정렬 Z를 Color Z로 간주하지 않는다."""
from functools import lru_cache
import json
import numpy as np
import cv2

def camera_transform(config):
    # [v3.5 추가] 광학계 (오른쪽, 아래, 전방) -> 지도 (전방, 좌측, 위).
    # 설치 보정은 고정 바닥 축의 Rz(yaw) @ Ry(pitch) @ Rx(roll).
    # 양의 pitch는 전방을 아래로 기울이고 양의 yaw는 좌측으로 회전한다.
    a, b, c = np.deg2rad([config["roll_deg"], config["pitch_deg"], config["yaw_deg"]])
    Rx = np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    Ry = np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    Rz = np.array([[np.cos(c),-np.sin(c),0],[np.sin(c),np.cos(c),0],[0,0,1]])
    optical_to_base = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    T = np.eye(4)
    T[:3,:3] = Rz @ Ry @ Rx @ optical_to_base
    T[:3,3] = [0, 0, config["height_m"]]
    return T

def sdk_intrinsics(info):
    import pyrealsense2 as rs
    intr = rs.intrinsics()
    for name in ("width", "height", "fx", "fy", "ppx", "ppy"):
        setattr(intr, name, info[name])
    intr.model = getattr(rs.distortion, info["distortion_model"])
    intr.coeffs = info["coeffs"]
    return intr

@lru_cache(maxsize=8)
def _rays(key):
    intr = json.loads(key)
    h, w = intr["height"], intr["width"]
    v, u = np.mgrid[:h,:w]
    # [v3.5 추가] 현재 D435 자료의 왜곡계수는 모두 0. 핀홀식을 벡터화한다.
    if not np.any(intr["coeffs"]):
        return np.stack(((u-intr["ppx"])/intr["fx"], (v-intr["ppy"])/intr["fy"], np.ones_like(u)), -1).astype(np.float32)
    # [v3.5 추가] 비영 왜곡계수는 SDK의 역투영 결과를 한 번 캐시해 픽셀당 Python 호출을 매 프레임 반복하지 않는다.
    import pyrealsense2 as rs
    si = sdk_intrinsics(intr)
    rays = np.asarray([rs.rs2_deproject_pixel_to_point(si, [float(x),float(y)], 1.0)
                       for y in range(h) for x in range(w)], dtype=np.float32)
    return rays.reshape(h,w,3)

def rays(intr):
    if min(intr["fx"], intr["fy"]) <= 0:
        raise ValueError("non-positive focal length")
    return _rays(json.dumps(intr, sort_keys=True))

def project(points, intr):
    points = np.asarray(points)
    z = points[...,2]
    safe = np.where(z > 1e-8, z, 1.0)
    x, y = points[...,0]/safe, points[...,1]/safe
    k = np.asarray(intr["coeffs"])
    model = intr["distortion_model"]
    if np.any(k):
        if model == "brown_conrady":
            r2 = x*x+y*y
            radial = 1+k[0]*r2+k[1]*r2*r2+k[4]*r2*r2*r2
            xx = x*radial + 2*k[2]*x*y+k[3]*(r2+2*x*x)
            yy = y*radial + 2*k[3]*x*y+k[2]*(r2+2*y*y)
            x,y = xx,yy
        elif model == "modified_brown_conrady":
            r2=x*x+y*y
            radial=1+k[0]*r2+k[1]*r2*r2+k[4]*r2*r2*r2
            x,y=x*radial,y*radial
            x,y=x+2*k[2]*x*y+k[3]*(r2+2*x*x), y+2*k[3]*x*y+k[2]*(r2+2*y*y)
        else:
            # [주의] 지원하지 않는 왜곡을 핀홀로 무시하지 않고 SDK에 위임한다.
            import pyrealsense2 as rs
            si = sdk_intrinsics(intr)
            flat = points.reshape(-1,3)
            uv = np.asarray([rs.rs2_project_point_to_pixel(si, p.tolist()) if p[2]>0 else [-1,-1]
                             for p in flat],np.float32)
            return uv.reshape(*points.shape[:-1],2)
    return np.stack((x*intr["fx"]+intr["ppx"], y*intr["fy"]+intr["ppy"]),-1)

def filtered_depth(raw, scale, settings):
    # [v3.5 추가] 원본 배열은 변경하지 않으며 결측 보간/시간 필터는 사용하지 않는다.
    z = raw.astype(np.float32)*scale
    valid = np.isfinite(z)&(z>=settings["min_m"])&(z<=settings["max_m"])
    z[~valid]=0
    used=z.copy()
    if settings["bilateral"]:
        # 깊이 경계 양쪽을 섞는 강한 평활화를 막도록 sigma를 미터 단위로 지정.
        smooth=cv2.bilateralFilter(z,5,settings["bilateral_sigma_m"],2)
        acceptable=valid&(np.abs(smooth-z)<settings["bilateral_sigma_m"])
        used[acceptable]=smooth[acceptable]
        used[~valid]=0
    if settings["neighbor_outlier"]:
        count=np.zeros_like(raw,dtype=np.uint8)
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            neighbor=np.roll(np.roll(z,dy,axis=0),dx,axis=1)
            good=valid&(neighbor>0)&(np.abs(z-neighbor)<=settings["neighbor_delta_m"])
            if dy>0: good[:dy]=False
            if dy<0: good[dy:]=False
            if dx>0: good[:,:dx]=False
            if dx<0: good[:,dx:]=False
            count+=good
        used[count<settings["neighbor_min"]]=0
    return z,used,valid

def colorize(optical, color_bgr, calibration):
    # [v3.5 추가] 원본 Depth 점을 R,t로 Color 광학계에 옮긴 후 투영한다.
    ext=calibration["depth_to_color"]
    R=np.asarray(ext["rotation"]).reshape(3,3,order="F")
    pc=optical@R.T+np.asarray(ext["translation_m"])
    uv=project(pc,calibration["color_intrinsics"])
    h,w=color_bgr.shape[:2]
    u,v=np.rint(uv).astype(np.int32).T
    good=(u>=0)&(u<w)&(v>=0)&(v<h)&(pc[:,2]>0)
    idx=np.flatnonzero(good)
    # Color 시점에서 앞쪽 표면에 가려진 점에는 뒤쪽 색상을 복사하지 않는다.
    front=np.full(h*w,np.inf,np.float32)
    # [v3.5] 픽셀별 깊이를 한 번 묶어 최소값 계산. 반복 scatter 갱신을 줄인다.
    flat=v[idx]*w+u[idx]
    if len(idx):
        order=np.argsort(flat,kind="stable")
        ordered=flat[order]
        starts=np.r_[0,np.flatnonzero(np.diff(ordered))+1]
        front[ordered[starts]]=np.minimum.reduceat(pc[idx[order],2],starts)
    good[idx]&=pc[idx,2]<=front[v[idx]*w+u[idx]]+0.03
    rgb=np.full((len(optical),3),130,np.uint8)
    rgb[good]=color_bgr[v[good],u[good],::-1]
    return rgb, uv, good, pc[:,2]

def voxel_indices(points, size):
    # 지도 판정 이전이 아닌 표시/PLY 단계에만 기본 적용하여 가는 다리를 보존한다.
    if size<=0 or len(points)==0: return np.arange(len(points))
    cells=np.floor(points/size).astype(np.int32)
    return np.unique(cells,axis=0,return_index=True)[1]
