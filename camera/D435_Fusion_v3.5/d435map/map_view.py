"""[v3.5 추가] 낮은 가상 시점의 지도. 비용은 Z=0 평면, 빨간 점의 높이는 실제 Depth 측정값."""
from functools import lru_cache
import numpy as np
import cv2

def basis(view):
    pitch=np.deg2rad(view["costmap_pitch_deg"])
    eye=np.array([-view["costmap_eye_back_m"],0.,view["costmap_eye_height_m"]])
    forward=np.array([np.cos(pitch),0.,np.sin(pitch)])
    right=np.array([0.,-1.,0.])
    up=np.cross(right,forward)
    return eye,np.stack((right,up,forward),axis=1)

def screen_points(points,view,w,h):
    eye,B=basis(view)
    q=(np.asarray(points)-eye)@B
    focal=w/(2*np.tan(np.deg2rad(view["costmap_hfov_deg"])/2))
    safe=np.maximum(q[...,2],.001)
    uv=np.stack((w*.5+focal*q[...,0]/safe,h*.28-focal*q[...,1]/safe),axis=-1)
    return uv,q[...,2]

@lru_cache(maxsize=8)
def ground_lookup(w,h,eye_height,eye_back,pitch,hfov,xmin,xmax,ymin,ymax,res):
    # [v3.5 추가] 가상 카메라 픽셀의 광선과 지도 평면 Z=0의 교차. 크기/시점이 같으면 캐시한다.
    view=dict(costmap_eye_height_m=eye_height,costmap_eye_back_m=eye_back,
              costmap_pitch_deg=pitch,costmap_hfov_deg=hfov)
    eye,B=basis(view);focal=w/(2*np.tan(np.deg2rad(hfov)/2))
    v,u=np.mgrid[:h,:w]
    rays=np.stack(((u-w*.5)/focal,-(v-h*.28)/focal,np.ones((h,w))),-1)@B.T
    t=np.full((h,w),np.inf)
    np.divide(-eye[2],rays[:,:,2],out=t,where=rays[:,:,2]<-1e-8)
    finite=np.isfinite(t)
    safe=np.where(finite,t,0.)
    X=eye[0]+safe*rays[:,:,0];Y=eye[1]+safe*rays[:,:,1]
    visible=finite&(t>.05)&(X>=xmin)&(X<xmax)&(Y>=ymin)&(Y<ymax)
    i=np.floor((X-xmin)/res).astype(np.int32)
    j=np.floor((Y-ymin)/res).astype(np.int32)
    return visible,i,j,t

def ground_view(snapshot,cost,colors,cfg,w,h,show_obstacles=True):
    view=cfg["low_angle"];g=cfg["grid"]
    out=np.full((h,w,3),(31,25,20),np.uint8)
    visible,i,j,depth=ground_lookup(w,h,view["costmap_eye_height_m"],view["costmap_eye_back_m"],
        view["costmap_pitch_deg"],view["costmap_hfov_deg"],g["x_min_m"],g["x_max_m"],
        g["y_min_m"],g["y_max_m"],g["resolution_m"])
    out[visible]=colors[i[visible],j[visible]]
    # 1m 눈금. 실제 미확인 셀을 녹색으로 보간하지 않는다.
    def segment(a,b):
        uv,z=screen_points(np.array([a,b]),view,w,h)
        if np.all(z>.05):
            ends=[tuple(np.rint(p).astype(int)) for p in uv]
            ok,p0,p1=cv2.clipLine((0,0,w,h),*ends)
            if ok: cv2.line(out,p0,p1,(105,105,105),1)
    label_rows=[]
    for x in np.arange(max(1.,np.ceil(g["x_min_m"])),g["x_max_m"]+.001):
        segment([x,g["y_min_m"],0],[x,g["y_max_m"],0])
        uv,z=screen_points(np.array([[x,0,0]]),view,w,h)
        xx,yy=np.rint(uv[0]).astype(int)
        if z[0]>.05 and 4<xx<w-40 and 20<yy<h-5 and all(abs(yy-other)>15 for other in label_rows):
            label_rows.append(yy)
            cv2.putText(out,f"{x:g}m",(xx+5,yy-3),cv2.FONT_HERSHEY_SIMPLEX,.36,(240,240,240),1,cv2.LINE_AA)
    for y in np.arange(np.ceil(g["y_min_m"]),g["y_max_m"]+.001):
        segment([max(.01,g["x_min_m"]),y,0],[g["x_max_m"],y,0])
    # [v3.5 추가] 막대를 임의 높이로 세우지 않고, 같은 프레임의 실제 장애물 점만 표시한다.
    ids=np.flatnonzero(snapshot.point_classes==1) if show_obstacles else np.empty(0,dtype=int)
    if len(ids):
        p=snapshot.points[ids]
        ij=np.floor((p[:,:2]-[g["x_min_m"],g["y_min_m"]])/g["resolution_m"]).astype(int)
        inside=(ij[:,0]>=0)&(ij[:,0]<cost.shape[0])&(ij[:,1]>=0)&(ij[:,1]<cost.shape[1])
        ids=ids[inside];ij=ij[inside]
        ids=ids[cost[ij[:,0],ij[:,1]]==254]
        if len(ids)>12000: ids=ids[np.linspace(0,len(ids)-1,12000,dtype=int)]
        uv,z=screen_points(snapshot.points[ids],view,w,h)
        pixels=np.rint(uv).astype(int)
        keep=(z>.05)&(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        pixels,z=pixels[keep],z[keep]
        if len(z):
            linear=pixels[:,1]*w+pixels[:,0]
            order=np.argsort(z,kind="stable")
            _,first=np.unique(linear[order],return_index=True)
            pick=order[first];xy=pixels[pick];zz=z[pick]
            hit=zz<=depth[xy[:,1],xy[:,0]]+.015
            out[xy[hit,1],xy[hit,0]]=(20,20,245)
    title=f"Low Angle eye {view['costmap_eye_height_m']:.2f}m | pitch {view['costmap_pitch_deg']:+.0f}deg"
    cv2.putText(out,title,(8,17),cv2.FONT_HERSHEY_SIMPLEX,.40,(235,235,235),1,cv2.LINE_AA)
    cv2.putText(out,"+X forward | +Y left | costs on Z=0"+(" | measured obstacle heights" if show_obstacles else " | COSTMAP ONLY"),
                (8,h-7),cv2.FONT_HERSHEY_SIMPLEX,.35,(235,235,235),1,cv2.LINE_AA)
    return out
