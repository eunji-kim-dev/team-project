"""[v3.5] 같은 장애물 점군을 Low Angle/High Angle 가상 시점에 투영. 원본 점군/Costmap을 변경하지 않는다."""
import numpy as np
import cv2
from .cloud_colors import prepare_obstacle_cloud,rainbow_lut

def projection_parameters(view,w,h,yaw_deg=0.,pitch_deg=None,zoom=1.):
    pitch=np.deg2rad(view["costmap_pitch_deg"] if pitch_deg is None else pitch_deg)
    yaw=np.deg2rad(yaw_deg)
    forward=np.array([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)])
    right=np.array([np.sin(yaw),-np.cos(yaw),0.])
    up=np.cross(right,forward)
    eye=np.array([-view["costmap_eye_back_m"]*np.cos(yaw),
                  -view["costmap_eye_back_m"]*np.sin(yaw),view["costmap_eye_height_m"]])
    focal=w/(2*np.tan(np.deg2rad(view["costmap_hfov_deg"])/2))*zoom
    return eye,np.stack((right,up,forward),axis=1),focal,w*.5,h*view.get("cloud_center_y_fraction",.5)

def project_prepared(points,parameters):
    eye,basis,focal,cx,cy=parameters
    q=(np.asarray(points)-eye)@basis
    safe=np.maximum(q[...,2],.001)
    return np.stack((cx+focal*q[...,0]/safe,cy-focal*q[...,1]/safe),-1),q[...,2]

def project_obstacles(points,view,w,h,yaw_deg=0.,pitch_deg=None,zoom=1.):
    return project_prepared(points,projection_parameters(view,w,h,yaw_deg,pitch_deg,zoom))

def obstacle_view(snapshot,cfg,w,h,yaw_deg=0.,pitch_deg=None,zoom=1.,view_name="low_angle",prepared=None):
    view=cfg[view_name];g=cfg["grid"]
    out=np.full((h,w,3),(31,25,20),np.uint8)
    # [v3.5 지연 개선] 격자선/점군마다 동일한 가상 카메라 삼각함수를 다시 계산하지 않는다.
    projection=projection_parameters(view,w,h,yaw_deg,pitch_deg,zoom)
    screen=lambda points:project_prepared(points,projection)
    # [v3.5 추가] 회색 선은 미터 기준선일 뿐이며, 바닥/미확인 영역의 채움이나 비용 색을 그리지 않는다.
    def line(a,b):
        uv,z=screen(np.array([a,b]))
        if np.all(z>.05):
            ends=[tuple(np.rint(p).astype(int)) for p in uv]
            ok,p0,p1=cv2.clipLine((0,0,w,h),*ends)
            if ok:cv2.line(out,p0,p1,(70,70,70),1)
    rows=[]
    for x in np.arange(max(1.,np.ceil(g["x_min_m"])),g["x_max_m"]+.001):
        line([x,g["y_min_m"],0],[x,g["y_max_m"],0])
        uv,z=screen(np.array([[x,0,0]]));xx,yy=np.rint(uv[0]).astype(int)
        if z[0]>.05 and 5<xx<w-40 and 30<yy<h-25 and all(abs(yy-y)>15 for y in rows):
            rows.append(yy)
            cv2.putText(out,f"{x:g}m",(xx+5,yy-3),cv2.FONT_HERSHEY_SIMPLEX,.36,(200,200,200),1,cv2.LINE_AA)
    for y in np.arange(np.ceil(g["y_min_m"]),g["y_max_m"]+.001):
        line([max(.01,g["x_min_m"]),y,0],[g["x_max_m"],y,0])
    points,colors,distances=prepare_obstacle_cloud(snapshot,cfg) if prepared is None else prepared
    setting=cfg["pointcloud_color"]
    show_range=setting["mode"]=="rainbow" and w>=240 and h>=100
    bw=min(180,w//3);bx=w-bw-10;by=8
    displayed_points=0
    if len(points):
        uv,z=screen(points);pixels=np.rint(uv).astype(int)
        keep=(z>.05)&(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>24)&(pixels[:,1]<h)
        # [표시 개수] 화면 밖/상단 설명/컬러바에 가려진 점은 표시 개수에서 제외한다.
        if show_range:
            keep &= ~((pixels[:,0]>=bx-4)&(pixels[:,0]<=w-3)&
                      (pixels[:,1]>=by-7)&(pixels[:,1]<=by+28))
        pixels,z,colors=pixels[keep],z[keep],colors[keep]
        if len(z):
            # [v3.5] 거리 색이 다르므로 가상 화면의 같은 픽셀에는 가장 앞쪽 점만 표시.
            linear=pixels[:,1]*w+pixels[:,0]
            if setting["mode"]=="red":
                out[pixels[:,1],pixels[:,0]]=(20,20,245)
                displayed_points=int(np.unique(linear).size)
            else:
                order=np.lexsort((z,linear))
                first=np.r_[True,np.diff(linear[order])!=0]
                chosen=order[first]
                # [표시 개수] 같은 화면 픽셀에 겹치는 점은 실제로 보이는 앞쪽 한 점만 센다.
                displayed_points=int(chosen.size)
                out[pixels[chosen,1],pixels[chosen,0]]=colors[chosen]
    pitch=view["costmap_pitch_deg"] if pitch_deg is None else pitch_deg
    # 글자 뒤의 점 때문에 설명이 읽히지 않는 것을 방지한다.
    cv2.rectangle(out,(0,0),(w-1,24),(22,22,22),-1)
    title=f"eye {view['costmap_eye_height_m']:.2f}m | back {view['costmap_eye_back_m']:.2f}m | pitch {pitch:+.0f}deg | {displayed_points} points"
    # 창이 작아져도 숫자 끝부분이 우측 컬러바와 겹치지 않도록 한 줄 폭을 맞춘다.
    title_width=cv2.getTextSize(title,cv2.FONT_HERSHEY_SIMPLEX,.40,1)[0][0]
    available=(bx-12 if show_range else w-16)
    title_size=.40*min(1.,max(1,available)/max(1,title_width))
    cv2.putText(out,title,(8,17),cv2.FONT_HERSHEY_SIMPLEX,title_size,(235,235,235),1,cv2.LINE_AA)
    # [화면 정리] Sensor Range 제목은 대시보드 제목 줄에 표시하고 컬러바는 바로 아래에 둔다.
    # 범례도 점군과 동일한 LUT/거리 설정 사용. 최대 거리 이상은 끝 보라색으로 포화.
    if show_range:
        gradient=cv2.resize(rainbow_lut()[None],(bw,9),interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(out,(bx-4,by-7),(w-3,by+28),(22,22,22),-1)
        out[by:by+9,bx:bx+bw]=gradient
        cv2.putText(out,f"{setting['min_distance_m']:g}m",(bx,by+23),cv2.FONT_HERSHEY_SIMPLEX,.33,(235,235,235),1,cv2.LINE_AA)
        far_label=f"{setting['max_distance_m']:g}m+"
        far_width=cv2.getTextSize(far_label,cv2.FONT_HERSHEY_SIMPLEX,.33,1)[0][0]
        cv2.putText(out,far_label,(bx+bw-far_width,by+23),cv2.FONT_HERSHEY_SIMPLEX,.33,(235,235,235),1,cv2.LINE_AA)
    # [v3.5 화면 정리] 하단 조작 안내를 숨기고 점군 표시 영역을 유지한다.
    if not snapshot.floor.ok:
        cv2.putText(out,"FLOOR FAILED: obstacle height classification unavailable",(8,48),
                    cv2.FONT_HERSHEY_SIMPLEX,.40,(60,170,255),1,cv2.LINE_AA)
    return out
