"""OpenCV 단일 대시보드. 3D는 점을 가상 카메라에 투영하여 CPU로 렌더링한다."""
import time
import numpy as np
import cv2
from .geometry import voxel_indices
from .costmap import inflate,obstacle_risk,display_cost
from .obstacle_view import obstacle_view
from .robot_input import RobotWidthInput,RobotHeightInput
from .cloud_colors import prepare_obstacle_cloud

# [추가] 입력칸 제목을 바꾸려면 아래 두 문자열만 수정한다. 단위/계산에는 영향 없음.
ROBOT_WIDTH_LABEL = "Robot Width (m)"
ROBOT_HEIGHT_LABEL = "Robot Height (m)"

PALETTE_BGR=np.array([[0,255,0],[0,0,255],[255,0,0]],np.uint8)
LABEL_LUT_BGR=np.full((256,3),70,np.uint8)
LABEL_LUT_BGR[:3]=PALETTE_BGR

def text(image,line,xy,size=.48,color=(235,235,235)):
    cv2.putText(image,str(line),xy,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)

def classification_color(ids):
    # [v3.5 지연 개선] uint8 라벨 색 변환을 LUT 한 번으로 처리. 3..255는 기존 회색.
    if ids.dtype==np.uint8:return LABEL_LUT_BGR[ids]
    result=np.full((*ids.shape,3),70,np.uint8)
    good=ids<3
    result[good]=PALETTE_BGR[ids[good]]
    return result

def cost_color(cost,in_view=None,unknown=None):
    image=np.full((*cost.shape,3),(80,80,80),np.uint8)
    image[cost==0]=(0,255,0)  # [v3.5 추가] 관측 자유 공간을 선명한 녹색으로 표시
    shell=(cost>0)&(cost<253)
    image[shell]=np.stack((np.zeros(shell.sum()),255-cost[shell]*.22,np.full(shell.sum(),255)),axis=1).astype(np.uint8)
    image[cost==253]=(255,0,200)
    image[cost==254]=(0,0,255)
    # [v3.5] 위험 색을 유지하면서 미관측 부분만 가는 사선으로 구별한다.
    if unknown is not None:
        yy,xx=np.indices(cost.shape)
        hatch=np.asarray(unknown,dtype=bool)&(cost>0)&(cost<255)&((xx+yy)%9==0)
        image[hatch]=((image[hatch].astype(np.uint16)+np.array([150,150,150]))//2).astype(np.uint8)
    # [v3.5 추가] 시야 밖 검정은 255 미확인 비용과 다른 표시 마스크. 원본 비용 값은 보존.
    if in_view is not None:image[~np.asarray(in_view,dtype=bool)]=(0,0,0)
    return image

def fit_image(image,w,h):
    out=np.full((h,w,3),22,np.uint8)
    scale=min(w/image.shape[1],h/image.shape[0])
    nw,nh=max(1,int(image.shape[1]*scale)),max(1,int(image.shape[0]*scale))
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_NEAREST)
    x,y=(w-nw)//2,(h-nh)//2
    out[y:y+nh,x:x+nw]=resized
    return out


def valid_semantic(s):
    result=getattr(s,"semantic",None)
    return (result is not None and result.status=="OK" and result.frame_id==s.stats["frame_id"]
            and isinstance(result.labels,np.ndarray) and result.labels.dtype==np.uint8
            and result.labels.shape==s.packet.capture.color.shape[:2]
            and result.labels.max(initial=0)<=2)

def semantic_overlay(s,alpha):
    # [v3.5 추가] 실제 RGB와 동일 좌표/프레임인 모델 예측만 반투명으로 겹친다.
    rgb=s.packet.capture.color
    if not valid_semantic(s): return rgb.copy()
    return cv2.addWeighted(rgb,1-alpha,classification_color(s.semantic.labels),alpha,0)

def fit_text(image,line,xy,width,size=.48,color=(235,235,235)):
    line=str(line).replace("\n"," ")
    # [v3.5 지연 개선] 매번 한 글자씩 폭을 재는 대신 최대 접두어 길이를 이진 탐색.
    if len(line)>4 and cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,size,1)[0][0]>width:
        lo,hi=1,len(line)-4
        while lo<hi:
            mid=(lo+hi+1)//2
            if cv2.getTextSize(line[:mid]+"...",cv2.FONT_HERSHEY_SIMPLEX,size,1)[0][0]<=width:lo=mid
            else:hi=mid-1
        line=line[:lo]+"..."
    text(image,line,xy,size,color)

def panel_rects(w,h):
    # [v3.5] 상단 RGB/SCNN/Costmap, 하단 Low Angle/High Angle. 원본 영상은 각 칸에 비율 유지 맞춤.
    gap=8
    # [v3.5 화면 정리] 하단 설명 4줄을 제거한 공간을 두 영상 행에 배분.
    top_h=(h-110-8-gap)//2
    bottom_h=h-110-8-gap-top_h
    available=w-4*gap
    rgb_w=int(available*.31);mask_w=rgb_w;map_w=available-rgb_w-mask_w
    bottom_w=(w-3*gap)//2
    return [(gap,110,rgb_w,top_h),
            (2*gap+rgb_w,110,mask_w,top_h),
            (3*gap+rgb_w+mask_w,110,map_w,top_h),
            (gap,110+top_h+gap,bottom_w,bottom_h),
            (2*gap+bottom_w,110+top_h+gap,w-3*gap-bottom_w,bottom_h)]

def map_rect(cfg,w,h):
    # [v3.5] 지도 격자의 실제 가로/세로 미터 비율을 보존하고 범례 공간을 확보.
    g=cfg["grid"];xm=g["x_max_m"]-g["x_min_m"];ym=g["y_max_m"]-g["y_min_m"]
    # 범례의 글자/색상칸을 키워도 지도는 미터 종횡비를 유지한다.
    legend_width=max(190,min(300,int(w*.40)))
    scale=min(max(1,w-24-30-8-legend_width)/ym,max(1,h-30)/xm)
    return 24,8,max(1,int(ym*scale)),max(1,int(xm*scale))

def robot_field_boxes(cfg,w,h):
    # [추가] 제목/입력 테두리/마우스 영역에서 같은 배치를 공유한다.
    x0,_,mw,_=map_rect(cfg,w,h)
    left=x0+mw+30;gap=10;available=w-left-8
    width=(available-gap)//2
    return ((left,25,width,25),(left+width+gap,25,available-width-gap,25))


class Dashboard:
    def __init__(self,cfg,interactive_robot=False):
        self.cfg=cfg
        self.interactive_robot=interactive_robot
        self.robot_input=RobotWidthInput(cfg["robot"].get("width_m",2*cfg["robot"]["radius_m"]))
        self.height_input=RobotHeightInput(cfg["grid"]["obstacle_max_height_m"],
                                           cfg["grid"]["obstacle_min_height_m"])
        self.robot_field_rect=None;self.height_field_rect=None
        self.azimuth=cfg["view"]["azimuth_deg"]
        self.elevation=cfg["view"]["elevation_deg"]
        self.low_angle_zoom=1.0
        self.map_mode="top"  # [v3.5 수정] 우하단은 항상 위에서 보는 격자 지도.
        self.cloud_mode=cfg["view"]["cloud_mode"]
        self.low_angle_yaw=0.
        self.low_angle_pitch=cfg["low_angle"]["costmap_pitch_deg"]
        self.high_angle_yaw=0.
        self.high_angle_pitch=cfg["high_angle"]["costmap_pitch_deg"]
        self.high_angle_zoom=1.
        self.active_angle="low_angle"
    def begin_robot_edit(self,selected):
        # [추가] 한 번에 한 칸만 편집. 다른 칸의 Enter 미확정 입력은 취소한다.
        for field in (self.robot_input,self.height_input):
            if field is not selected and field.editing:field.cancel()
        selected.begin()

    def mouse(self,event,x,y,flags=None,param=None):
        if event==cv2.EVENT_LBUTTONDOWN and self.interactive_robot:
            for field,rect in ((self.robot_input,self.robot_field_rect),
                               (self.height_input,self.height_field_rect)):
                if rect:
                    x0,y0,x1,y1=rect
                    if x0<=x<=x1 and y0<=y<=y1:
                        self.begin_robot_edit(field);return
    def key(self,k):
        if self.interactive_robot:
            for field in (self.robot_input,self.height_input):
                if k in (ord(field.shortcut.lower()),ord(field.shortcut.upper())):
                    self.begin_robot_edit(field);return True
            for field in (self.robot_input,self.height_input):
                if field.editing and field.key(k):return True
        # [수정] A/D와 +/-는 선택된 패널과 관계없이 Low Angle/High Angle 시점을 함께 조작한다.
        # 실제 점군 좌표나 Costmap 계산은 바꾸지 않고 표시용 회전/배율만 변경한다.
        if k in (ord("a"),ord("A"),ord("d"),ord("D")):
            delta=5 if k in (ord("a"),ord("A")) else -5
            self.low_angle_yaw+=delta;self.high_angle_yaw+=delta
            if self.cloud_mode=="rgb":self.azimuth-=delta
            return
        if k in (ord("+"),ord("="),ord("-")):
            factor=1.1 if k in (ord("+"),ord("=")) else 1/1.1
            self.low_angle_zoom=float(np.clip(self.low_angle_zoom*factor,.5,2.5))
            self.high_angle_zoom=float(np.clip(self.high_angle_zoom*factor,.5,2.5))
            return
        # 높이/기울기는 두 시점의 개별 설정을 유지. TAB으로 W/X 적용 대상을 선택한다.
        if k==9:
            self.active_angle="high_angle" if self.active_angle=="low_angle" else "low_angle"
            return
        if self.active_angle=="high_angle":
            if k in (ord("w"),ord("x")):self.high_angle_pitch=float(np.clip(self.high_angle_pitch+(2 if k==ord("w") else -2),-80,80))
            return
        if k==ord("c"): self.cloud_mode="rgb" if self.cloud_mode=="obstacle_low_angle" else "obstacle_low_angle"
        if self.cloud_mode=="obstacle_low_angle":
            if k in (ord("w"),ord("x")):self.low_angle_pitch=float(np.clip(self.low_angle_pitch+(2 if k==ord("w") else -2),-80,80))
            return
        if k in (ord("w"),ord("x")): self.elevation=float(np.clip(self.elevation+(5 if k==ord("w") else -5),-10,85))

    def prepared_cloud(self,s):
        # 새 처리 프레임당 한 번만 두 장애물 화면의 입력을 준비한다.
        if getattr(self,"_cloud_snapshot",None) is not s:
            self._cloud_snapshot=s
            self._cloud_prepared=prepare_obstacle_cloud(s,self.cfg)
        return self._cloud_prepared

    def high_angle_cloud(self,s,w,h):
        return obstacle_view(s,self.cfg,w,h,self.high_angle_yaw,self.high_angle_pitch,self.high_angle_zoom,view_name="high_angle",prepared=self.prepared_cloud(s))

    def low_angle_cloud(self,s,w,h):
        if self.cloud_mode=="obstacle_low_angle":
            return obstacle_view(s,self.cfg,w,h,self.low_angle_yaw,self.low_angle_pitch,self.low_angle_zoom,prepared=self.prepared_cloud(s))
        image=np.full((h,w,3),(28,24,20),np.uint8)
        az,el=np.deg2rad([self.azimuth,self.elevation])
        target=np.array([2.3,0,.4])
        eye=target+np.array([-np.cos(el)*np.cos(az),-np.cos(el)*np.sin(az),np.sin(el)])*7/self.low_angle_zoom
        front=(target-eye); front/=np.linalg.norm(front)
        right=np.cross(front,[0,0,1]); right/=np.linalg.norm(right)
        up=np.cross(right,front)
        basis=np.stack((right,up,front),axis=1)
        def screen(p):
            q=(p-eye)@basis
            z=q[:,2]
            return np.column_stack((w/2+q[:,0]*w/np.maximum(z,.01),
                                    h/2-q[:,1]*w/np.maximum(z,.01))),z
        # [v3.5 추가] 1m 바닥 눈금과 고정 지도 축을 함께 보여 시점 변경과 좌표 변환을 구별한다.
        for x in range(7):
            p,_=screen(np.array([[x,-3,0],[x,3,0]]))
            cv2.line(image,tuple(p[0].astype(int)),tuple(p[1].astype(int)),(65,55,45),1)
        for y in range(-3,4):
            p,_=screen(np.array([[0,y,0],[6,y,0]]))
            cv2.line(image,tuple(p[0].astype(int)),tuple(p[1].astype(int)),(65,55,45),1)
        ids=np.arange(len(s.points)) if self.cfg["view"]["show_untextured_points"] else np.flatnonzero(s.color_valid)
        maximum=self.cfg["processing"]["display_max_points"]
        if len(ids)>maximum*2: ids=ids[np.linspace(0,len(ids)-1,maximum*2,dtype=int)]
        ids=ids[voxel_indices(s.points[ids],self.cfg["processing"]["display_voxel_m"])]
        if len(ids)>maximum: ids=ids[np.linspace(0,len(ids)-1,maximum,dtype=int)]
        p,z=screen(s.points[ids])
        ij=np.rint(p).astype(int)
        good=(z>.1)&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
        ij,z,rgb=ij[good],z[good],s.rgb[ids][good]
        order=np.argsort(z)[::-1]
        for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
            image[np.minimum(ij[order,1]+dy,h-1),np.minimum(ij[order,0]+dx,w-1)]=rgb[order,::-1]
        origin,_=screen(np.array([[0,0,0]])); origin=tuple(origin[0].astype(int))
        for end,name,col in [([1,0,0],"+X",(80,100,255)),([0,1,0],"+Y",(70,255,90)),([0,0,1],"+Z",(255,150,60))]:
            p,_=screen(np.array([end])); end2=tuple(p[0].astype(int))
            cv2.arrowedLine(image,origin,end2,col,2,tipLength=.15)
            text(image,name,end2,.5,col)
        return image

    def map(self,s,w,h,now):
        physical=s.physical.copy()
        # [v3.5 추가] 마지막 화면만 남아 있는 동안에도 오래된 상태를 자유가 아닌 미확인으로 표시.
        elapsed=max(0,now-s.packet.capture.received_monotonic) if not s.packet.replay else 0
        if elapsed>self.cfg["grid"]["free_stale_s"]: physical[physical==0]=-1
        if elapsed>self.cfg["grid"]["obstacle_stale_s"]: physical[physical==100]=-1
        cfg=getattr(s,"applied_config",None) or self.cfg
        risk=getattr(s,"risk_cost",None)
        if elapsed>cfg["grid"]["obstacle_stale_s"]:risk=np.zeros_like(physical,dtype=np.uint8)
        elif risk is None:
            risk=obstacle_risk(physical,cfg["grid"]["resolution_m"],cfg["robot"],
                (cfg["grid"]["x_min_m"],cfg["grid"]["y_min_m"]),s.transform[:2,3])
        cost=display_cost(physical,risk)
        unknown=physical==-1

        # [v3.5 수정] 지도는 확대하고, 진단/범례를 옆에 배치하여 색 영역을 가리지 않는다.
        in_view=getattr(s,"in_view",None)
        if in_view is None:in_view=np.ones_like(cost,bool)
        cost[~in_view]=255  # 검정 시야 밖은 화면 범례의 위험 셀 수에서 제외.
        out=self.top_map(cost,w,h,in_view,unknown)
        x0,y0,mw,mh=map_rect(self.cfg,w,h)
        lx=x0+mw+30
        legend=[("Obstacle",cost==254,(0,0,255)),("Forbidden <=0.25m",cost==253,(255,0,200)),
                ("Inflation",(cost>0)&(cost<253),(0,230,255)),
                ("Observed free",cost==0,(0,255,0)),("Unknown (no risk)",(cost==255)&in_view,(80,80,80)),
                ("Unknown (risk)",unknown&(risk>0)&in_view,(150,150,150)),
                ("Outside FOV",~in_view,(0,0,0))]
        r=cfg["robot"]
        legend[1]=(f"Forbidden <={r['radius_m']+r['safety_margin_m']:.2f}m",legend[1][1],legend[1][2])
        legend[2]=(f"Inflation <={r['inflation_radius_m']:.2f}m",legend[2][1],legend[2][2])
        legend_width=w-lx-8
        for label,(fx,fy,fw,fh) in zip((ROBOT_WIDTH_LABEL,ROBOT_HEIGHT_LABEL),
                                      robot_field_boxes(cfg,w,h)):
            # 제목은 줄임표 없이 각 칸 너비에 맞춰 줄인다.
            label_px=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.48,1)[0][0]
            text(out,label,(fx,18),.48*min(1.,max(1,fw-2)/max(1,label_px)))
        # [v3.5 화면 정리] 입력 안내와 진단 문구를 없애고 범례를 남는 세로 공간에 맞춘다.
        row_height=(h-78)/len(legend)
        swatch=max(16,min(24,int(row_height*.48)))
        label_size=min(.62,max(.45,row_height/75))
        label_width=legend_width-swatch-9
        # 긴 항목도 생략하지 않고 범례 열 안에 들어가는 공통 글자 크기 사용.
        widest=max(cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,label_size,1)[0][0]
                   for label,_,_ in legend)
        if widest>label_width:label_size*=label_width/widest
        count_size=max(.33,min(.43,label_size*.8))
        for i,(label,mask,col) in enumerate(legend):
            yy=int(72+i*row_height)
            cv2.rectangle(out,(lx,yy-12),(lx+swatch,yy-12+swatch),col,-1)
            if label=="Outside FOV":
                cv2.rectangle(out,(lx,yy-12),(lx+swatch,yy-12+swatch),(170,170,170),1)
            if label=="Unknown (risk)":
                for dx in range(-swatch,swatch,7):
                    ok,a,b=cv2.clipLine((lx,yy-12,swatch+1,swatch+1),
                        (lx+dx,yy-12+swatch),(lx+dx+swatch,yy-12))
                    if ok:cv2.line(out,a,b,(230,230,230),1)
            text(out,label,(lx+swatch+9,yy),label_size)
            text(out,f"{int(mask.sum())} cells",(lx+swatch+9,yy+15),count_size,(195,195,195))
        return out

    def top_map(self,cost,w,h,in_view=None,unknown=None):
        # [v3.5 수정] NEAREST 확대: 계산 격자/비용을 변경하지 않고 선명하게 표시한다.
        out=np.full((h,w,3),22,np.uint8)
        x0,y0,mw,mh=map_rect(self.cfg,w,h)
        shown=cv2.resize(cost[::-1,::-1],(mw,mh),interpolation=cv2.INTER_NEAREST)
        shown_view=np.ones_like(shown,bool) if in_view is None else cv2.resize(
            in_view[::-1,::-1].astype(np.uint8),(mw,mh),interpolation=cv2.INTER_NEAREST).astype(bool)
        shown_unknown=None if unknown is None else cv2.resize(unknown[::-1,::-1].astype(np.uint8),(mw,mh),interpolation=cv2.INTER_NEAREST).astype(bool)
        pixels=cost_color(shown,shown_view,shown_unknown)
        g=self.cfg["grid"]
        # [v3.5] 숫자/격자선은 좌표와 비율을 유지하고 색상 셀을 덮지 않는다.
        gridlines=np.zeros((mh,mw),np.uint8)
        for x in np.arange(np.ceil(g["x_min_m"]),g["x_max_m"]+1e-6,1):
            yy=min(mh-1,int((g["x_max_m"]-x)/(g["x_max_m"]-g["x_min_m"])*mh))
            cv2.line(gridlines,(0,yy),(mw-1,yy),1,1)
            text(out,f"{x:g}",(x0+mw+4,y0+yy+4),.34)
        for y in np.arange(np.ceil(g["y_min_m"]),g["y_max_m"]+1e-6,1):
            xx=min(mw-1,int((g["y_max_m"]-y)/(g["y_max_m"]-g["y_min_m"])*mw))
            cv2.line(gridlines,(xx,0),(xx,mh-1),1,1)
            text(out,f"{y:g}",(x0+xx-5,y0+mh+14),.34)
        pixels[(gridlines>0)&(shown==255)&shown_view]=(105,105,105)
        out[y0:y0+mh,x0:x0+mw]=pixels
        return out


    def render(self,s,rates=None,status="",live_capture=None,size=None):
        # [v3.5] 실제 창의 영상 영역 크기에 맞춰 재배치. 작은 창은 전체를 등비 축소한다.
        w,h=size or (self.cfg["view"]["width"],self.cfg["view"]["height"])
        w,h=max(1,int(w)),max(1,int(h))
        started=time.perf_counter()
        if w<1200 or h<800:
            image=fit_image(self._render(s,rates,status,live_capture,1200,800),w,h)
        else:image=self._render(s,rates,status,live_capture,w,h)
        for name in ("robot","height"):
            base=getattr(self,f"_{name}_field_base",None)
            rect=None
            if base:
                bx,by,bw,bh,basew,baseh=base
                scale=min(w/basew,h/baseh);nw,nh=int(basew*scale),int(baseh*scale)
                ox,oy=(w-nw)//2,(h-nh)//2
                rect=(ox+int(bx*scale),oy+int(by*scale),ox+int((bx+bw)*scale),oy+int((by+bh)*scale))
            setattr(self,f"{name}_field_rect",rect)
        self.last_render_ms=(time.perf_counter()-started)*1000
        return image

    def _render(self,s,rates,status,live_capture,w,h):
        cfg=(getattr(s,"applied_config",None) if s is not None else None) or self.cfg
        self._robot_field_base=None;self._height_field_base=None
        out=np.full((h,w,3),20,np.uint8)
        rects=panel_rects(w,h)
        now=time.monotonic(); rates=rates or {}
        names=["RGB","RGB + Fast SCNN mask","Local Costmap",
               ("Obstacle Point Cloud | Low Angle" if self.cloud_mode=="obstacle_low_angle" else "Colored Point Cloud | Low Angle"),
               "Obstacle Point Cloud | High Angle"]
        if s is None:
            text(out,"D435 v3.5 | RGB / SCNN / COSTMAP / Low Angle / High Angle",(12,24),.65)
            fit_text(out,status or "Waiting for processed RGB-D... Q/Esc: quit",(12,58),w-24,.48)
            for x,y,pw,ph in rects:
                cv2.rectangle(out,(x,y),(x+pw-1,y+ph-1),(65,65,65),1)
                text(out,"Waiting for frame",(x+20,y+70),.6)
            for name,(x,y,pw,ph) in zip(names,rects):
                fit_text(out,name,(x+10,y+21),pw-20,.5)
            if status:fit_text(out,status,(12,90),w-24,.55,(60,190,255))
            if live_capture is not None:
                x,y,pw,ph=rects[0]
                out[y+30:y+ph,x:x+pw]=fit_image(live_capture.color,pw,ph-30)
                fit_text(out,"Live RGB | waiting for first processed frame",(12,25),w-24,.5)
            return out
        c=s.packet.capture
        age=0 if s.packet.replay else max(0,now-c.received_monotonic)
        stale=age>cfg["processing"]["max_frame_age_s"]
        g,r=cfg["grid"],cfg["robot"]
        sem=getattr(s,"semantic",None)
        ok=valid_semantic(s)
        sem_status=sem.status if sem is not None else "NOT CONNECTED"
        if sem_status=="OK" and not ok: sem_status="FRAME/MASK MISMATCH"
        # [v3.5 추가] 렌즈 설치 높이와 로봇 통과 높이 설정을 구분해서 표시한다.
        line1=f"D435 v3.5 | USB {s.packet.calibration['device'].get('usb_type_descriptor','?')} | camera H = {cfg['camera']['height_m']:.2f}m | frame {str(s.stats['frame_id']).split(':',1)[0]}"
        line2=f"FPS acquire {rates.get('acquire',0):.1f} / SCNN {rates.get('scnn',0):.1f} / cloud {rates.get('cloud',0):.1f} / map {rates.get('map',0):.1f} | compute {s.stats['processing_ms']:.1f}ms | age {age*1000:.0f}ms"
        line3=f"Valid depth {100*s.stats['valid_depth_fraction']:.1f}%"
        line4=""  # [v3.5 화면 정리] 비교할 최신 RGB가 없는 저장 화면에서는 수신/지도 시각을 표시하지 않음.
        for i,line in enumerate([line1,line2,line3,line4]):
            fit_text(out,line,(12,20+i*23),w-24,.48,(60,170,255) if i==2 and (stale or not s.floor.ok) else (235,235,235))
        # [v3.5 추가] 동일 계산 결과를 매 UI 틱마다 재렌더하지 않는다. 시점 조작과
        # 유효기간 변화 시에는 갱신하며 최신 RGB는 SCNN 완료를 기다리지 않는다.
        cache_key=(id(s),self.cloud_mode,self.azimuth,self.elevation,self.low_angle_yaw,self.low_angle_pitch,self.low_angle_zoom,self.high_angle_yaw,self.high_angle_pitch,self.high_angle_zoom,
                   age>g["free_stale_s"],age>g["obstacle_stale_s"],w,h)
        if getattr(self,"_panels_key",None)!=cache_key:
            self._panels_key=cache_key
            self._panels_snapshot=s  # id 재사용으로 인한 잘못된 캐시 방지
            processed=(semantic_overlay(s,cfg["semantics"]["overlay_alpha"]),
                self.map(s,rects[2][2],rects[2][3]-30,now),
                self.low_angle_cloud(s,rects[3][2],rects[3][3]-30),
                self.high_angle_cloud(s,rects[4][2],rects[4][3]-30))
            # [v3.5] 크기 맞춤도 결과/창 크기가 바뀔 때만 수행한다.
            self._processed_panels=tuple(fit_image(panel,rect[2],rect[3]-30)
                for panel,rect in zip(processed,rects[1:]))
        first=rects[0]
        panels=[fit_image(c.color if live_capture is None else live_capture.color,
                          first[2],first[3]-30),*self._processed_panels]
        if live_capture is not None:
            # [v3.5 지연 진단] 최신 RGB를 일부러 늦추지 않고 실제 관측 시각 차이를 표시.
            gap_ms=max(0,live_capture.received_monotonic-c.received_monotonic)*1000
            line4=f"RGB-to-result gap {gap_ms:.0f}ms"
            cv2.rectangle(out,(0,77),(w,100),(20,20,20),-1)
            fit_text(out,line4,(12,89),w-24,.46)
        for k,(panel,name,(x,y,pw,ph)) in enumerate(zip(panels,names,rects)):
            out[y+30:y+ph,x:x+pw]=panel
            cv2.rectangle(out,(x,y),(x+pw-1,y+ph-1),(65,65,65),1)
            # [화면 정리] 거리 범례 제목은 양쪽 점군 제목과 같은 줄/글자 크기로 우측 정렬.
            title_size=.48;title_width=pw-20
            if k in (3,4) and cfg["pointcloud_color"]["mode"]=="rainbow" and (k==4 or self.cloud_mode=="obstacle_low_angle"):
                range_title="Sensor Range"
                range_width=cv2.getTextSize(range_title,cv2.FONT_HERSHEY_SIMPLEX,title_size,1)[0][0]
                range_x=x+pw-10-range_width
                text(out,range_title,(range_x,y+21),title_size)
                title_width=range_x-(x+10)-12
            fit_text(out,name,(x+10,y+21),title_width,title_size,(60,210,255) if k==1 and not ok else (235,235,235))
        # [추가] 입력은 편집 중 값, 저장 대시보드는 해당 처리 프레임에 적용된 값만 표시.
        self.robot_input.sync(r.get("width_m",2*r["radius_m"]))
        self.height_input.sync(g["obstacle_max_height_m"])
        mx,my,mpw,mph=rects[2]
        for name,field,box in zip(("robot","height"),(self.robot_input,self.height_input),
                                  robot_field_boxes(cfg,mpw,mph-30)):
            fx,fy,fw,fh=box;fx+=mx;fy+=my+30
            setattr(self,f"_{name}_field_base",(fx,fy,fw,fh,w,h))
            cv2.rectangle(out,(fx,fy),(fx+fw,fy+fh),(42,42,42),-1)
            # 기존 화면 정리를 유지: 잘못된 입력은 빨간 테두리로만 알린다.
            invalid=field.editing and field.message.startswith("Invalid:")
            field_color=(70,70,255) if invalid else (70,220,255) if field.editing else (140,140,140)
            cv2.rectangle(out,(fx,fy),(fx+fw,fy+fh),field_color,1)
            if field.editing and field.select_all:
                cv2.rectangle(out,(fx+3,fy+3),(fx+min(65,fw-3),fy+fh-3),(130,75,35),-1)
            fit_text(out,field.text+(" |" if field.editing else " m"),(fx+6,fy+18),fw-12,.45)
        if not ok:
            # [v3.5 추가] 모델 미제공/오류 시 RGB는 유지하고 현재 상태를 명확히 표시한다.
            x,y,pw,ph=rects[1]
            cv2.rectangle(out,(x+2,y+ph-58),(x+pw-2,y+ph-2),(22,22,22),-1)
            fit_text(out,"NO SCNN MASK - "+sem_status,(x+12,y+ph-36),pw-24,.48,(60,210,255))
            message=sem.message if sem is not None else "Configure the trained model"
            fit_text(out,message,(x+12,y+ph-14),pw-24,.42)
        return out
