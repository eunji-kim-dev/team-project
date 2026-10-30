"""물리 점유와 중심 위치 비용을 분리. 미관측=255이며 시간 만료는 자유가 아닌 미확인."""
import numpy as np
import cv2
from functools import lru_cache
from .geometry import project

UNKNOWN=-1
FREE=0
OBSTACLE=100

def height_samples(g):
    # [v3.5 수정] 상한이 바뀌어도 공통 높이의 검사 위치는 유지한다.
    # 바닥 기준 고정 격자 + 양 끝점. 0.80/1.50m는 같은 0.10m 격자에 속한다.
    lo,hi,step=(g[k] for k in ("obstacle_min_height_m","obstacle_max_height_m","free_vertical_step_m"))
    middle=np.arange(np.floor(lo/step)+1,np.ceil(hi/step))*step
    middle=middle[(middle>lo+1e-9)&(middle<hi-1e-9)]
    return np.unique(np.round(np.r_[lo,middle,hi],9))


def visibility_depth(depth_m, policy="valid_center", minimum=5):
    # [v3.5 수정] 기존 erode는 주변 9픽셀 중 깊이 0 하나만 있어도 유효 중심값까지 0으로 만들었다.
    # 새 방식: 중심에 실제 측정값 + 주변 최소 5/9 측정값이 있어야 검사한다.
    # 유효 이웃 중 가장 가까운 깊이를 택하므로 관측된 얇은 장애물을 평균으로 지우지 않는다.
    # 중심 결측은 그대로 0이며 보간/홀 채우기를 하지 않는다. 실제 누락/가림은 UNKNOWN이다.
    measured=np.isfinite(depth_m)&(depth_m>0)
    clean=np.where(measured,depth_m,0).astype(np.float32)
    kernel=np.ones((3,3),np.uint8)
    if policy=="strict":
        return cv2.erode(clean,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    nearest=cv2.erode(np.where(measured,clean,np.inf),kernel,
                      borderType=cv2.BORDER_CONSTANT,borderValue=float("inf"))
    count=cv2.filter2D(measured.astype(np.float32),-1,kernel,borderType=cv2.BORDER_CONSTANT)
    ok=measured&(count>=minimum)
    ok[[0,-1],:]=False;ok[:,[0,-1]]=False
    return np.where(ok,nearest,0).astype(np.float32)

class LocalGrid:
    def __init__(self, config):
        self.cfg=config
        self.g=config["grid"]
        g=self.g; r=g["resolution_m"]
        self.shape=(round((g["x_max_m"]-g["x_min_m"])/r),
                    round((g["y_max_m"]-g["y_min_m"])/r))
        i,j=np.indices(self.shape)
        self.xy=np.stack((g["x_min_m"]+(i+.5)*r,g["y_min_m"]+(j+.5)*r),-1)
        self.reset()
    def reset(self):
        self.physical=np.full(self.shape,UNKNOWN,np.int8)
        self.last_obstacle=np.full(self.shape,-np.inf)
        self.last_free=np.full(self.shape,-np.inf)
        self.clear_streak=np.zeros(self.shape,np.uint16)
        self.updated_at=None
        self.free_diagnostics={}
        self.visibility_masks={}
    def begin_current_frame(self):
        # [v3.5 추가] 과거 유지/해제 카운터를 재할당하지 않고 현재 점유 배열만 초기화한다.
        self.physical.fill(UNKNOWN)
        self.updated_at=None
        self.free_diagnostics={}
    def update_current(self,occupied,free,t):
        # [v3.5 추가] 이번 프레임의 증거만 사용. 과거 장애물 유지/제거 시간은 적용하지 않는다.
        self.physical[free&~occupied]=FREE
        self.physical[occupied]=OBSTACLE
        self.updated_at=t
    def indices(self, points):
        g=self.g; r=g["resolution_m"]
        ij=np.floor((points[:,:2]-[g["x_min_m"],g["y_min_m"]])/r).astype(np.int32)
        inside=(ij[:,0]>=0)&(ij[:,0]<self.shape[0])&(ij[:,1]>=0)&(ij[:,1]<self.shape[1])
        return ij,inside
    def evidence_cells(self, points, minimum=1):
        ij,inside=self.indices(points)
        # [v3.5 지연 개선] 같은 셀에 대한 반복 np.add.at 대신 정수 빈도 집계.
        # minimum=1은 존재 여부만 필요하므로 중복 인덱스에 True 대입하면 동일하다.
        flat=ij[inside,0]*self.shape[1]+ij[inside,1]
        return self.evidence_from_flat(flat,minimum)
    def evidence_from_flat(self,flat,minimum=1):
        # [v3.5] 같은 점의 격자 인덱스를 한 번만 계산해 여러 분류 마스크에 재사용.
        if minimum==1:
            out=np.zeros(self.shape,bool);out.ravel()[flat]=True
            return out
        count=np.bincount(flat,minlength=self.shape[0]*self.shape[1]).reshape(self.shape)
        return count>=minimum
    def visible_free(self, floor_points, depth_m, intr, T, plane, obstacle_depth=None, supported=None):
        # [v3.5 추가] 실제 바닥 관측이 있는 셀에서만 검사한다. 빈 점군을 자유로 바꾸지 않는다.
        if supported is None:supported=self.evidence_cells(floor_points)
        self.visibility_masks={key:np.zeros(self.shape,bool) for key in ("supported","outside","missing","occluded","obstacle_occluded","volume_clear")}
        self.visibility_masks["supported"]=supported.copy()
        cells=np.argwhere(supported)
        free=np.zeros(self.shape,bool)
        self.free_diagnostics={"floor_support_cells":len(cells),"no_floor_support_cells":int(free.size-len(cells)),
            "out_of_view_cells":0,"missing_depth_cells":0,"occluded_or_margin_cells":0,"volume_clear_cells":0,
            "reason_priority":"outside_view > missing_depth > occluded_or_margin; one reason per rejected supported cell"}
        if not len(cells): return free
        xy=self.xy[supported]
        g=self.g; n=np.asarray(plane.normal)
        # [v3.5 추가] 셀 중앙/네 모서리의 전체 높이 구간이 관측 Depth 앞쪽일 때만 자유.
        # [80cm 로봇] 상한은 config의 obstacle_max_height_m=0.80. 렌즈 높이 0.74와 별개이다.
        # 유한한 샘플 해상도의 보수적 검사이며 연속 공간의 무장애 보증은 아니다.
        heights=height_samples(g)
        self.free_diagnostics["sample_heights_m"]=heights.tolist()
        half=g["resolution_m"]*.49
        offsets=np.array([[0,0],[-half,-half],[-half,half],[half,-half],[half,half]])
        min_depth=visibility_depth(depth_m,g.get("visibility_policy","strict"),g.get("visibility_min_neighbors",5))
        self.free_diagnostics["visibility_policy"]=g.get("visibility_policy","strict")
        self.free_diagnostics["visibility_min_neighbors"]=g.get("visibility_min_neighbors",5)
        self.free_diagnostics["usable_depth_fraction"]=float(np.mean(min_depth>0))
        # [v3.5 수정] 바닥 표면의 가까운 이웃을 장애물 가림으로 간주하지 않는다.
        # 전체 높이 확인은 raw 깊이, 관측 바닥 정책의 가림은 지면 위 실제 측정으로 분리.
        blockers=None
        if obstacle_depth is not None:
            blockers=cv2.erode(np.where(obstacle_depth>0,obstacle_depth,np.inf).astype(np.float32),
                              np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=float("inf"))
        obstacle_blocked=np.zeros(len(cells),bool)
        valid=np.ones(len(cells),bool)
        outside=np.zeros(len(cells),bool);missing=outside.copy();blocked=outside.copy()
        # 배열 크기와 지연을 줄여 분할 검사하고, 한 높이라도 가려짐/결측이면 미확인으로 둔다.
        for offset in offsets:
            qxy=xy+offset
            floor_z=-(qxy@n[:2]+plane.offset_m)/n[2]
            p=np.empty((len(cells),len(heights),3),np.float32)
            p[:,:,:2]=qxy[:,None,:]
            p[:,:,2]=floor_z[:,None]+heights[None,:]
            optical=(p-T[:3,3])@T[:3,:3]
            uv=project(optical,intr)
            u,v=np.rint(uv).astype(np.int32).transpose(2,0,1)
            ok=(optical[:,:,2]>0)&(u>=1)&(u<depth_m.shape[1]-1)&(v>=1)&(v<depth_m.shape[0]-1)
            measured=min_depth[np.clip(v,0,depth_m.shape[0]-1),np.clip(u,0,depth_m.shape[1]-1)]
            if blockers is not None:
                blocker=blockers[np.clip(v,0,depth_m.shape[0]-1),np.clip(u,0,depth_m.shape[1]-1)]
                obstacle_blocked |= np.any(ok&np.isfinite(blocker)&(optical[:,:,2]+g["free_depth_margin_m"]>=blocker),axis=1)
            # [v3.5 추가] 바닥 인식 성공과 통과 공간 검사 실패를 구분하는 진단 수치.
            outside|=np.any(~ok,axis=1)
            missing|=np.any(ok&(measured<=0),axis=1)
            blocked|=np.any(ok&(measured>0)&(optical[:,:,2]+g["free_depth_margin_m"]>=measured),axis=1)
            ok&=(measured>0)&(optical[:,:,2]+g["free_depth_margin_m"]<measured)
            valid&=np.all(ok,axis=1)
        if blockers is None:obstacle_blocked=blocked.copy()
        self.free_diagnostics.update(obstacle_occluded_cells=int(obstacle_blocked.sum()),
            out_of_view_cells=int(outside.sum()),
            missing_depth_cells=int((~outside&missing).sum()),
            occluded_or_margin_cells=int((~outside&~missing&blocked).sum()),
            volume_clear_cells=int(valid.sum()))
        for name,mask in (("outside",outside),("missing",missing),("occluded",blocked),("obstacle_occluded",obstacle_blocked),("volume_clear",valid)):
            self.visibility_masks[name][cells[mask,0],cells[mask,1]]=True
        free[cells[valid,0],cells[valid,1]]=True
        return free
    def age(self, t):
        # [v3.5 추가] 관측이 끊기면 자유로 간주하지 않고 오래된 상태를 UNKNOWN으로 되돌린다.
        old_obstacle=(self.physical==OBSTACLE)&(t-self.last_obstacle>self.g["obstacle_stale_s"])
        old_free=(self.physical==FREE)&(t-self.last_free>self.g["free_stale_s"])
        self.physical[old_obstacle|old_free]=UNKNOWN
    def update(self, occupied, free, t):
        self.age(t)
        free=free&~occupied
        eligible=free&(t-self.last_obstacle>=self.g["obstacle_hold_s"])
        self.clear_streak[~eligible]=0
        self.clear_streak[eligible]=np.minimum(self.clear_streak[eligible]+1,65534)
        clear=eligible&((~np.isfinite(self.last_obstacle))|(self.clear_streak>=self.g["clear_confirmations"]))
        # 처음 관측한 자유 셀은 즉시 허용. 이전 장애물의 해제에는 유지 시간과 연속 확인을 요구한다.
        self.physical[clear]=FREE
        self.last_free[clear]=t
        self.physical[occupied]=OBSTACLE
        self.last_obstacle[occupied]=t
        self.clear_streak[occupied]=0
        self.updated_at=t
    def cost(self):
        return inflate(self.physical,self.g["resolution_m"],self.cfg["robot"],(self.g["x_min_m"],self.g["y_min_m"]))
    def metadata(self):
        return {"array_order":"[i_forward_X, j_left_Y]",
                "accumulation_mode":self.cfg["mapping"]["mode"],
                "reference":"current frame camera floor origin; no history" if self.cfg["mapping"]["mode"]=="camera_relative" else "fixed camera floor frame; R after movement",
                "origin_cell_corner_m":[self.g["x_min_m"],self.g["y_min_m"],0],
                "origin_definition":"floor point below original depth optical center",
                "resolution_m":self.g["resolution_m"],"shape":list(self.shape),
                "cell_center":"X=x_min+(i+0.5)*res; Y=y_min+(j+0.5)*res",
                "physical_values":{"unknown":-1,"free":0,"obstacle":100},
                "cost_values":{"free":0,"inflation":[1,252],"forbidden":253,"physical_obstacle":254,"unknown":255},
                "robot":dict(self.cfg["robot"]),
                "cost_direction":"camera-to-cell ray; first occupied grid cell blocks rear costs; measured obstacles preserved",
                "distance_definition":"Euclidean distance between grid-cell centers; no second robot-radius dilation",
                "free_evidence":self.g.get("free_space_policy","volume_confirmed"),
                "free_policy_definition":"observed_floor: measured floor and no observed occlusion/conflict; missing upper depth allowed. volume_confirmed: all sampled heights clear",
                "fov_definition":"calibrated native Depth horizontal field across grid column heights; outside is black using separate mask; unknown cost stays 255",
                "last_successful_map_update_monotonic":self.updated_at}

def _camera_occluded_dda(obstacles,resolution,origin_xy=(0.,0.),camera_xy=(0.,0.),targets=None):
    """카메라→셀 중심 선분의 2D 격자 교차 검사. 가린 셀 뒤를 Unknown으로 보존."""
    # [v3.5] X축 반원이 아닌 실제 카메라 위치 기준 DDA ray traversal.
    # 정확히 모서리를 통과하면 양쪽 인접 셀도 검사하여 대각선 틈으로 비용이 새지 않게 한다.
    shape=np.array(obstacles.shape)
    targets=(~obstacles) if targets is None else (targets&~obstacles)
    ids=np.argwhere(targets)
    result=np.zeros(obstacles.shape,bool)
    if not len(ids) or not obstacles.any():return result
    start=(np.asarray(camera_xy,dtype=float)-origin_xy)/resolution
    end=ids+.5;direction=end-start
    # 외부 카메라도 지도 진입점부터 검사. 지도 경계 위에서 시작하는 경우 포함.
    with np.errstate(divide="ignore",invalid="ignore"):
        near=np.where(direction!=0,np.minimum(-start/direction,(shape-start)/direction),-np.inf)
        far=np.where(direction!=0,np.maximum(-start/direction,(shape-start)/direction),np.inf)
    entry=np.maximum(0,np.max(near,axis=1))
    leave=np.minimum(1,np.min(far,axis=1))
    valid=entry<=leave
    ids=ids[valid];direction=direction[valid];entry=entry[valid]
    if not len(ids):return result
    cur=np.floor(start+(entry[:,None]+1e-9)*direction).astype(np.int32)
    step=np.sign(direction).astype(np.int32)
    with np.errstate(divide="ignore",invalid="ignore"):
        delta=np.where(direction!=0,1/np.abs(direction),np.inf)
        edge=cur+(step>0)
        tmax=np.where(direction!=0,(edge-start)/direction,np.inf)
    active=np.arange(len(ids))
    def hits(c):
        inside=np.all((c>=0)&(c<shape),axis=1)
        out=np.zeros(len(c),bool)
        out[inside]=obstacles[c[inside,0],c[inside,1]]
        return out
    for _ in range(int(shape.sum())+3):
        if not len(active):break
        blocked=hits(cur[active])
        hit_ids=ids[active[blocked]]
        result[hit_ids[:,0],hit_ids[:,1]]=True
        done=np.all(cur[active]==ids[active],axis=1)
        active=active[~blocked&~done]
        if not len(active):break
        equal=np.isclose(tmax[active,0],tmax[active,1],rtol=0,atol=1e-12)
        both=active[equal]
        if len(both):
            cx=cur[both].copy();cx[:,0]+=step[both,0]
            cy=cur[both].copy();cy[:,1]+=step[both,1]
            corner_hit=hits(cx)|hits(cy)
            q=ids[both[corner_hit]];result[q[:,0],q[:,1]]=True
            keep=np.ones(len(active),bool);keep[np.flatnonzero(equal)[corner_hit]]=False
            active=active[keep]
        move_x=tmax[active,0]<=tmax[active,1]+1e-12
        move_y=tmax[active,1]<=tmax[active,0]+1e-12
        ax=active[move_x];ay=active[move_y]
        cur[ax,0]+=step[ax,0];cur[ay,1]+=step[ay,1]
        tmax[ax,0]+=delta[ax,0];tmax[ay,1]+=delta[ay,1]
    return result


@lru_cache(maxsize=1)
def _camera_paths(shape,resolution,origin_xy,camera_xy):
    # [v3.5 지연 개선] 격자와 카메라 원점이 같으면 광선 경로는 불변.
    # 첫 프레임에만 경로를 만들고 이후에는 현재 장애물 배열만 조회한다.
    # 점유/SCNN/과거 장애물은 캐시하지 않으므로 회전 시 과거 지도가 남지 않는다.
    count=int(np.prod(shape));slots=2*sum(shape)+5
    if count*slots>8_000_000:return None  # 큰 사용자 격자는 제한 메모리 DDA로 처리.
    shape_a=np.asarray(shape);ids=np.indices(shape).reshape(2,-1).T
    start=(np.asarray(camera_xy)-origin_xy)/resolution
    direction=ids+.5-start
    with np.errstate(divide="ignore",invalid="ignore"):
        near=np.where(direction!=0,np.minimum(-start/direction,(shape_a-start)/direction),-np.inf)
        far=np.where(direction!=0,np.maximum(-start/direction,(shape_a-start)/direction),np.inf)
    entry=np.maximum(0,near.max(axis=1));leave=np.minimum(1,far.min(axis=1))
    cur=np.floor(start+(entry[:,None]+1e-9)*direction).astype(np.int32)
    step=np.sign(direction).astype(np.int32)
    with np.errstate(divide="ignore",invalid="ignore"):
        delta=np.where(direction!=0,1/np.abs(direction),np.inf)
        tmax=np.where(direction!=0,(cur+(step>0)-start)/direction,np.inf)
    matrix=np.full((count,slots),-1,np.int32);counts=np.zeros(count,np.int32)
    active=np.flatnonzero(entry<=leave)
    def record(rays,cells):
        inside=(cells[:,0]>=0)&(cells[:,0]<shape[0])&(cells[:,1]>=0)&(cells[:,1]<shape[1])
        rows=rays[inside];q=cells[inside]
        matrix[rows,counts[rows]]=q[:,0]*shape[1]+q[:,1];counts[rows]+=1
    for _ in range(sum(shape)+3):
        if not len(active):break
        record(active,cur[active])
        active=active[~np.all(cur[active]==ids[active],axis=1)]
        if not len(active):break
        both=active[np.abs(tmax[active,0]-tmax[active,1])<=1e-12]
        if len(both):
            cx=cur[both].copy();cx[:,0]+=step[both,0];record(both,cx)
            cy=cur[both].copy();cy[:,1]+=step[both,1];record(both,cy)
        move_x=tmax[active,0]<=tmax[active,1]+1e-12
        move_y=tmax[active,1]<=tmax[active,0]+1e-12
        ax=active[move_x];ay=active[move_y]
        cur[ax,0]+=step[ax,0];cur[ay,1]+=step[ay,1]
        tmax[ax,0]+=delta[ax,0];tmax[ay,1]+=delta[ay,1]
    assert np.all(counts>0),"Every target centre must have a grid path"
    indices=matrix[matrix>=0]
    starts=np.r_[0,np.cumsum(counts[:-1],dtype=np.int64)]
    return indices,starts

def camera_occluded(obstacles,resolution,origin_xy=(0.,0.),camera_xy=(0.,0.),targets=None):
    if not obstacles.any():return np.zeros(obstacles.shape,bool)
    paths=_camera_paths(obstacles.shape,float(resolution),tuple(origin_xy),tuple(camera_xy))
    if paths is None:return _camera_occluded_dda(obstacles,resolution,origin_xy,camera_xy,targets)
    indices,starts=paths
    blocked=np.logical_or.reduceat(obstacles.ravel()[indices],starts).reshape(obstacles.shape)
    blocked &= ~obstacles
    if targets is not None:blocked &= targets
    return blocked


def risk_layers(physical,resolution,robot,origin_xy=(0.,0.),camera_xy=(0.,0.)):
    obstacles=physical==OBSTACLE
    cost=np.zeros(physical.shape,np.uint8);rear=np.zeros(physical.shape,bool)
    if not obstacles.any():return cost,rear
    distance=cv2.distanceTransform((~obstacles).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)*resolution
    forbidden=robot["radius_m"]+robot["safety_margin_m"]
    outer=robot["inflation_radius_m"]
    if robot.get("cost_direction","all")=="camera_visible":
        # 비용 후보 또는 자유 셀만 검사하면 나머지는 이미 Unknown이므로 결과가 같다.
        rear=camera_occluded(obstacles,resolution,origin_xy,camera_xy,
                             (distance<=outer+1e-7)|(physical==FREE))
    shell=(distance>forbidden+1e-7)&(distance<=outer+1e-7)&~rear
    cost[shell]=np.clip(np.rint(252*np.exp(-robot["cost_scaling_per_m"]*(distance[shell]-forbidden))),1,252).astype(np.uint8)
    cost[(distance<=forbidden+1e-7)&~rear]=253
    # 실제 관측한 다른 높이의 장애물은 2D 가림 계산으로 삭제하지 않는다.
    cost[obstacles]=254
    return cost,rear

def obstacle_risk(physical,resolution,robot,origin_xy=(0.,0.),camera_xy=(0.,0.)):
    return risk_layers(physical,resolution,robot,origin_xy,camera_xy)[0]

def inflate(physical,resolution,robot,origin_xy=(0.,0.),camera_xy=(0.,0.)):
    cost,rear=risk_layers(physical,resolution,robot,origin_xy,camera_xy)
    cost[(physical==UNKNOWN)|rear]=255
    return cost


def display_cost(physical,risk):
    # [v3.5] 시각화에는 위험을 우선 표시하고 unknown 마스크를 사선으로 별도 합성.
    out=risk.copy()
    out[(physical==UNKNOWN)&(risk==0)]=255
    return out
