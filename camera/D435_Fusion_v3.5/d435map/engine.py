"""카메라/GUI와 분리한 계산 모듈. 한 입력 프레임에 대응하는 스냅샷을 반환한다."""
from dataclasses import dataclass
import time
import json
import copy
import numpy as np
from .geometry import camera_transform, rays, filtered_depth, colorize
from .floor import FloorResult
from .costmap import LocalGrid,risk_layers
from .semantics import FastSCNN
from .fusion import sample_semantics,fit_fused_floor,classify_points
from .field_of_view import horizontal_view
from .obstacle_support import supported_obstacles
from .candidate_review import CandidateReviewer
from .robot import validate_robot_height
from .semantic_free import projected_floor_free

@dataclass
class Packet:
    capture: object
    calibration: dict
    source_id: str = ""
    replay: bool = False

@dataclass
class Snapshot:
    packet: Packet
    points: np.ndarray
    rgb: np.ndarray
    color_valid: np.ndarray
    point_classes: np.ndarray
    source_indices: np.ndarray
    filtered_depth: np.ndarray
    measured_valid: np.ndarray
    color_classes: np.ndarray
    floor: object
    physical: np.ndarray
    cost: np.ndarray
    transform: np.ndarray
    grid_metadata: dict
    stats: dict
    semantic: object = None
    in_view: object = None
    volume_clear: object = None
    free_evidence: object = None
    semantic_native: object = None
    semantic_confidence_native: object = None
    rejected_obstacles_native: object = None
    semantic_conflict_native: object = None
    risk_cost: object = None
    local_review_native: object = None
    temporal_review_native: object = None
    rear_unknown: object = None
    applied_config: object = None
    semantic_free: object = None
    semantic_free_continuity: object = None

class Processor:
    def __init__(self, config):
        self.cfg=copy.deepcopy(config)
        self.grid=LocalGrid(self.cfg)
        self.T=camera_transform(config["camera"])
        self.processed=0
        self.started=time.monotonic()
        self.last_signature=None
        self.semantic=FastSCNN(config["semantics"])
        self.reviewer=CandidateReviewer(self.cfg)

    def set_robot(self,robot):
        # [v3.5] 작업 스레드의 프레임 시작 경계에서만 호출. 취득/모델/후보 기록은 재시작하지 않음.
        updated=copy.deepcopy(self.cfg);updated["robot"]=copy.deepcopy(robot)
        self.cfg=updated;self.grid.cfg=updated;self.grid.g=updated["grid"];self.reviewer.cfg=updated

    def set_robot_height(self,height):
        # [추가] 다음 프레임 전에 높이를 변경하고 이전 높이로 확인한 자유/후보 증거는 섞지 않는다.
        # 카메라와 SCNN은 그대로 유지하고, 높이에 의존하는 지도/후보 기록만 비운다.
        height=validate_robot_height(height,self.cfg["grid"]["obstacle_min_height_m"])
        if height==self.cfg["grid"]["obstacle_max_height_m"]:return
        updated=copy.deepcopy(self.cfg)
        updated["grid"]["obstacle_max_height_m"]=height
        updated["grid"]["robot_height_is_provisional"]=False
        self.cfg=updated;self.grid.cfg=updated;self.grid.g=updated["grid"];self.reviewer.cfg=updated
        self.grid.reset();self.reviewer.reset()

    def process(self, packet):
        start=time.monotonic()
        # [v3.5 추가] 이동/좌우 회전 때 과거 방향의 장애물을 남기지 않는 프레임별 로컬 지도.
        # D435 IMU/세계 좌표 자세를 추정하지 않는다. 현재 카메라 기준의 입력만 사용한다.
        if self.cfg["mapping"]["mode"]=="camera_relative":
            self.grid.begin_current_frame()
        c=packet.capture; cal=packet.calibration
        if c.original_depth.dtype!=np.uint16 or c.aligned_depth.dtype!=np.uint16:
            raise ValueError("Raw depth must be Z16 uint16")
        if not np.isfinite(cal["depth_scale_m"]) or cal["depth_scale_m"]<=0:
            raise ValueError("Invalid depth scale")
        if c.color.dtype!=np.uint8 or c.color.shape!=(cal["color_intrinsics"]["height"],cal["color_intrinsics"]["width"],3):
            raise ValueError("Color dimensions/dtype do not match calibration")
        if c.aligned_depth.shape!=c.color.shape[:2]: raise ValueError("Aligned depth dimensions mismatch")
        signature=(c.depth_frame["frame_number"],c.color_frame["frame_number"],c.depth_frame["timestamp_ms"])
        if signature==self.last_signature:
            raise ValueError("duplicate RGB-D frame; not counted as new map evidence")
        self.last_signature=signature
        frame_id=f"{c.depth_frame['frame_number']}:{c.color_frame['frame_number']}:{c.depth_frame['timestamp_ms']}"
        # [v3.5 추가] 원본 RGB와 같은 프레임에서 추론한다. 기하 분류/정답지를 SCNN처럼 표시하지 않는다.
        semantic=self.semantic.predict(c.color,frame_id)
        semantics_done=time.monotonic()
        if c.original_depth.shape!=(cal["depth_intrinsics"]["height"],cal["depth_intrinsics"]["width"]):
            raise ValueError("native-depth dimensions do not match its calibration")
        raw,filtered,valid=filtered_depth(c.original_depth,cal["depth_scale_m"],self.cfg["depth"])
        keep=filtered>0
        indices=np.flatnonzero(keep)
        optical=(rays(cal["depth_intrinsics"])*filtered[:,:,None]).reshape(-1,3)[indices]
        points=optical@self.T[:3,:3].T+self.T[:3,3]
        rgb,uv,color_ok,color_z=colorize(optical,c.color,cal)
        # [v3.5 추가] RGB 예측을 native 점의 Color 투영 좌표로 가져온다. 동일 frame ID를 필수 확인.
        point_labels,point_confidence,fusion_info=sample_semantics(
            semantic,frame_id,c.color.shape[:2],uv,color_ok,self.cfg["fusion"])
        floor,floor_source=fit_fused_floor(points,point_labels,point_confidence,
                                         self.cfg["floor"],self.cfg["fusion"],fusion_info["active"])
        fusion_info["floor_source"]=floor_source
        age=time.monotonic()-c.received_monotonic
        if not c.pairing.get("within_threshold",False):
            floor=FloorResult(False,"RGB/Depth temporal pairing rejected",[0.,0.,1.],0.)
        elif float(valid.mean())<self.cfg["depth"]["min_valid_fraction"]:
            floor=FloorResult(False,"insufficient valid native depth",[0.,0.,1.],0.)
        elif not packet.replay and age>self.cfg["processing"]["max_frame_age_s"]:
            floor=FloorResult(False,"input too old; map update blocked",[0.,0.,1.],0.)
        classes=np.full(len(points),2,np.uint8)
        grid_start=time.monotonic()
        g=self.cfg["grid"]
        semantic_added=np.zeros(self.grid.shape,bool)
        continuity_added=np.zeros(self.grid.shape,bool)
        semantic_free_info={"reason":"floor unavailable"}
        self.grid.free_diagnostics={"status":"floor_failed","floor_support_cells":0,"volume_clear_cells":0}
        in_view,fov_info=horizontal_view(self.grid,cal["depth_intrinsics"],self.T,
                                         self.cfg["grid"]["obstacle_max_height_m"],floor)
        volume_clear=np.zeros(self.grid.shape,bool)
        evidence=np.zeros(self.grid.shape,np.uint8)
        rejected_native=np.zeros(raw.shape,np.uint8)
        semantic_conflict_native=np.zeros(raw.shape,np.uint8)
        local_review_native=np.zeros(raw.shape,np.uint8)
        temporal_review_native=np.zeros(raw.shape,np.uint8)
        review_info={"status":"floor_failed"}
        support_info={"enabled":self.cfg["obstacle_support"]["enabled"],"status":"floor_failed","rejected_points":0}
        if floor.ok:
            masks=classify_points(points,floor,point_labels,point_confidence,self.cfg,fusion_info["active"])
            g=self.cfg["grid"]
            hard=masks["geometry_obstacle"]
            low=masks["semantic_obstacle"]
            ij,inside=self.grid.indices(points)
            point_flat=ij[:,0]*self.grid.shape[1]+ij[:,1]
            def point_evidence(mask,minimum=1):
                return self.grid.evidence_from_flat(point_flat[inside&mask],minimum)
            low_cells=point_evidence(low,self.cfg["fusion"]["semantic_obstacle_min_points"])
            supported_low=np.zeros(len(points),bool)
            supported_low[inside]=low_cells[ij[inside,0],ij[inside,1]]
            candidates=hard|(low&supported_low)
            obstacle,rejected,support_info=supported_obstacles(raw,indices,candidates,self.cfg["obstacle_support"])
            # [v3.5] 두 센서의 상충 증거는 자유/장애물 확정과 구분한다.
            # 동일 obstacle 마스크를 점군 표시와 Costmap 점유에 함께 사용한다.
            semantic_held=obstacle&masks["depth_semantic_conflict"]
            semantic_conflict_native.ravel()[indices[semantic_held]]=255
            obstacle &= ~semantic_held
            rejected |= semantic_held
            held_cells=point_evidence(semantic_held)
            fusion_info.update(depth_floor_conflict_points=int(semantic_held.sum()),
                depth_floor_conflict_cells=int(held_cells.sum()),
                obstacle_min_height_m=g["obstacle_min_height_m"],
                conflict_policy="measured height >= minimum but high-confidence SCNN floor: UNKNOWN, never free")
            # [v3.5] 전역 5cm 기준/기존 SCNN 충돌 판정 이후 약한 작은 후보만 재검사.
            signature=json.dumps({k:cal[k] for k in ("depth_intrinsics","color_intrinsics","depth_to_color","depth_scale_m")},sort_keys=True)
            obstacle,local_held,temporal_held,review_info=self.reviewer.process(
                points,indices,raw,obstacle,masks["floor"],point_labels,point_confidence,floor,
                c.color,c.received_unix_ns/1e9,signature)
            rejected |= local_held|temporal_held
            local_review_native.ravel()[indices[local_held]]=255
            temporal_review_native.ravel()[indices[temporal_held]]=255
            rejected_native.ravel()[indices[rejected]]=255
            uncertain=point_evidence(rejected)
            support_info["uncertain_cells"]=int(uncertain.sum())
            # [v3.5 추가] 보류한 점은 바닥으로 바꾸지 않는다. 가림 검사는 원 관측을
            # 그대로 보수적으로 사용하며, 해당 격자는 독립적인 장애물이 없으면 미확인.
            on_floor=masks["floor"]&~obstacle
            classes[on_floor]=0
            classes[obstacle]=1
            occupied=point_evidence(obstacle,g["min_obstacle_points"])
            # [v3.5 추가] 지면 위에 실제 측정된 표면만 가림 검사에 사용. 모호한 낮은 표면도 보수적으로 포함.
            barrier=(masks["height"]>self.cfg["floor"]["inlier_distance_m"])&(
                masks["height"]<=g["obstacle_max_height_m"])&~on_floor
            blocker_depth=np.zeros_like(raw)
            blocker_depth.ravel()[indices[barrier]]=raw.ravel()[indices[barrier]]
            volume_clear=self.grid.visible_free(None,raw,cal["depth_intrinsics"],self.T,floor,blocker_depth,supported=point_evidence(on_floor))
            observed=self.grid.visibility_masks["supported"]
            occluded=self.grid.visibility_masks["obstacle_occluded"]
            conflict=point_evidence(masks["conflict"])
            sem_floor=point_evidence(masks["semantic_floor"])
            # [v3.5 추가] 관측 바닥 정책: 바닥 점은 지도에서 유지. 상부 결측만으로 버리지 않는다.
            # 실제 더 가까운 관측/기하 장애물/SCNN 충돌은 남기고, 높이 전체 확인 마스크는 따로 보존.
            free=(observed&~occluded&~conflict) if g["free_space_policy"]=="observed_floor" else volume_clear&~conflict
            free &= ~uncertain
            in_view |= observed|occupied  # 경계 셀의 실제 측정은 시야 안쪽 증거로 우선한다.
            free &= in_view
            # [v3.5] 요청한 SCNN 바닥 보완. 장애물/불확실 후보/실제 가림은 우선한다.
            projected,semantic_free_info=projected_floor_free(self.grid,semantic,frame_id,floor,self.T,
                cal,raw,in_view,occupied|uncertain|conflict,self.cfg,
                measured_free=free&~occupied,continuity_out=continuity_added)
            semantic_added=projected&~free&~occupied&~occluded&~conflict
            continuity_added &= semantic_added
            free |= semantic_added
            evidence[semantic_added]|=64
            evidence[observed]|=1
            evidence[sem_floor]|=2
            evidence[volume_clear]|=4
            evidence[low_cells]|=8
            evidence[conflict]|=16
            evidence[held_cells]|=32
            fusion_info.update(semantic_floor_points=int(masks["semantic_floor"].sum()),
                semantic_low_obstacle_points=int((low&supported_low).sum()),
                semantic_low_obstacle_cells=int(low_cells.sum()),conflict_cells=int(conflict.sum()))
            self.grid.free_diagnostics.update(status="OK",obstacle_conflict_cells=int((free&occupied).sum()),
                current_free_evidence_cells=int((free&~occupied).sum()),free_space_policy=g["free_space_policy"],
                observed_floor_cells=int(observed.sum()),semantic_floor_cells=int(sem_floor.sum()),
                semantic_conflict_cells=int(conflict.sum()),
                observed_free_without_full_volume_cells=int((free&~occupied&~volume_clear).sum()))
            if self.cfg["mapping"]["mode"]=="camera_relative":
                self.grid.update_current(occupied,free,c.received_monotonic)
            else:
                self.grid.update(occupied,free,c.received_monotonic)
        else:
            # 바닥 검출 실패 시 새로운 장애물/자유 공간 증거를 지도에 통합하지 않는다.
            self.reviewer.reset()
            self.grid.clear_streak.fill(0)
            self.grid.age(c.received_monotonic)
        # [v3.5 추가] 누적 모드에서도 현재 시야 밖 자료를 자유 공간으로 노출하지 않는다.
        self.grid.physical[~in_view]=-1
        risk,rear_unknown=risk_layers(self.grid.physical,g["resolution_m"],self.cfg["robot"],
            (g["x_min_m"],g["y_min_m"]),self.T[:2,3])
        self.grid.physical[rear_unknown]=-1
        semantic_added &= ~rear_unknown
        evidence[~semantic_added] &= np.uint8(191)  # bit64만 제거
        # [v3.5] 연결 추정은 수치 지도에 반영하되 bit128로 직접 측정과 구분한다.
        continuity_added &= semantic_added&(self.grid.physical==0)
        evidence[continuity_added] |= 128
        cost=risk.copy();cost[self.grid.physical==-1]=255
        grid_ms=(time.monotonic()-grid_start)*1000
        diagnostics=dict(self.grid.free_diagnostics,
            mapped_free_cells=int((self.grid.physical==0).sum()),physical_obstacle_cells=int((self.grid.physical==100).sum()),
            green_cells=int((cost==0).sum()),forbidden_cells=int((cost==253).sum()),
            inflation_cells=int(((cost>0)&(cost<253)).sum()),unknown_cells=int((cost==255).sum()),
            mapping_mode=self.cfg["mapping"]["mode"],outside_view_cells=int((~in_view).sum()),
            in_view_unknown_cells=int(((cost==255)&in_view).sum()),horizontal_fov=fov_info,fusion=fusion_info,obstacle_support=support_info,
            candidate_review=review_info,
            rear_policy_unknown_cells=int((rear_unknown&in_view).sum()),
            robot=copy.deepcopy(self.cfg["robot"]),
            semantic_free_added_cells=int(semantic_added.sum()),
            semantic_free_continuity_cells=int(continuity_added.sum()),
            semantic_free=semantic_free_info,
            measured_free_cells=int(((self.grid.physical==0)&~semantic_added).sum()),
            risk_forbidden_cells=int(((risk==253)&in_view).sum()),
            risk_inflation_cells=int(((risk>0)&(risk<253)&in_view).sum()),
            unknown_with_risk_cells=int(((self.grid.physical==-1)&(risk>0)&in_view).sum()))
        h,w=c.color.shape[:2]
        color_classes=np.full((h,w),255,np.uint8)
        if len(points):
            ids=np.flatnonzero(color_ok)
            u,v=np.rint(uv[ids]).astype(np.int32).T
            order=np.argsort(color_z[ids])[::-1]
            # [v3.5 추가] 표시 격자는 Color 픽셀. 더 가까운 점을 마지막에 써 가려짐을 처리한다.
            color_classes[v[order],u[order]]=classes[ids[order]]
        self.processed+=1
        done=time.monotonic()
        stats={"frame_id":frame_id,
               "source_id":packet.source_id,"received_at":c.received_at,
               "completed_monotonic":done,"input_age_ms":max(0,age*1000),
               "processing_ms":(done-start)*1000,"costmap_ms":grid_ms,
               "stage_ms":{"semantics":(semantics_done-start)*1000,
                           "geometry_floor":(grid_start-semantics_done)*1000,
                           "costmap":grid_ms,"finalize":(done-grid_start)*1000-grid_ms},
               "latency_ms":None if packet.replay else max(0,(done-c.received_monotonic)*1000),
               "valid_depth_fraction":float(valid.mean()),"point_count":len(points),
               "filtered_removed_pixels":int(valid.sum()-keep.sum()),
               "depth_interpolated_pixels":0,"map_updated":floor.ok,
               "replay":packet.replay,"scnn":semantic.record(),"map_diagnostics":diagnostics}
        return Snapshot(packet,points.astype(np.float32),rgb,color_ok,classes,indices,filtered,valid,
                        color_classes,floor,self.grid.physical.copy(),cost,self.T.copy(),
                        self.grid.metadata(),stats,semantic,in_view,volume_clear,evidence,
                        point_labels,point_confidence,rejected_native,semantic_conflict_native,
                        risk,local_review_native,temporal_review_native,rear_unknown,copy.deepcopy(self.cfg),semantic_added,continuity_added)
