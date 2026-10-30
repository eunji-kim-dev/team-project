"""로봇 너비 하나에서 비용 거리를 유도한다. 입력 변경은 프레임 경계에서 적용."""
import copy,math
MIN_WIDTH_M=0.05
MAX_WIDTH_M=3.0

def derive_robot(width,template=None):
    if isinstance(width,bool):raise ValueError("Robot width must be a number")
    width=float(width)
    if not math.isfinite(width) or not MIN_WIDTH_M<=width<=MAX_WIDTH_M:
        raise ValueError("Robot width must be 0.05..3.00 m")
    robot=copy.deepcopy(template or {})
    forbidden=width/2+0.05
    robot.update(width_m=width,radius_m=width/2,safety_margin_m=0.05,
                 forbidden_radius_m=forbidden,inflation_extra_m=0.15,
                 inflation_radius_m=forbidden+0.15)
    robot.setdefault("cost_scaling_per_m",6)
    robot.setdefault("cost_direction","camera_visible")
    return robot


def validate_robot_height(height,minimum=0.05):
    # [추가] 실제 로봇 키: 카메라 설치 높이와 별개. 잘못된 입력은 계산 전에 거절한다.
    if isinstance(height,bool):raise ValueError("Robot height must be a number")
    height=float(height)
    if not math.isfinite(height) or not minimum<height<=3.0:
        raise ValueError(f"Robot height must be >{minimum:g} and <=3.00 m")
    return height
