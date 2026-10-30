"""[v3.5] 너비/높이 숫자 입력을 공통 처리. 실제 설정은 worker 프레임 경계에서 적용."""
from .robot import derive_robot,validate_robot_height


class RobotDimensionInput:
    # [추가] 두 입력칸의 편집/검증/대기 상태를 하나의 구현으로 관리한다.
    def __init__(self,value,shortcut,validator,hint):
        self.applied=float(value);self.text=f"{value:.2f}";self.editing=False
        self.shortcut=shortcut;self.validator=validator;self.hint=hint
        self.select_all=False;self.request=None;self.waiting=None
        self.message=f"Click / {shortcut.upper()}: edit | Enter: apply"
    def begin(self):
        self.editing=True;self.select_all=True
        value=self.applied if self.waiting is None else self.waiting
        self.text=f"{value:.2f}";self.message=self.hint+" | Esc: cancel"
    def cancel(self):
        self.editing=False;self.select_all=False
        value=self.applied if self.waiting is None else self.waiting
        self.text=f"{value:.2f}";self.message="Edit cancelled"
    def key(self,k):
        if not self.editing:
            if k in (ord(self.shortcut.lower()),ord(self.shortcut.upper())):self.begin();return True
            return False
        if k in (-1,255):return False
        if k==27:self.cancel();return True
        if k in (10,13):
            try:value=self.validator(self.text)
            except (ValueError,TypeError):
                self.message="Invalid: "+self.hint;return True
            self.request=value;self.waiting=value;self.editing=False;self.message="Queued: next processed frame"
            return True
        if k in (8,127):
            self.text="" if self.select_all else self.text[:-1];self.select_all=False
        elif 0<=k<256 and chr(k) in "0123456789." and (self.select_all or len(self.text)<10):
            self.text=chr(k) if self.select_all else self.text+chr(k);self.select_all=False
        return True
    def take_request(self):
        value=self.request;self.request=None;return value
    def sync(self,value):
        self.applied=float(value)
        if self.waiting is not None and abs(value-self.waiting)<1e-9:
            self.waiting=None;self.message=f"Applied | Click / {self.shortcut.upper()}: edit"
        if not self.editing and self.waiting is None:self.text=f"{value:.2f}"


class RobotWidthInput(RobotDimensionInput):
    def __init__(self,width):
        super().__init__(width,"b",lambda value:derive_robot(value)["width_m"],"Width 0.05..3.00 m")


class RobotHeightInput(RobotDimensionInput):
    def __init__(self,height,minimum=0.05):
        super().__init__(height,"h",lambda value:validate_robot_height(value,minimum),
                         f"Height >{minimum:g}..3.00 m")
