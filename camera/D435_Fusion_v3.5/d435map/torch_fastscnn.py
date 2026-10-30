"""[v3.5 추가] 팀 MMSeg FastSCNN 구성에 대응하는 PyTorch 추론 전용 모델.
공식 구조 참고: OpenMMLab mmsegmentation v1.2.2 FastSCNN / sep_fcn_head.
모델 입출력 설정: models/standard_fastscnn.json. 재학습하지 않는다.
"""
from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F

def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ConvBN(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, groups=1, activate=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel//2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=1e-5, momentum=.01)
        self.activate = activate
    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x) if self.activate else x

class Separable(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        # 학습 코드의 dw_act_cfg=None: depthwise 뒤에 ReLU를 넣지 않는다.
        self.depthwise_conv = ConvBN(cin, cin, 3, stride, cin, activate=False)
        self.pointwise_conv = ConvBN(cin, cout)
    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))

class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBN(3,32,3,2)
        self.dsconv1 = Separable(32,48,2)
        self.dsconv2 = Separable(48,64,2)
    def forward(self,x):
        return self.dsconv2(self.dsconv1(self.conv(x)))

class Residual(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.skip = stride==1 and cin==cout
        self.conv = nn.Sequential(ConvBN(cin,cin*6), ConvBN(cin*6,cin*6,3,stride,cin*6),
                                  ConvBN(cin*6,cout,activate=False))
    def forward(self,x):
        y=self.conv(x)
        return x+y if self.skip else y

class GlobalFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        def group(cin,cout,stride):
            return nn.Sequential(Residual(cin,cout,stride), Residual(cout,cout,1),Residual(cout,cout,1))
        self.bottleneck1=group(64,64,2)
        self.bottleneck2=group(64,96,2)
        self.bottleneck3=group(96,128,1)
        self.ppm=nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(k),ConvBN(128,32)) for k in (1,2,3,6)])
        self.out=ConvBN(256,128,3)
    def forward(self,x):
        x=self.bottleneck3(self.bottleneck2(self.bottleneck1(x)))
        return self.out(torch.cat([x]+[resize(branch(x),x.shape[-2:]) for branch in self.ppm],dim=1))

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwconv=ConvBN(128,128,3,groups=128)
        self.conv_lower_res=ConvBN(128,128,activate=False)
        self.conv_higher_res=ConvBN(64,128,activate=False)
    def forward(self, high,low):
        low=self.conv_lower_res(self.dwconv(resize(low,high.shape[-2:])))
        return F.relu(self.conv_higher_res(high)+low)

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.learning_to_downsample=Downsample()
        self.global_feature_extractor=GlobalFeatures()
        self.feature_fusion=Fusion()
    def forward(self,x):
        high=self.learning_to_downsample(x)
        return self.feature_fusion(high,self.global_feature_extractor(high))

class Head(nn.Module):
    def __init__(self,cin=128,aux=False):
        super().__init__()
        channels=32 if aux else 128
        self.convs=nn.Sequential(ConvBN(cin,channels,3)) if aux else nn.Sequential(Separable(128,128),Separable(128,128))
        self.dropout=nn.Dropout2d(.1)
        self.conv_seg=nn.Conv2d(channels,3,1)
    def forward(self,x):
        return self.conv_seg(self.dropout(self.convs(x)))

class TeamFastSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=Backbone()
        self.decode_head=Head()
        # 보조 헤드는 학습용. 전 가중치의 엄격한 키/크기 검사에 포함하되 추론에는 쓰지 않는다.
        self.auxiliary_head=nn.ModuleList([Head(128,True),Head(64,True)])
    def forward(self,x):
        return resize(self.decode_head(self.backbone(x)),x.shape[-2:])

class TorchBackend:
    def __init__(self,path,threads=2):
        torch.set_num_threads(threads)
        # [수정] 텐서 가중치와 일반적인 학습 체크포인트 래퍼를 모두 지원한다.
        # 임의 객체를 실행하는 역직렬화는 사용하지 않고 모델 키/크기는 엄격하게 검사한다.
        state=torch.load(path,map_location="cpu",weights_only=True)
        if isinstance(state,Mapping):
            if "state_dict" in state:state=state["state_dict"]
            elif "model_state_dict" in state:state=state["model_state_dict"]
        if not isinstance(state,Mapping) or not state or not all(
                isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
            raise ValueError("Expected Team FastSCNN tensor state_dict (or state_dict/model_state_dict checkpoint)")
        if all(k.startswith("module.") for k in state):
            state={k.removeprefix("module."):v for k,v in state.items()}
        self.model=TeamFastSCNN()
        self.model.load_state_dict(state,strict=True)
        # [v3.5] 전체 체크포인트 검증 후 추론에 쓰지 않는 학습 보조 헤드만 해제.
        self.model.auxiliary_head=nn.ModuleList()
        del state
        self.model.eval()
        # 첫 프레임 지연을 줄이도록 카메라 시작 전에 커널을 준비한다.
        with torch.inference_mode():
            self.model(torch.zeros(1,3,480,640))
    def __call__(self,array):
        with torch.inference_mode():
            return self.model(torch.from_numpy(array)).numpy()
