"""Fast-SCNN 모델 정의 (mmseg 버전을 순수 PyTorch 로 옮김)
- mmseg/mmcv 설치 없이 체크포인트를 읽으려고 만듦. 모듈 이름을 mmseg 와 똑같이 둬서 state_dict 가 strict 로 들어감
- mmseg 1.2.2 원본과 같은 가중치로 비교 → 로짓 차이 0.0 확인함
- forward(x) 는 (1, 클래스수, H/8, W/8) 로짓. 입력 크기로 키우기 + argmax 는 export_onnx 의 ExportWrapper 가 함
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- mmcv 기본 블록 ----------

class ConvBNAct(nn.Module):
    """mmcv ConvModule: conv → bn → relu. bn 있으면 conv bias 안 씀"""

    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.01)   # 학습 설정의 momentum 과 같음
        self.activate = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """mmcv DepthwiseSeparableConvModule: depthwise 3x3 → pointwise 1x1. depthwise 뒤엔 활성화 없음"""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, dw_act=False, pw_act=True):
        super().__init__()
        self.depthwise_conv = ConvBNAct(in_ch, in_ch, kernel, stride, padding, groups=in_ch, act=dw_act)
        self.pointwise_conv = ConvBNAct(in_ch, out_ch, 1, act=pw_act)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class InvertedResidual(nn.Module):
    """MobileNetV2 블록: 1x1 확장 → 3x3 depthwise → 1x1 축소. stride 1 이고 채널 같으면 입력을 더함"""

    def __init__(self, in_ch, out_ch, stride, expand_ratio=6):
        super().__init__()
        hidden_ch = int(round(in_ch * expand_ratio))   # 중간에 넓어지는 채널
        self.use_res_connect = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden_ch, 1, act=True))
        layers.append(ConvBNAct(hidden_ch, hidden_ch, 3, stride, padding=1, groups=hidden_ch, act=True))
        layers.append(ConvBNAct(hidden_ch, out_ch, 1, act=False))   # 마지막은 활성화 없음
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


# ---------- Fast-SCNN ----------

class LearningToDownsample(nn.Module):
    """입력 → 1/8. conv(s2) → dsconv(s2) → dsconv(s2)"""

    def __init__(self, in_ch=3, dw_channels=(32, 48), out_ch=64):
        super().__init__()
        self.conv = ConvBNAct(in_ch, dw_channels[0], 3, stride=2, padding=1, act=True)
        self.dsconv1 = DSConv(dw_channels[0], dw_channels[1], 3, stride=2, padding=1)
        self.dsconv2 = DSConv(dw_channels[1], out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.dsconv2(self.dsconv1(self.conv(x)))


class PPM(nn.ModuleList):
    """피라미드 풀링. 스케일별로 풀링 → 1x1 conv → 원래 크기로 업샘플
    480x640 입력이면 15x20 이 들어와서 2·3·6 풀링이 ONNX 로 안 나감 → export_onnx.freeze_adaptive_pools 가 바꿔 줌"""

    def __init__(self, pool_scales=(1, 2, 3, 6), in_ch=128, out_ch=32):
        super().__init__()
        for scale in pool_scales:
            self.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), ConvBNAct(in_ch, out_ch, 1, act=True)))

    def forward(self, x):
        size = x.shape[2:]
        return [F.interpolate(branch(x), size=size, mode="bilinear", align_corners=False) for branch in self]


class GlobalFeatureExtractor(nn.Module):
    """1/8 → 1/32. bottleneck 3묶음 → PPM → 3x3 conv"""

    def __init__(self, in_ch=64, block_channels=(64, 96, 128), out_ch=128, expand_ratio=6,
                 num_blocks=(3, 3, 3), strides=(2, 2, 1), pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.bottleneck1 = self._make_layer(in_ch, block_channels[0], num_blocks[0], strides[0], expand_ratio)
        self.bottleneck2 = self._make_layer(block_channels[0], block_channels[1], num_blocks[1], strides[1], expand_ratio)
        self.bottleneck3 = self._make_layer(block_channels[1], block_channels[2], num_blocks[2], strides[2], expand_ratio)
        self.ppm = PPM(pool_scales, block_channels[2], block_channels[2] // 4)             # 스케일당 32채널
        self.out = ConvBNAct(block_channels[2] * 2, out_ch, 3, padding=1, act=True)        # 128 + 32×4 = 256 → 128

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks, stride, expand_ratio):
        layers = [InvertedResidual(in_ch, out_ch, stride, expand_ratio)]      # 첫 블록만 stride 적용
        for _ in range(1, num_blocks):
            layers.append(InvertedResidual(out_ch, out_ch, 1, expand_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck3(self.bottleneck2(self.bottleneck1(x)))
        x = torch.cat([x, *self.ppm(x)], dim=1)
        return self.out(x)


class FeatureFusionModule(nn.Module):
    """1/8 특징(higher) + 1/32 특징(lower). lower 를 4배 키워서 더하고 relu"""

    def __init__(self, higher_in_ch=64, lower_in_ch=128, out_ch=128):
        super().__init__()
        self.dwconv = ConvBNAct(lower_in_ch, out_ch, 3, padding=1, groups=out_ch, act=True)   # depthwise 3x3
        self.conv_lower_res = ConvBNAct(out_ch, out_ch, 1, act=False)
        self.conv_higher_res = ConvBNAct(higher_in_ch, out_ch, 1, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, higher, lower):
        lower = F.interpolate(lower, size=higher.shape[2:], mode="bilinear", align_corners=False)
        lower = self.conv_lower_res(self.dwconv(lower))
        higher = self.conv_higher_res(higher)
        return self.relu(higher + lower)


class FastSCNN(nn.Module):
    """백본. 출력 (higher 1/8·64ch, lower 1/32·128ch, fusion 1/8·128ch). 추론은 fusion 만 씀"""

    def __init__(self, in_ch=3, downsample_dw_channels=(32, 48), global_in_ch=64,
                 global_block_channels=(64, 96, 128), global_block_strides=(2, 2, 1),
                 global_out_ch=128, fusion_out_ch=128):
        super().__init__()
        self.learning_to_downsample = LearningToDownsample(in_ch, downsample_dw_channels, global_in_ch)
        self.global_feature_extractor = GlobalFeatureExtractor(global_in_ch, global_block_channels, global_out_ch,
                                                               strides=global_block_strides)
        self.feature_fusion = FeatureFusionModule(global_in_ch, global_out_ch, fusion_out_ch)

    def forward(self, x):
        higher = self.learning_to_downsample(x)
        lower = self.global_feature_extractor(higher)
        fusion = self.feature_fusion(higher, lower)
        return higher, lower, fusion


class DepthwiseSeparableFCNHead(nn.Module):
    """헤드. DSConv 2개 → dropout → 1x1 conv(클래스 수)"""

    def __init__(self, in_ch=128, channels=128, num_classes=3, num_convs=2, dropout_ratio=0.1):
        super().__init__()
        convs = [DSConv(in_ch, channels, 3, padding=1)]
        for _ in range(1, num_convs):
            convs.append(DSConv(channels, channels, 3, padding=1))
        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)   # 얘만 bias 있음

    def forward(self, x):
        return self.conv_seg(self.dropout(self.convs(x)))


class FastSCNNSegmentor(nn.Module):
    """mmseg EncoderDecoder 에서 추론에 필요한 부분만. backbone → decode_head(fusion)"""

    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = FastSCNN()
        self.decode_head = DepthwiseSeparableFCNHead(num_classes=num_classes)

    def forward(self, x):
        higher, lower, fusion = self.backbone(x)
        return self.decode_head(fusion)   # (N, 클래스수, H/8, W/8)


# ---------- export_onnx 가 부르는 함수 ----------

def build_model(num_classes: int = 3) -> nn.Module:
    return FastSCNNSegmentor(num_classes)


def adapt_state_dict(state_dict: dict) -> dict:
    """보조 헤드(auxiliary_head.*) 키 제거. 학습 때만 쓰던 거라 추론 모델엔 없음"""
    aux_keys = [key for key in state_dict if key.startswith("auxiliary_head.")]
    if aux_keys:
        print(f"[adapt] auxiliary_head 키 {len(aux_keys)}개 제외 (추론에 안 씀)")
    return {key: value for key, value in state_dict.items() if key not in aux_keys}


if __name__ == "__main__":
    # 구조 확인: python -m deploy.fastscnn_mmseg [best.pth]
    import sys
    from deploy.export_onnx import load_state_dict, freeze_adaptive_pools

    model = build_model(3).eval()
    print(f"파라미터 {sum(p.numel() for p in model.parameters()):,}개")
    if len(sys.argv) > 1:
        print(model.load_state_dict(adapt_state_dict(load_state_dict(sys.argv[1])), strict=True))

    x = torch.rand(1, 3, 480, 640)
    with torch.no_grad():
        before = model(x)
    print("로짓", tuple(before.shape))
    print("static pool 교체:", freeze_adaptive_pools(model, x))
    with torch.no_grad():
        after = model(x)
    print("교체 전후 최대 차이", float((before - after).abs().max()))
