"""QAT 용 가짜 양자화 (PyTorch 만 씀, 외부 라이브러리 없음)
- 학습은 float 로 하되 값을 int8 격자에 붙였다 떼는 연산을 끼움 → 모델이 양자화 오차에 익숙해짐
- 역전파는 반올림 없는 것처럼 통과시킴 (STE)
- ONNX 로 내보내면 이 연산이 QuantizeLinear → DequantizeLinear (Q/DQ) 가 됨 → TensorRT 가 캘리브레이션 없이 INT8 엔진 만듦
- TensorRT 규칙: 대칭 양자화, zero_point 0, 활성화는 per-tensor, 가중치는 per-channel(axis 0), Q/DQ 는 conv 입력·가중치에만
- 쓰는 순서: fuse_conv_bn → quantize_convs → calibrate → 학습 → freeze_for_export → export
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QMAX = 127.0    # scale = |x|max / 127
QMIN = -128.0   # clamp 범위는 ONNX/TensorRT 와 같게 [-128, 127]


class _FakeQuant(torch.autograd.Function):
    """x → round(x/scale) 을 clamp 한 뒤 다시 scale 곱함. 기울기는 그대로 통과"""

    @staticmethod
    def forward(ctx, x, scale, axis, num_channels):
        if axis < 0:
            s = scale                                                    # per-tensor
        else:
            s = scale.view([-1 if dim == axis else 1 for dim in range(x.dim())])   # per-channel 브로드캐스트
        return torch.clamp(torch.round(x / s), QMIN, QMAX) * s

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

    @staticmethod
    def symbolic(g, x, scale, axis, num_channels):
        # ONNX 로 나갈 때 모양. zero_point 는 int8 0
        if axis < 0:
            zero_point = g.op("Constant", value_t=torch.zeros((), dtype=torch.int8))
            quantized = g.op("QuantizeLinear", x, scale, zero_point)
            return g.op("DequantizeLinear", quantized, scale, zero_point)
        zero_point = g.op("Constant", value_t=torch.zeros(num_channels, dtype=torch.int8))
        quantized = g.op("QuantizeLinear", x, scale, zero_point, axis_i=axis)
        return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)


def fake_quant(x, scale, axis=-1):
    num_channels = int(scale.shape[0]) if axis >= 0 else 0   # ONNX zero_point 길이
    return _FakeQuant.apply(x, scale, axis, num_channels)


class ActQuantizer(nn.Module):
    """활성화(conv 입력) 양자화기. 배치별 |x| 최댓값을 EMA 또는 누적 max로 모아 scale을 정한 뒤 고정."""

    def __init__(self, momentum: float = 0.9, method: str = "ema"):
        super().__init__()
        if method not in ("ema", "max"):
            raise ValueError(f"지원하지 않는 캘리브레이션 방식: {method}")
        self.register_buffer("scale", torch.tensor(1.0))   # 0-d → ONNX 에서 per-tensor 스칼라
        self.register_buffer("amax", torch.tensor(0.0))    # EMA 또는 누적 |x| 최댓값
        self.register_buffer("n_obs", torch.tensor(0))     # 본 배치 수
        self.momentum = momentum
        self.method = method
        self.calibrating = False   # True: scale 재는 중 (양자화 안 함)
        self.enabled = True        # False: 이 conv 의 양자화(입력·가중치 둘 다) 끄고 float 통과 (비교용)

    def forward(self, x):
        if self.calibrating:
            current_amax = x.detach().abs().amax()
            if self.n_obs.item() == 0:
                self.amax.copy_(current_amax)
            elif self.method == "max":
                self.amax.copy_(torch.maximum(self.amax, current_amax))
            else:
                self.amax.mul_(self.momentum).add_(current_amax * (1 - self.momentum))
            self.n_obs += 1
            self.scale.copy_((self.amax / QMAX).clamp_min(1e-8))
            return x
        if not self.enabled:
            return x
        return fake_quant(x, self.scale, axis=-1)


class QuantConv2d(nn.Module):
    """nn.Conv2d 감싸서 입력(per-tensor)·가중치(per-channel)에 가짜 양자화 끼움"""

    def __init__(self, conv: nn.Conv2d, calib_method: str = "ema"):
        super().__init__()
        self.conv = conv
        self.input_quant = ActQuantizer(method=calib_method)
        self.register_buffer("w_scale", torch.ones(conv.out_channels))   # export 때 고정되는 가중치 scale
        self.w_scale_frozen = False   # 학습 중엔 매번 가중치에서 계산, export 땐 상수
        self.act_quant_enabled = True      # False: 입력(활성화) 양자화만 끔 (민감도 분석용)
        self.weight_quant_enabled = True   # False: 가중치 양자화만 끔 (민감도 분석용). 둘 다 끄려면 input_quant.enabled = False

    def weight_scale(self) -> torch.Tensor:
        if self.w_scale_frozen:
            return self.w_scale
        amax_per_channel = self.conv.weight.detach().abs().amax(dim=(1, 2, 3))
        return (amax_per_channel / QMAX).clamp_min(1e-8)

    def forward(self, x):
        quantizer = self.input_quant
        if quantizer.calibrating or (quantizer.enabled and self.act_quant_enabled):
            x = quantizer(x)
        weight_off = quantizer.calibrating or not quantizer.enabled or not self.weight_quant_enabled
        weight = self.conv.weight if weight_off else fake_quant(self.conv.weight, self.weight_scale(), axis=0)
        return F.conv2d(x, weight, self.conv.bias, self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)


# ---------- 모델 변환 ----------

@torch.no_grad()
def fuse_conv_bn(model: nn.Module) -> int:
    """conv 뒤의 BN 을 conv 가중치·bias 에 접어 넣고 BN 은 Identity 로. 접은 개수 반환
    새 w = w · γ/√(var+ε) (채널별), 새 b = β − mean · γ/√(var+ε)"""
    fused_count = 0
    for module in model.modules():
        conv = getattr(module, "conv", None)
        bn = getattr(module, "bn", None)
        if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)):
            continue
        inv_std = bn.weight / torch.sqrt(bn.running_var + bn.eps)   # γ/√(var+ε)
        fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                          conv.padding, conv.dilation, conv.groups, bias=True).to(conv.weight.device)   # 모델이 GPU 면 GPU 에
        fused.weight.copy_(conv.weight * inv_std.view(-1, 1, 1, 1))
        old_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused.bias.copy_(bn.bias + (old_bias - bn.running_mean) * inv_std)
        module.conv = fused
        module.bn = nn.Identity()
        fused_count += 1
    return fused_count


def quantize_convs(model: nn.Module, skip: tuple[str, ...] = (), *, calib_method: str = "ema") -> int:
    """모든 nn.Conv2d 를 QuantConv2d 로 교체. skip 접두사는 건너뜀. 바꾼 개수 반환"""
    if calib_method not in ("ema", "max"):
        raise ValueError(f"지원하지 않는 캘리브레이션 방식: {calib_method}")
    replaced = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Conv2d) and not any(full_name.startswith(prefix) for prefix in skip):
                setattr(parent, child_name, QuantConv2d(child, calib_method=calib_method))
                replaced += 1
    return replaced


def quantizers(model: nn.Module) -> list[ActQuantizer]:
    return [module for module in model.modules() if isinstance(module, ActQuantizer)]


def set_calibrating(model: nn.Module, on: bool) -> None:
    for quantizer in quantizers(model):
        quantizer.calibrating = on


def named_quant_convs(model: nn.Module) -> list[tuple[str, QuantConv2d]]:
    """(모듈 이름, QuantConv2d) 목록. 이름은 state_dict 키의 앞부분 (예: core.backbone.learning_to_downsample.conv.conv)"""
    return [(name, module) for name, module in model.named_modules() if isinstance(module, QuantConv2d)]


def set_quant_enabled(model: nn.Module, on: bool) -> None:
    """False 면 가짜 양자화(입력·가중치) 전부 끄고 float 로 돎. 한쪽만 끄려면 QuantConv2d.act_quant_enabled / weight_quant_enabled"""
    for quantizer in quantizers(model):
        quantizer.enabled = on


@torch.no_grad()
def calibrate(model: nn.Module, batches, device="cpu") -> None:
    """batches 를 흘려서 활성화 scale 정함"""
    was_training = model.training
    model.eval()
    set_calibrating(model, True)
    for x in batches:
        model(x.to(device))
    set_calibrating(model, False)
    model.train(was_training)


@torch.no_grad()
def freeze_for_export(model: nn.Module) -> None:
    """가중치 scale 을 buffer 에 고정 → ONNX 에서 Q/DQ scale 이 상수(initializer)가 됨 (TensorRT 요구사항)"""
    for module in model.modules():
        if isinstance(module, QuantConv2d):
            module.w_scale.copy_(module.weight_scale())
            module.w_scale_frozen = True


def quant_summary(model: nn.Module) -> str:
    quant_convs = [module for module in model.modules() if isinstance(module, QuantConv2d)]
    scales = torch.stack([module.input_quant.scale for module in quant_convs])
    return (f"QuantConv2d {len(quant_convs)}개, 입력 scale 최소 {scales.min():.4g} / "
            f"중간 {scales.median():.4g} / 최대 {scales.max():.4g}")
